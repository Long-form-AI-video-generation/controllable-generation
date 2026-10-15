"""Immutable, hash-verified prepared-control bundles."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import CANONICAL_EXPERT_ORDER, canonicalize_expert_names


FORMAT_VERSION = 1
CONTROL_ARRAY_ORDER = CANONICAL_EXPERT_ORDER
CONTROL_FILE_NAME = "controls.npz"
METADATA_FILE_NAME = "control_metadata.json"


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_content_sha256(name: str, array: np.ndarray) -> str:
    """Hash control identity independently of the NPZ ZIP layout."""

    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _validate_control_array(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 5 or array.shape[:2] != (1, 1):
        raise ValueError(
            f"{name} must have shape [1,1,T,H,W], got {tuple(array.shape)}"
        )
    if any(dimension <= 0 for dimension in array.shape[2:]):
        raise ValueError(f"{name} has non-positive temporal or spatial dimensions")
    if name == "depth":
        if array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError("depth must be finite float32")
    elif name == "canny":
        if array.dtype != np.uint8 or not np.isin(array, (0, 1)).all():
            raise ValueError("canny must be binary uint8")
    elif name == "mask":
        if array.dtype != np.uint8 or int(array.max()) > 149:
            raise ValueError("mask must be uint8 ADE20K labels in [0,149]")
    else:
        raise ValueError(f"unknown control array {name!r}")
    return np.ascontiguousarray(array)


def _validate_controls(controls: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    names = canonicalize_expert_names(list(controls))
    temporal_lengths: set[int] = set()
    result: dict[str, np.ndarray] = {}
    for name in names:
        array = _validate_control_array(name, controls[name])
        temporal_lengths.add(int(array.shape[2]))
        result[name] = array
    if len(temporal_lengths) != 1:
        raise ValueError("every control array must have the same temporal length")
    return result


@dataclass(frozen=True)
class PreparedControlBundle:
    """Fully validated bundle data ready for CPU-to-GPU inference transfer."""

    root: Path
    controls: Mapping[str, np.ndarray]
    metadata: Mapping[str, Any]

    @property
    def artifact_sha256(self) -> str:
        return str(self.metadata["controls_file_sha256"])


def write_prepared_control_bundle(
    output_dir: str | Path,
    *,
    controls: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
) -> PreparedControlBundle:
    """Atomically create one immutable prepared-control bundle.

    The caller supplies preprocessing identity and source-frame metadata. This
    function adds control-array and file hashes, then validates the final
    directory exactly as a later inference run will.
    """

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(output)
    validated = _validate_controls(controls)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        controls_path = temporary / CONTROL_FILE_NAME
        with controls_path.open("wb") as stream:
            np.savez_compressed(
                stream,
                **{name: validated[name] for name in CONTROL_ARRAY_ORDER if name in validated},
            )
        details = {
            name: {
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "content_sha256": array_content_sha256(name, array),
            }
            for name, array in validated.items()
        }
        final_metadata = dict(metadata)
        final_metadata.update(
            {
                "format_version": FORMAT_VERSION,
                "control_names": list(validated),
                "controls": details,
                "controls_file_sha256": sha256_file(controls_path),
            }
        )
        (temporary / METADATA_FILE_NAME).write_bytes(_canonical_json(final_metadata))
        os.replace(temporary, output)
    except BaseException:
        if temporary.exists():
            for child in temporary.iterdir():
                child.unlink()
            temporary.rmdir()
        raise
    return load_prepared_control_bundle(output)


def load_prepared_control_bundle(
    bundle_dir: str | Path,
    *,
    expected_experts: Sequence[str] | None = None,
    expected_frame_num: int | None = None,
) -> PreparedControlBundle:
    """Load and verify a bundle before WAN or any control model is built."""

    root = Path(bundle_dir)
    metadata_path = root / METADATA_FILE_NAME
    controls_path = root / CONTROL_FILE_NAME
    if not metadata_path.is_file() or not controls_path.is_file():
        raise FileNotFoundError("prepared controls require controls.npz and control_metadata.json")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("prepared-control metadata is not valid JSON") from error
    if metadata.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported prepared-control format version")
    if metadata.get("controls_file_sha256") != sha256_file(controls_path):
        raise ValueError("prepared controls file SHA-256 mismatch")

    with np.load(controls_path, allow_pickle=False) as archive:
        names = tuple(archive.files)
        expected_names = tuple(
            name for name in CONTROL_ARRAY_ORDER if name in metadata.get("control_names", [])
        )
        if names != expected_names:
            raise ValueError(
                f"prepared controls use unexpected key order {names}; expected {expected_names}"
            )
        controls = {name: np.asarray(archive[name]) for name in names}
    validated = _validate_controls(controls)
    if tuple(validated) != tuple(metadata.get("control_names", [])):
        raise ValueError("prepared-control names do not match metadata")

    for name, array in validated.items():
        saved = metadata.get("controls", {}).get(name)
        expected = {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "content_sha256": array_content_sha256(name, array),
        }
        if saved != expected:
            raise ValueError(f"prepared {name} metadata or content hash mismatch")

    if expected_experts is not None:
        expected_names = canonicalize_expert_names(list(expected_experts))
        if tuple(validated) != expected_names:
            raise ValueError(
                f"prepared experts {tuple(validated)} do not match requested {expected_names}"
            )
    if expected_frame_num is not None:
        actual = next(iter(validated.values())).shape[2]
        if actual != int(expected_frame_num):
            raise ValueError(
                f"prepared controls use {actual} frames, expected {expected_frame_num}"
            )
    return PreparedControlBundle(root=root, controls=validated, metadata=metadata)
