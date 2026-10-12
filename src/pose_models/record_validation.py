"""Structural validation for ``pose-pilot-v1`` pose records.

The checks follow the pilot pose record contract: exact field sets, literal
identifiers, canonical joint order, normalized coordinates, frame ordering and
the joint validity rule.  They check structure only; they do not judge whether
a pose is anatomically plausible, whether hashes match real artifacts, or
whether ``source_frame_index`` values equal the output of
``src/data/frame_sampling.py``.

This module uses only the standard library so records can be checked without
model weights, videos or third-party packages.

Usage::

    python -m src.pose_models.record_validation path/to/record.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "pose-pilot-v1"
COORDINATE_SPACE = "original_frame_normalized_pixel_centers"
SAMPLING_POLICY = "shared_select_frame_indices_v1"
TIMING = "constant_fps"
PARTITIONS = ("train", "val", "test")
SUBJECT_STATUSES = ("selected", "absent", "ambiguous")

CANONICAL_JOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

RECORD_KEYS = frozenset(
    {
        "schema_version",
        "joint_names",
        "coordinate_space",
        "source",
        "sampling",
        "extraction",
        "frames",
    }
)
SOURCE_KEYS = frozenset(
    {
        "video_id",
        "shot_id",
        "video_sha256",
        "annotations_sha256",
        "split_manifest_sha256",
        "partition",
        "width",
        "height",
        "frame_count",
        "fps",
        "timing",
    }
)
SAMPLING_KEYS = frozenset({"start_frame", "end_frame", "num_frames", "policy"})
EXTRACTION_KEYS = frozenset(
    {
        "extractor_id",
        "extractor_revision",
        "mapping_revision",
        "preprocessing_revision",
        "subject_policy_revision",
        "weights_sha256",
        "config_sha256",
        "validity_threshold",
    }
)
FRAME_KEYS = frozenset(
    {"source_frame_index", "subject_status", "subject_id", "joints"}
)
JOINT_KEYS = frozenset({"xy", "confidence", "valid"})

UNAVAILABLE_JOINT = {"xy": None, "confidence": None, "valid": False}

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class PoseRecordValidationError(ValueError):
    """Raised when a pose record does not follow ``pose-pilot-v1``."""


def _fail(path: str, message: str) -> None:
    raise PoseRecordValidationError(f"{path}: {message}")


def _require_object(value: Any, keys: frozenset[str], path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        _fail(path, f"must be an object, got {type(value).__name__}")
    missing = sorted(keys.difference(value))
    unknown = sorted(set(value).difference(keys))
    if missing:
        _fail(path, f"missing required fields {missing}")
    if unknown:
        _fail(path, f"unknown fields {unknown}")
    return value


def _require_list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(path, f"must be an array, got {type(value).__name__}")
    return value


def _require_literal(value: Any, expected: str, path: str) -> None:
    if not isinstance(value, str) or value != expected:
        _fail(path, f"must be exactly {expected!r}, got {value!r}")


def _require_choice(value: Any, choices: Iterable[str], path: str) -> str:
    choices = tuple(choices)
    if not isinstance(value, str) or value not in choices:
        _fail(path, f"must be one of {list(choices)}, got {value!r}")
    return value


def _require_nonempty_str(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(path, f"must be a nonempty string, got {value!r}")
    return value


def _require_sha256(value: Any, path: str) -> None:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        _fail(path, f"must be a 64-character lowercase hex SHA-256, got {value!r}")


def _require_int(value: Any, path: str, minimum: int | None = None) -> int:
    # ``bool`` is a subclass of ``int``; the contract rejects it as a number.
    if type(value) is not int:
        _fail(path, f"must be an integer, got {value!r}")
    if minimum is not None and value < minimum:
        _fail(path, f"must be >= {minimum}, got {value}")
    return value


def _require_number(
    value: Any,
    path: str,
    low: float | None = None,
    high: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(path, f"must be a number, got {value!r}")
    if not math.isfinite(value):
        _fail(path, f"must be finite, got {value!r}")
    if (low is not None and value < low) or (high is not None and value > high):
        _fail(path, f"must be in [{low}, {high}], got {value!r}")
    return value


def _validate_source(source: Any) -> Mapping[str, Any]:
    source = _require_object(source, SOURCE_KEYS, "source")
    _require_nonempty_str(source["video_id"], "source.video_id")
    _require_nonempty_str(source["shot_id"], "source.shot_id")
    for key in ("video_sha256", "annotations_sha256", "split_manifest_sha256"):
        _require_sha256(source[key], f"source.{key}")
    _require_choice(source["partition"], PARTITIONS, "source.partition")
    _require_int(source["width"], "source.width", minimum=2)
    _require_int(source["height"], "source.height", minimum=2)
    _require_int(source["frame_count"], "source.frame_count", minimum=1)
    fps = _require_number(source["fps"], "source.fps")
    if fps <= 0:
        _fail("source.fps", f"must be positive, got {fps!r}")
    _require_literal(source["timing"], TIMING, "source.timing")
    return source


def _validate_sampling(sampling: Any, frame_count: int) -> Mapping[str, Any]:
    sampling = _require_object(sampling, SAMPLING_KEYS, "sampling")
    start = _require_int(sampling["start_frame"], "sampling.start_frame", minimum=0)
    end = _require_int(sampling["end_frame"], "sampling.end_frame")
    if end <= start:
        _fail("sampling.end_frame", f"must be greater than start_frame {start}, got {end}")
    if end > frame_count:
        _fail(
            "sampling.end_frame",
            f"must be <= source.frame_count {frame_count}, got {end}",
        )
    _require_int(sampling["num_frames"], "sampling.num_frames", minimum=1)
    _require_literal(sampling["policy"], SAMPLING_POLICY, "sampling.policy")
    return sampling


def _validate_extraction(extraction: Any) -> Mapping[str, Any]:
    extraction = _require_object(extraction, EXTRACTION_KEYS, "extraction")
    for key in (
        "extractor_id",
        "extractor_revision",
        "mapping_revision",
        "preprocessing_revision",
        "subject_policy_revision",
    ):
        _require_nonempty_str(extraction[key], f"extraction.{key}")
    _require_sha256(extraction["weights_sha256"], "extraction.weights_sha256")
    _require_sha256(extraction["config_sha256"], "extraction.config_sha256")
    _require_number(
        extraction["validity_threshold"],
        "extraction.validity_threshold",
        low=0.0,
        high=1.0,
    )
    return extraction


def _validate_joint(joint: Any, path: str, selected: bool, threshold: float) -> None:
    joint = _require_object(joint, JOINT_KEYS, path)
    xy = joint["xy"]
    if xy is not None:
        xy = _require_list(xy, f"{path}.xy")
        if len(xy) != 2:
            _fail(f"{path}.xy", f"must contain exactly 2 numbers, got {len(xy)}")
        for axis, coordinate in enumerate(xy):
            _require_number(coordinate, f"{path}.xy[{axis}]", low=0.0, high=1.0)

    confidence = joint["confidence"]
    if confidence is not None:
        _require_number(confidence, f"{path}.confidence", low=0.0, high=1.0)

    valid = joint["valid"]
    if type(valid) is not bool:
        _fail(f"{path}.valid", f"must be a boolean, got {valid!r}")

    if selected:
        expected = xy is not None and confidence is not None and confidence >= threshold
        if valid != expected:
            _fail(
                f"{path}.valid",
                f"must be {expected} for xy={xy!r}, confidence={confidence!r} "
                f"and validity_threshold={threshold!r}",
            )
    elif joint != UNAVAILABLE_JOINT:
        _fail(path, f"must be exactly {UNAVAILABLE_JOINT} when the subject is not selected")


def _validate_frame(
    frame: Any,
    path: str,
    start: int,
    end: int,
    threshold: float,
) -> Mapping[str, Any]:
    frame = _require_object(frame, FRAME_KEYS, path)
    index = _require_int(frame["source_frame_index"], f"{path}.source_frame_index")
    if not start <= index < end:
        _fail(
            f"{path}.source_frame_index",
            f"must be inside the sampling interval [{start}, {end}), got {index}",
        )

    status = _require_choice(
        frame["subject_status"], SUBJECT_STATUSES, f"{path}.subject_status"
    )
    selected = status == "selected"
    if selected:
        _require_nonempty_str(frame["subject_id"], f"{path}.subject_id")
    elif frame["subject_id"] is not None:
        _fail(f"{path}.subject_id", f"must be null when subject_status is {status!r}")

    joints = _require_list(frame["joints"], f"{path}.joints")
    if len(joints) != len(CANONICAL_JOINT_NAMES):
        _fail(
            f"{path}.joints",
            f"must contain exactly {len(CANONICAL_JOINT_NAMES)} joints, got {len(joints)}",
        )
    for joint_index, joint in enumerate(joints):
        _validate_joint(joint, f"{path}.joints[{joint_index}]", selected, threshold)
    return frame


def validate_pose_record(record: Any) -> None:
    """Return ``None`` if ``record`` follows ``pose-pilot-v1``; raise otherwise.

    Raises:
        PoseRecordValidationError: describing the first violation found, with a
            JSON-style path such as ``frames[0].joints[3].xy[1]``.
    """

    record = _require_object(record, RECORD_KEYS, "record")
    _require_literal(record["schema_version"], SCHEMA_VERSION, "schema_version")
    joint_names = _require_list(record["joint_names"], "joint_names")
    if tuple(joint_names) != CANONICAL_JOINT_NAMES:
        _fail("joint_names", f"must exactly match the canonical order {list(CANONICAL_JOINT_NAMES)}")
    _require_literal(record["coordinate_space"], COORDINATE_SPACE, "coordinate_space")

    source = _validate_source(record["source"])
    sampling = _validate_sampling(record["sampling"], source["frame_count"])
    extraction = _validate_extraction(record["extraction"])

    frames = _require_list(record["frames"], "frames")
    if not frames:
        _fail("frames", "must be nonempty")
    if len(frames) != sampling["num_frames"]:
        _fail(
            "frames",
            f"must contain sampling.num_frames={sampling['num_frames']} frames, "
            f"got {len(frames)}",
        )

    start, end = sampling["start_frame"], sampling["end_frame"]
    threshold = extraction["validity_threshold"]
    subject_id: str | None = None
    previous: Mapping[str, Any] | None = None
    for frame_index, frame in enumerate(frames):
        path = f"frames[{frame_index}]"
        frame = _validate_frame(frame, path, start, end, threshold)

        if frame["subject_status"] == "selected":
            if subject_id is None:
                subject_id = frame["subject_id"]
            elif frame["subject_id"] != subject_id:
                _fail(
                    f"{path}.subject_id",
                    f"must match the record's selected subject {subject_id!r}, "
                    f"got {frame['subject_id']!r}",
                )

        if previous is not None:
            index = frame["source_frame_index"]
            previous_index = previous["source_frame_index"]
            if index < previous_index:
                _fail(
                    f"{path}.source_frame_index",
                    f"must be nondecreasing, got {index} after {previous_index}",
                )
            # Indices are nondecreasing, so repeats are always adjacent.
            if index == previous_index and frame != previous:
                _fail(
                    path,
                    f"repeats source_frame_index {index} but does not repeat the "
                    "previous frame observation identically",
                )
        previous = frame


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PoseRecordValidationError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_constant(name: str) -> None:
    raise PoseRecordValidationError(f"non-finite JSON number {name} is not allowed")


def load_pose_record(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read a JSON pose record, validate it, and return it as a dictionary.

    Unlike plain ``json.load``, duplicate object keys and ``NaN``/``Infinity``
    literals are rejected instead of being silently accepted.

    Raises:
        PoseRecordValidationError: if the file cannot be read, is not valid
            JSON, or does not follow ``pose-pilot-v1``.
    """

    location = os.fspath(path)
    try:
        with open(location, "r", encoding="utf-8") as handle:
            text = handle.read()
    except (OSError, UnicodeDecodeError) as error:
        raise PoseRecordValidationError(f"{location}: cannot read file: {error}") from error

    try:
        record = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except PoseRecordValidationError as error:
        raise PoseRecordValidationError(f"{location}: {error}") from error
    except json.JSONDecodeError as error:
        raise PoseRecordValidationError(
            f"{location}: invalid JSON at line {error.lineno} column {error.colno}: {error.msg}"
        ) from error

    try:
        validate_pose_record(record)
    except PoseRecordValidationError as error:
        raise PoseRecordValidationError(f"{location}: {error}") from error
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check that a JSON file is a valid pose-pilot-v1 pose record."
    )
    parser.add_argument("path", help="pose record JSON file to check")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Check one JSON file; return 0 if valid and 1 if invalid."""

    args = build_parser().parse_args(argv)
    try:
        load_pose_record(args.path)
    except PoseRecordValidationError as error:
        print(f"INVALID: {error}", file=sys.stderr)
        return 1
    print(f"OK: {args.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
