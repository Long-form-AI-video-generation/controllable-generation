"""Strict CPU-first loading for the three independently trained controls."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn

from src.depth_models.control_adapter import ControlAdapter
from src.mask_models.control_adapter import MaskControlAdapter
from src.sketch_models.control_adapter import SketchControlAdapter
from src.sketch_models.injection import CFG_CONTROL_POLICY

from .contracts import EXPERT_SPECS, INJECTION_LAYERS, ExpertSpec


class ZeroLinear(nn.Module):
    """State-compatible zero projection without a WAN dependency."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("zero-conv dimension must be positive")
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.proj(value)


@dataclass(frozen=True)
class LoadedExpert:
    """Inference-only state for one validated control checkpoint."""

    spec: ExpertSpec
    adapter: nn.Module
    zero_convs: nn.ModuleList
    dit_dim: int
    checkpoint_path: Path
    checkpoint_sha256: str
    global_step: int | None
    epoch: int | None
    best_val_loss: float | None
    control_metadata: Mapping[str, Any] | None


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"checkpoint {field!r} must be a mapping")
    return value


def _as_tensor_mapping(value: object, field: str) -> Mapping[str, torch.Tensor]:
    mapping = _as_mapping(value, field)
    invalid = [key for key, item in mapping.items() if not isinstance(item, torch.Tensor)]
    if invalid:
        raise ValueError(f"checkpoint {field!r} contains non-tensor entries: {invalid}")
    return mapping  # type: ignore[return-value]


def expected_control_metadata(spec: ExpertSpec) -> Mapping[str, Any] | None:
    """Return the exact metadata contract saved by the Canny/mask trainers."""

    common: dict[str, Any] = {
        "control_type": spec.control_type,
        "control_key": spec.control_key,
        "input_channels": 1,
        "condition_dim": 256,
        "target_spatial": 16,
        "injection_layers": list(INJECTION_LAYERS),
        "cfg_control_policy": CFG_CONTROL_POLICY,
        "control_ratio_cap": spec.standalone_ratio_cap,
    }
    if spec.name == "depth":
        # The validated depth checkpoint predates explicit metadata.  It is
        # accepted only through this explicit depth-only compatibility path.
        return None
    if spec.name == "canny":
        return {
            **common,
            "condition_encoder": "SketchConditionEncoder",
        }
    if spec.name == "mask":
        return {
            **common,
            "condition_encoder": "MaskConditionEncoder",
            "num_classes": 150,
            "embedding_dim": 16,
        }
    raise AssertionError(f"unhandled expert {spec.name}")


def _instantiate_adapter(spec: ExpertSpec, dit_dim: int) -> nn.Module:
    if spec.name == "depth":
        return ControlAdapter(
            control_dim=256,
            hidden_dim=512,
            dit_dim=dit_dim,
            num_controls=1,
            use_gradient_checkpointing=False,
        )
    if spec.name == "canny":
        return SketchControlAdapter(
            hidden_dim=512,
            dit_dim=dit_dim,
            condition_dim=256,
            target_spatial=16,
            use_gradient_checkpointing=False,
        )
    if spec.name == "mask":
        return MaskControlAdapter(
            hidden_dim=512,
            dit_dim=dit_dim,
            condition_dim=256,
            target_spatial=16,
            use_gradient_checkpointing=False,
        )
    raise AssertionError(f"unhandled expert {spec.name}")


def _infer_dit_dim(zero_state: Mapping[str, torch.Tensor]) -> int:
    expected_keys = {
        f"{index}.proj.{parameter}"
        for index in range(len(INJECTION_LAYERS))
        for parameter in ("weight", "bias")
    }
    actual_keys = set(zero_state)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys.difference(actual_keys))
        extra = sorted(actual_keys.difference(expected_keys))
        raise ValueError(
            "zero_convs state keys are incompatible; "
            f"missing={missing}, extra={extra}"
        )

    dimension: int | None = None
    for index in range(len(INJECTION_LAYERS)):
        weight = zero_state[f"{index}.proj.weight"]
        bias = zero_state[f"{index}.proj.bias"]
        if weight.ndim != 2 or weight.shape[0] != weight.shape[1]:
            raise ValueError(f"zero_convs layer {index} must have square 2D weight")
        if bias.ndim != 1 or bias.shape[0] != weight.shape[0]:
            raise ValueError(f"zero_convs layer {index} bias shape is incompatible")
        if dimension is None:
            dimension = int(weight.shape[0])
        elif dimension != int(weight.shape[0]):
            raise ValueError("zero_convs layers disagree on the DiT dimension")
    assert dimension is not None
    return dimension


def _validate_metadata(spec: ExpertSpec, checkpoint: Mapping[str, Any]) -> Mapping[str, Any] | None:
    expected = expected_control_metadata(spec)
    actual = checkpoint.get("control_metadata")
    if expected is None:
        if actual is None:
            return None
        actual_mapping = _as_mapping(actual, "control_metadata")
        if actual_mapping.get("control_type") != "depth":
            raise ValueError("legacy depth checkpoint metadata identifies another control")
        return actual_mapping
    if actual != expected:
        raise ValueError(
            f"checkpoint metadata is incompatible with {spec.name}; "
            f"expected {expected}, got {actual}"
        )
    return _as_mapping(actual, "control_metadata")


def load_expert_checkpoint(
    name: str,
    checkpoint_path: str | Path,
) -> LoadedExpert:
    """Load adapter/projection inference state without constructing WAN.

    Optimizer and scheduler state may be present in a training checkpoint, but
    this function intentionally never reads or restores it.
    """

    try:
        spec = EXPERT_SPECS[name]
    except KeyError as error:
        raise ValueError(f"unknown control expert {name!r}") from error

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(path)

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    checkpoint_mapping = _as_mapping(checkpoint, "root")
    model_state = _as_tensor_mapping(checkpoint_mapping.get("model"), "model")
    zero_state = _as_tensor_mapping(
        checkpoint_mapping.get("zero_convs"),
        "zero_convs",
    )
    metadata = _validate_metadata(spec, checkpoint_mapping)
    dit_dim = _infer_dit_dim(zero_state)

    adapter = _instantiate_adapter(spec, dit_dim)
    zero_convs = nn.ModuleList(
        [ZeroLinear(dit_dim) for _ in INJECTION_LAYERS]
    )
    try:
        adapter.load_state_dict(model_state, strict=True)
    except RuntimeError as error:
        raise ValueError(
            f"{spec.name} adapter state is incompatible with DiT dimension {dit_dim}"
        ) from error
    try:
        zero_convs.load_state_dict(zero_state, strict=True)
    except RuntimeError as error:
        raise ValueError(f"{spec.name} zero_convs state is incompatible") from error

    adapter.eval()
    zero_convs.eval()
    return LoadedExpert(
        spec=spec,
        adapter=adapter,
        zero_convs=zero_convs,
        dit_dim=dit_dim,
        checkpoint_path=path.resolve(),
        checkpoint_sha256=file_sha256(path),
        global_step=_optional_int(checkpoint_mapping.get("global_step")),
        epoch=_optional_int(checkpoint_mapping.get("epoch")),
        best_val_loss=_optional_float(checkpoint_mapping.get("best_val_loss")),
        control_metadata=metadata,
    )


def load_requested_experts(
    checkpoint_paths: Mapping[str, str | Path],
) -> dict[str, LoadedExpert]:
    """Load a unique set of experts and require one shared DiT width."""

    if not checkpoint_paths:
        raise ValueError("at least one checkpoint is required")
    unknown = sorted(set(checkpoint_paths).difference(EXPERT_SPECS))
    if unknown:
        raise ValueError(f"unknown checkpoint experts: {unknown}")

    normalized_paths = [Path(path).resolve() for path in checkpoint_paths.values()]
    if len(set(normalized_paths)) != len(normalized_paths):
        raise ValueError("one checkpoint path must not be reused for multiple experts")

    loaded = {
        name: load_expert_checkpoint(name, checkpoint_paths[name])
        for name in EXPERT_SPECS
        if name in checkpoint_paths
    }
    dimensions = {expert.dit_dim for expert in loaded.values()}
    if len(dimensions) != 1:
        raise ValueError(f"experts disagree on DiT dimension: {sorted(dimensions)}")
    return loaded


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("checkpoint numeric metadata must not be boolean")
    return int(value)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("checkpoint numeric metadata must not be boolean")
    return float(value)
