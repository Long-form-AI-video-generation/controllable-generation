"""Stable contracts shared by multi-control inference components.

This module deliberately does not import WAN or checkpoint files.  Keeping the
basic contracts dependency-free lets configuration failures happen before a
large model is constructed.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping


CANONICAL_EXPERT_ORDER = ("depth", "canny", "mask")
INJECTION_LAYERS = (0, 4, 8, 12, 16, 20, 24, 28)
DEFAULT_RATIO_CAP = 0.1


@dataclass(frozen=True)
class ExpertSpec:
    """The non-negotiable identity and loading contract for one control."""

    name: str
    control_key: str
    adapter_import: str
    preprocessing_kind: str
    control_type: str
    injection_layers: tuple[int, ...] = INJECTION_LAYERS
    standalone_ratio_cap: float = DEFAULT_RATIO_CAP

    def __post_init__(self) -> None:
        if self.name not in CANONICAL_EXPERT_ORDER:
            raise ValueError(f"unknown expert name {self.name!r}")
        if not self.control_key:
            raise ValueError("control_key must be non-empty")
        if not self.adapter_import:
            raise ValueError("adapter_import must be non-empty")
        if not self.preprocessing_kind:
            raise ValueError("preprocessing_kind must be non-empty")
        if not self.control_type:
            raise ValueError("control_type must be non-empty")
        if self.injection_layers != INJECTION_LAYERS:
            raise ValueError(
                "integration requires the canonical eight injection layers "
                f"{INJECTION_LAYERS}, got {self.injection_layers}"
            )
        if not 0.0 < self.standalone_ratio_cap <= 1.0:
            raise ValueError("standalone_ratio_cap must be in (0, 1]")


EXPERT_SPECS: Mapping[str, ExpertSpec] = MappingProxyType(
    {
        "depth": ExpertSpec(
            name="depth",
            control_key="depth_encoded",
            adapter_import="src.depth_models.control_adapter.ControlAdapter",
            preprocessing_kind="midas",
            control_type="depth",
        ),
        "canny": ExpertSpec(
            name="canny",
            control_key="sketch_encoded",
            adapter_import=(
                "src.sketch_models.control_adapter.SketchControlAdapter"
            ),
            preprocessing_kind="canny-v1",
            control_type="canny",
        ),
        "mask": ExpertSpec(
            name="mask",
            control_key="mask_encoded",
            adapter_import="src.mask_models.control_adapter.MaskControlAdapter",
            preprocessing_kind="segformer-b5-ade20k",
            control_type="mask",
        ),
    }
)


def canonicalize_expert_names(names: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Validate an expert collection and return it in canonical order."""

    requested = tuple(str(name).strip() for name in names)
    if not requested:
        raise ValueError("at least one control expert is required")
    if any(not name for name in requested):
        raise ValueError("control expert names must be non-empty")

    unknown = sorted(set(requested).difference(EXPERT_SPECS))
    if unknown:
        raise ValueError(f"unknown control experts: {unknown}")
    if len(set(requested)) != len(requested):
        raise ValueError(f"control experts must not repeat: {requested}")

    selected = set(requested)
    return tuple(name for name in CANONICAL_EXPERT_ORDER if name in selected)


def parse_control_combination(value: str) -> tuple[str, ...]:
    """Parse a CLI combination such as ``depth+canny`` deterministically."""

    if not isinstance(value, str):
        raise ValueError("control combination must be a string")
    raw_parts = value.split("+")
    if not raw_parts or any(not part.strip() for part in raw_parts):
        raise ValueError(f"invalid control combination {value!r}")
    return canonicalize_expert_names(raw_parts)


def combination_name(experts: tuple[str, ...] | list[str]) -> str:
    """Return the canonical, filename-safe form of an expert combination."""

    return "+".join(canonicalize_expert_names(experts))


@dataclass(frozen=True)
class ActivationState:
    """Immutable per-generation control activation snapshot.

    ``adapter_signals`` intentionally keeps references to already-computed
    tensors.  The controller owns their lifecycle and clears the whole state
    between generations rather than copying large GPU tensors.
    """

    enabled_experts: tuple[str, ...]
    adapter_signals: Mapping[str, Any]
    strengths: Mapping[str, float]
    ratio_caps: Mapping[str, float]
    combined_ratio_cap: float
    diagnostics_enabled: bool
    control_artifact_sha256: str
    generation_id: int

    def __post_init__(self) -> None:
        experts = canonicalize_expert_names(list(self.enabled_experts))
        object.__setattr__(self, "enabled_experts", experts)

        signal_keys = set(self.adapter_signals)
        strength_keys = set(self.strengths)
        cap_keys = set(self.ratio_caps)
        required = set(experts)
        if signal_keys != required:
            raise ValueError(
                "adapter signals must match enabled experts exactly; "
                f"expected {sorted(required)}, got {sorted(signal_keys)}"
            )
        if strength_keys != required or cap_keys != required:
            raise ValueError("strengths and ratio caps must match enabled experts")
        if any(float(value) < 0.0 for value in self.strengths.values()):
            raise ValueError("control strengths must be non-negative")
        if any(not 0.0 < float(value) <= 1.0 for value in self.ratio_caps.values()):
            raise ValueError("per-expert ratio caps must be in (0, 1]")
        if not 0.0 < float(self.combined_ratio_cap) <= 1.0:
            raise ValueError("combined_ratio_cap must be in (0, 1]")
        if len(self.control_artifact_sha256) != 64:
            raise ValueError("control_artifact_sha256 must be a SHA-256 hex digest")
        try:
            int(self.control_artifact_sha256, 16)
        except ValueError as error:
            raise ValueError("control_artifact_sha256 must be hexadecimal") from error
        if self.generation_id < 0:
            raise ValueError("generation_id must be non-negative")

        object.__setattr__(
            self,
            "adapter_signals",
            MappingProxyType(dict(self.adapter_signals)),
        )
        object.__setattr__(self, "strengths", MappingProxyType(dict(self.strengths)))
        object.__setattr__(self, "ratio_caps", MappingProxyType(dict(self.ratio_caps)))
