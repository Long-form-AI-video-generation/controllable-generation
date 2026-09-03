"""WAN-independent validation for multi-control inference requests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite

from .contracts import CANONICAL_EXPERT_ORDER, canonicalize_expert_names, parse_control_combination


SUPPORTED_OUTPUT_SIZES = ((480, 832),)


def parse_output_size(value: str) -> tuple[int, int]:
    """Parse the existing ``HEIGHT*WIDTH`` WAN CLI convention."""

    if not isinstance(value, str):
        raise ValueError("size must use the HEIGHT*WIDTH string format")
    pieces = value.split("*")
    if len(pieces) != 2:
        raise ValueError("size must use the HEIGHT*WIDTH string format")
    try:
        result = tuple(int(piece) for piece in pieces)
    except ValueError as error:
        raise ValueError("size values must be integers") from error
    if result not in SUPPORTED_OUTPUT_SIZES:
        allowed = ", ".join(f"{height}*{width}" for height, width in SUPPORTED_OUTPUT_SIZES)
        raise ValueError(f"unsupported size {value!r}; expected one of {allowed}")
    return result


def validate_combinations(values: Sequence[str]) -> tuple[tuple[str, ...], ...]:
    """Parse unique requested combinations in user-specified evaluation order."""

    if not values:
        raise ValueError("at least one control combination is required")
    parsed = tuple(parse_control_combination(value) for value in values)
    if len(set(parsed)) != len(parsed):
        raise ValueError("control combinations must be unique")
    return parsed


def validate_strengths(
    strengths: Mapping[str, float],
    *,
    required_experts: Sequence[str],
) -> dict[str, float]:
    """Validate and canonicalize individual expert strengths."""

    required = canonicalize_expert_names(list(required_experts))
    unknown = sorted(set(strengths).difference(CANONICAL_EXPERT_ORDER))
    if unknown:
        raise ValueError(f"unknown strengths for experts: {unknown}")
    missing = [name for name in required if name not in strengths]
    if missing:
        raise ValueError(f"missing strengths for requested experts: {missing}")

    result: dict[str, float] = {}
    for name in required:
        value = float(strengths[name])
        if not isfinite(value) or value < 0.0:
            raise ValueError(
                f"strength for {name!r} must be finite and non-negative"
            )
        result[name] = value
    return result


@dataclass(frozen=True)
class InferenceConfig:
    """Fully validated non-WAN settings used by the integration entry point."""

    frame_num: int
    steps: int
    fps: int
    output_size: tuple[int, int]
    combinations: tuple[tuple[str, ...], ...]
    strengths: Mapping[str, float]
    combined_ratio_cap: float

    def __post_init__(self) -> None:
        if self.frame_num <= 0 or self.frame_num % 4 != 1:
            raise ValueError("frame_num must be positive and satisfy 4n+1")
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.output_size not in SUPPORTED_OUTPUT_SIZES:
            raise ValueError(f"unsupported output_size {self.output_size}")
        if not self.combinations:
            raise ValueError("at least one control combination is required")
        if len(set(self.combinations)) != len(self.combinations):
            raise ValueError("control combinations must be unique")

        normalized_combinations = tuple(
            canonicalize_expert_names(list(combination))
            for combination in self.combinations
        )
        object.__setattr__(self, "combinations", normalized_combinations)
        active = tuple(
            expert
            for expert in CANONICAL_EXPERT_ORDER
            if any(expert in combination for combination in normalized_combinations)
        )
        object.__setattr__(
            self,
            "strengths",
            validate_strengths(self.strengths, required_experts=active),
        )
        if not isfinite(float(self.combined_ratio_cap)) or not (
            0.0 < float(self.combined_ratio_cap) <= 1.0
        ):
            raise ValueError("combined_ratio_cap must be finite and in (0, 1]")


def build_inference_config(
    *,
    frame_num: int,
    steps: int,
    fps: int,
    size: str,
    control_combinations: Sequence[str],
    strengths: Mapping[str, float],
    combined_ratio_cap: float,
) -> InferenceConfig:
    """Construct an :class:`InferenceConfig` from CLI-shaped values."""

    return InferenceConfig(
        frame_num=int(frame_num),
        steps=int(steps),
        fps=int(fps),
        output_size=parse_output_size(size),
        combinations=validate_combinations(control_combinations),
        strengths=strengths,
        combined_ratio_cap=float(combined_ratio_cap),
    )
