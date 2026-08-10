"""WAN-independent sketch inference configuration helpers."""

from __future__ import annotations

from collections.abc import Sequence


def validate_inference_settings(
    frame_num: int,
    control_strengths: Sequence[float],
) -> tuple[float, ...]:
    if frame_num <= 0 or frame_num % 4 != 1:
        raise ValueError("frame_num must be positive and satisfy 4n+1")
    strengths = tuple(float(value) for value in control_strengths)
    if not strengths:
        raise ValueError("at least one control strength is required")
    if any(value < 0.0 for value in strengths):
        raise ValueError("control strengths must be non-negative")
    return strengths


def strength_tag(strength: float) -> str:
    return f"{float(strength):g}".replace(".", "p")
