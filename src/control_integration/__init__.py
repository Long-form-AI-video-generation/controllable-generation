"""Inference-time composition of depth, Canny, and semantic-mask controls."""

from .contracts import (
    CANONICAL_EXPERT_ORDER,
    EXPERT_SPECS,
    ActivationState,
    ExpertSpec,
    canonicalize_expert_names,
    parse_control_combination,
)

__all__ = [
    "ActivationState",
    "CANONICAL_EXPERT_ORDER",
    "EXPERT_SPECS",
    "ExpertSpec",
    "canonicalize_expert_names",
    "parse_control_combination",
]
