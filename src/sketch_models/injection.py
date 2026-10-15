"""Pure control-token alignment and residual helpers."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F


CFG_CONTROL_POLICY = "both_text_branches"


def validate_cfg_control_policy(policy: str) -> str:
    if policy != CFG_CONTROL_POLICY:
        raise ValueError(
            f"unsupported cfg_control_policy {policy!r}; "
            f"expected {CFG_CONTROL_POLICY!r}"
        )
    return policy


def _validate_grid(grid: Sequence[int], name: str) -> tuple[int, int, int]:
    if len(grid) != 3:
        raise ValueError(f"{name} must contain (T,H,W)")
    result = tuple(int(value) for value in grid)
    if any(value <= 0 for value in result):
        raise ValueError(f"{name} values must be positive, got {result}")
    return result


def resize_control_tokens(
    control_tokens: torch.Tensor,
    *,
    source_grid: Sequence[int],
    target_grid: Sequence[int],
) -> torch.Tensor:
    """Resize ``[B,S,C]`` tokens to the real WAN grid without padding."""

    if control_tokens.ndim != 3:
        raise ValueError(
            f"control_tokens must be [B,S,C], got {tuple(control_tokens.shape)}"
        )
    source_t, source_h, source_w = _validate_grid(source_grid, "source_grid")
    target_t, target_h, target_w = _validate_grid(target_grid, "target_grid")
    expected_source = source_t * source_h * source_w
    if control_tokens.shape[1] != expected_source:
        raise ValueError(
            f"source grid contains {expected_source} tokens but tensor has "
            f"{control_tokens.shape[1]}"
        )

    batch, _, channels = control_tokens.shape
    volume = control_tokens.reshape(
        batch,
        source_t,
        source_h,
        source_w,
        channels,
    ).permute(0, 4, 1, 2, 3)
    resized = F.interpolate(
        volume.float(),
        size=(target_t, target_h, target_w),
        mode="trilinear",
        align_corners=False,
    )
    return resized.permute(0, 2, 3, 4, 1).reshape(
        batch,
        target_t * target_h * target_w,
        channels,
    )


def normalize_control_residual(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    *,
    ratio_cap: float = 0.1,
    control_strength: float = 1.0,
) -> torch.Tensor:
    """Apply the residual norm cap and inference strength."""

    if hidden_states.shape != residual.shape:
        raise ValueError(
            "hidden_states and residual must have identical real-token shapes"
        )
    if ratio_cap < 0.0:
        raise ValueError("ratio_cap must be non-negative")
    if control_strength < 0.0:
        raise ValueError("control_strength must be non-negative")

    hidden_norm = hidden_states.norm(dim=-1, keepdim=True).mean()
    residual_norm = residual.norm(dim=-1, keepdim=True).mean()
    if float(residual_norm.detach()) > 0.0:
        scale = (hidden_norm / residual_norm).clamp(max=1.0) * ratio_cap
        residual = residual * scale
    else:
        # Preserve a bounded first-step gradient through a zero projection.
        residual = residual * ratio_cap
    return residual * control_strength


def build_padded_control_residual(
    hidden_states: torch.Tensor,
    control_tokens: torch.Tensor,
    projection,
    *,
    source_grid: Sequence[int],
    target_grid: Sequence[int],
    ratio_cap: float = 0.1,
    control_strength: float = 1.0,
) -> torch.Tensor:
    """Project real tokens and append exact zeros for WAN sequence padding."""

    if hidden_states.ndim != 3:
        raise ValueError("hidden_states must be [B,L,C]")
    target_t, target_h, target_w = _validate_grid(target_grid, "target_grid")
    real_length = target_t * target_h * target_w
    if real_length > hidden_states.shape[1]:
        raise ValueError(
            f"WAN real grid has {real_length} tokens but sequence length is "
            f"{hidden_states.shape[1]}"
        )

    real_control = resize_control_tokens(
        control_tokens,
        source_grid=source_grid,
        target_grid=target_grid,
    )
    if real_control.shape[0] != hidden_states.shape[0]:
        raise ValueError("control and WAN batch sizes differ")
    if real_control.shape[2] != hidden_states.shape[2]:
        raise ValueError("control and WAN feature widths differ")

    projected = projection(real_control)
    projected = normalize_control_residual(
        hidden_states[:, :real_length],
        projected,
        ratio_cap=ratio_cap,
        control_strength=control_strength,
    )
    projected = projected.to(dtype=hidden_states.dtype)
    if real_length == hidden_states.shape[1]:
        return projected

    padding = projected.new_zeros(
        projected.shape[0],
        hidden_states.shape[1] - real_length,
        projected.shape[2],
    )
    return torch.cat([projected, padding], dim=1)
