"""Pure, bounded addition of independently trained control residuals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import torch

from src.sketch_models.injection import build_padded_control_residual

from .contracts import CANONICAL_EXPERT_ORDER, canonicalize_expert_names


@dataclass(frozen=True)
class FusionDiagnostics:
    """Non-mutating norm measurements for one injection layer."""

    per_expert_mean_norm: Mapping[str, float]
    fused_mean_norm_before_cap: float
    fused_mean_norm_after_cap: float
    hidden_mean_norm: float
    cap_factor: float
    token_ratio_p50: float
    token_ratio_p95: float
    token_ratio_max: float


def _validate_hidden(hidden_states: torch.Tensor, real_length: int) -> None:
    if hidden_states.ndim != 3:
        raise ValueError("hidden_states must be [B,L,C]")
    if not 0 < real_length <= hidden_states.shape[1]:
        raise ValueError(
            f"real_length must be in [1, {hidden_states.shape[1]}], got {real_length}"
        )


def mean_token_l2(values: torch.Tensor) -> torch.Tensor:
    """Mean L2 norm, accumulated in float32 for numerical stability."""

    if values.ndim != 3:
        raise ValueError("values must be [B,L,C]")
    return values.float().norm(dim=-1).mean()


def build_expert_residual(
    hidden_states: torch.Tensor,
    control_tokens: torch.Tensor,
    projection: Callable[[torch.Tensor], torch.Tensor],
    *,
    source_grid: Sequence[int],
    target_grid: Sequence[int],
    ratio_cap: float,
    strength: float,
) -> torch.Tensor:
    """Match the proven single-control residual formula with safe padding.

    The imported helper resizes only the real WAN grid, projects those tokens,
    normalizes with the standalone ratio cap, then appends literal zeros.  That
    avoids a projection bias leaking into sequence padding.
    """

    return build_padded_control_residual(
        hidden_states,
        control_tokens,
        projection,
        source_grid=source_grid,
        target_grid=target_grid,
        ratio_cap=ratio_cap,
        control_strength=strength,
    )


def _validate_residuals(
    hidden_states: torch.Tensor,
    residuals: Mapping[str, torch.Tensor],
    *,
    real_length: int,
) -> tuple[str, ...]:
    names = canonicalize_expert_names(list(residuals))
    for name in names:
        residual = residuals[name]
        if residual.shape != hidden_states.shape:
            raise ValueError(
                f"{name} residual shape {tuple(residual.shape)} does not match "
                f"hidden states {tuple(hidden_states.shape)}"
            )
        if real_length < hidden_states.shape[1] and torch.count_nonzero(
            residual[:, real_length:]
        ).item():
            raise ValueError(f"{name} residual has nonzero padding tokens")
    return names


def fuse_residuals(
    hidden_states: torch.Tensor,
    residuals: Mapping[str, torch.Tensor],
    *,
    real_length: int,
    combined_ratio_cap: float,
    diagnostics: bool = False,
) -> tuple[torch.Tensor, FusionDiagnostics | None]:
    """Sum canonical residuals and apply one hard global norm ceiling.

    Per-expert normalization must already have occurred.  The global cap is a
    ceiling on their sum, not a second always-on attenuation: a sum below the
    budget is returned unchanged.
    """

    _validate_hidden(hidden_states, real_length)
    if not 0.0 < float(combined_ratio_cap) <= 1.0:
        raise ValueError("combined_ratio_cap must be in (0, 1]")
    if not residuals:
        zero = torch.zeros_like(hidden_states)
        return zero, _diagnostics(hidden_states, {}, zero, real_length, 1.0) if diagnostics else None

    names = _validate_residuals(
        hidden_states,
        residuals,
        real_length=real_length,
    )
    fused = torch.zeros_like(hidden_states)
    # Canonical ordering makes a reproducible summation order explicit.
    for name in names:
        fused = fused + residuals[name]

    real_hidden = hidden_states[:, :real_length]
    real_fused = fused[:, :real_length]
    hidden_norm = mean_token_l2(real_hidden)
    fused_norm = mean_token_l2(real_fused)
    if float(fused_norm.detach()) > 0.0:
        max_norm = hidden_norm * float(combined_ratio_cap)
        cap_factor = (max_norm / fused_norm).clamp(max=1.0)
        if float(cap_factor.detach()) < 1.0:
            fused = fused * cap_factor.to(dtype=fused.dtype)
    else:
        cap_factor = fused_norm.new_tensor(1.0)

    result_diagnostics = None
    if diagnostics:
        result_diagnostics = _diagnostics(
            hidden_states,
            {name: residuals[name] for name in names},
            fused,
            real_length,
            float(cap_factor.detach()),
            before_cap=float(fused_norm.detach()),
            hidden_norm=float(hidden_norm.detach()),
        )
    return fused, result_diagnostics


def _diagnostics(
    hidden_states: torch.Tensor,
    residuals: Mapping[str, torch.Tensor],
    fused: torch.Tensor,
    real_length: int,
    cap_factor: float,
    *,
    before_cap: float | None = None,
    hidden_norm: float | None = None,
) -> FusionDiagnostics:
    real_hidden = hidden_states[:, :real_length]
    real_fused = fused[:, :real_length]
    hidden_norm = (
        float(mean_token_l2(real_hidden).detach())
        if hidden_norm is None
        else hidden_norm
    )
    before_cap = (
        float(mean_token_l2(real_fused).detach())
        if before_cap is None
        else before_cap
    )
    denominator = real_hidden.float().norm(dim=-1).clamp_min(1e-12)
    ratios = (real_fused.float().norm(dim=-1) / denominator).flatten()
    quantiles = torch.quantile(
        ratios,
        torch.tensor([0.5, 0.95], device=ratios.device),
    )
    return FusionDiagnostics(
        per_expert_mean_norm={
            name: float(mean_token_l2(value[:, :real_length]).detach())
            for name, value in residuals.items()
        },
        fused_mean_norm_before_cap=before_cap,
        fused_mean_norm_after_cap=float(mean_token_l2(real_fused).detach()),
        hidden_mean_norm=hidden_norm,
        cap_factor=cap_factor,
        token_ratio_p50=float(quantiles[0].detach()),
        token_ratio_p95=float(quantiles[1].detach()),
        token_ratio_max=float(ratios.max().detach()),
    )
