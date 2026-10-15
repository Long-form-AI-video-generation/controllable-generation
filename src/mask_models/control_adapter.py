"""Adapter for semantic mask edge conditioning."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .condition_encoder import MaskConditionEncoder


MASK_CONTROL_KEY = "mask_encoded"


class MaskControlAdapter(nn.Module):
    """Encode one semantic mask sequence and project it into WAN's DiT width."""

    def __init__(
        self,
        *,
        hidden_dim: int = 512,
        dit_dim: int = 2048,
        condition_dim: int = 256,
        target_spatial: int = 16,
        dropout: float = 0.1,
        use_gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or dit_dim <= 0 or condition_dim <= 0:
            raise ValueError("adapter dimensions must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")

        self.control_key = MASK_CONTROL_KEY
        self.target_spatial = target_spatial
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.condition_encoder = MaskConditionEncoder(
            out_channels=condition_dim,
            target_spatial=target_spatial,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )
        self.control_projection = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, dit_dim),
            nn.SiLU(),
            nn.LayerNorm(dit_dim),
            nn.Dropout(dropout),
        )
        self.modality_gate = nn.Parameter(torch.zeros(1))
        self._initialize_linear_weights()

    def _initialize_linear_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _project(self, features: torch.Tensor) -> torch.Tensor:
        batch, channels, frames, height, width = features.shape
        tokens = features.flatten(2).transpose(1, 2)
        projected = self.control_projection(tokens)
        projected = projected * torch.sigmoid(self.modality_gate)
        output = self.fusion(projected)
        expected = frames * height * width
        if output.shape != (batch, expected, output.shape[-1]):
            raise RuntimeError("unexpected adapter output shape")
        return output

    def forward(self, control_features: dict[str, torch.Tensor]) -> torch.Tensor:
        keys = set(control_features)
        if keys != {self.control_key}:
            raise ValueError(
                f"expected only {self.control_key!r}, got {sorted(keys)}"
            )
        control = control_features[self.control_key]
        encoded = self.condition_encoder(control)
        if self.use_gradient_checkpointing and self.training:
            return checkpoint(
                self._project,
                encoded,
                use_reentrant=False,
            )
        return self._project(encoded)

    def get_modality_weights(self) -> dict[str, float]:
        return {
            "mask": float(torch.sigmoid(self.modality_gate).detach().cpu())
        }
