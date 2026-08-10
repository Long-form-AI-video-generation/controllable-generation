"""Trainable spatio-temporal encoder for sparse semantic mask edge sequences."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _group_count(channels: int) -> int:
    return next(
        groups for groups in range(min(8, channels), 0, -1)
        if channels % groups == 0
    )


class ConvBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: tuple[int, int, int] = (1, 3, 3),
        stride: tuple[int, int, int] = (1, 1, 1),
        padding: tuple[int, int, int] = (0, 1, 1),
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TemporalRefinement(nn.Module):
    """Refine time independently at every spatial position."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv3d(
            channels,
            channels,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.activation(self.norm(x) + residual)


class MaskConditionEncoder(nn.Module):
    """Map ``[B,1,T,H,W]`` semantic mask controls to ``[B,256,T,16,16]``."""

    def __init__(
        self,
        *,
        out_channels: int = 256,
        num_classes: int = 150,
        embedding_dim: int = 16,
        target_spatial: int = 16,
        use_gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if out_channels <= 0:
            raise ValueError("out_channels must be positive")
        if target_spatial <= 0:
            raise ValueError("target_spatial must be positive")

        if num_classes <= 1 or embedding_dim <= 0:
            raise ValueError("num_classes and embedding_dim must be positive")
        self.out_channels = out_channels
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.target_spatial = target_spatial
        self.use_gradient_checkpointing = use_gradient_checkpointing

        self.class_embedding = nn.Embedding(num_classes, embedding_dim)
        self.stem = ConvBlock3d(embedding_dim, 32)
        self.downsample = nn.Sequential(
            ConvBlock3d(32, 64, stride=(1, 2, 2)),
            ConvBlock3d(64, 128, stride=(1, 2, 2)),
            ConvBlock3d(128, out_channels, stride=(1, 2, 2)),
        )
        self.temporal = TemporalRefinement(out_channels)
        self.output_projection = nn.Conv3d(out_channels, out_channels, kernel_size=1)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _validate_input(self, control: torch.Tensor) -> None:
        if control.ndim != 5:
            raise ValueError(
                f"expected mask control [B,1,T,H,W], got {tuple(control.shape)}"
            )
        if control.shape[1] != 1:
            raise ValueError(
                f"expected one mask channel, got {control.shape[1]}"
            )
        if control.shape[2] <= 0 or control.shape[3] <= 0 or control.shape[4] <= 0:
            raise ValueError("mask control dimensions must be non-empty")
        if control.dtype != torch.long:
            raise TypeError(f"mask control must be torch.long, got {control.dtype}")
        minimum = int(control.detach().amin())
        maximum = int(control.detach().amax())
        if minimum < 0 or maximum >= self.num_classes:
            raise ValueError(
                f"mask class IDs must be in [0,{self.num_classes - 1}], got [{minimum},{maximum}]"
            )

    def _encode(self, control: torch.Tensor) -> torch.Tensor:
        ids = control[:, 0]
        x = self.class_embedding(ids).permute(0, 4, 1, 2, 3).contiguous()
        x = self.stem(x)
        x = self.downsample(x)
        if x.shape[-2:] != (self.target_spatial, self.target_spatial):
            x = F.interpolate(
                x,
                size=(x.shape[2], self.target_spatial, self.target_spatial),
                mode="trilinear",
                align_corners=False,
            )
        x = self.temporal(x)
        return self.output_projection(x)

    def forward(self, control: torch.Tensor) -> torch.Tensor:
        self._validate_input(control)
        if self.use_gradient_checkpointing and self.training:
            return checkpoint(
                self._encode,
                control,
                use_reentrant=False,
            )
        return self._encode(control)

