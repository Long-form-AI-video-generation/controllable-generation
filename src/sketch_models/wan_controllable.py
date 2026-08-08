"""Sketch-specific one-control WAN wrapper built on the merged depth wrapper."""

from __future__ import annotations

import torch

from src.depth_models.wan_controllable import ControllableWAN as DepthControllableWAN

from .control_adapter import SketchControlAdapter
from .injection import (
    CFG_CONTROL_POLICY,
    build_padded_control_residual,
    validate_cfg_control_policy,
)


class ControllableWAN(DepthControllableWAN):
    """WAN 2.2 with one checkpointed Canny condition path."""

    def __init__(
        self,
        checkpoint_dir: str,
        device: str = "cuda",
        control_injection_layers: tuple[int, ...] = (0, 4, 8, 12, 16, 20, 24, 28),
        *,
        cfg_control_policy: str = CFG_CONTROL_POLICY,
        control_ratio_cap: float = 0.1,
    ) -> None:
        self.cfg_control_policy = validate_cfg_control_policy(cfg_control_policy)
        self.control_ratio_cap = float(control_ratio_cap)
        if self.control_ratio_cap < 0.0:
            raise ValueError("control_ratio_cap must be non-negative")

        super().__init__(
            checkpoint_dir=checkpoint_dir,
            device=device,
            control_injection_layers=list(control_injection_layers),
        )

        dit_dim = self.wan.config.dim
        self.control_adapter = SketchControlAdapter(
            hidden_dim=512,
            dit_dim=dit_dim,
            condition_dim=256,
            target_spatial=16,
            use_gradient_checkpointing=False,
        ).to(device)
        self._control_strength = 1.0

    def train(self, mode: bool = True):
        """Train sketch modules while keeping every pretrained component in eval."""
        super().train(mode)
        for frozen_component in (
            self.wan,
            getattr(self.vae, "model", None),
            getattr(self.text_encoder, "model", None),
        ):
            if frozen_component is not None:
                frozen_component.eval()
        self.control_adapter.train(mode)
        self.zero_convs.train(mode)
        return self

    def _control_injection_hook(self, module, inputs):
        if self._control_signal is None:
            return inputs
        if self._current_wan_grid is None:
            raise RuntimeError("WAN patch grid was not captured")

        hidden_states = inputs[0]
        hook_index = self._block_to_hook_idx[id(module)]
        source_tokens = self._control_signal
        if source_tokens.shape[1] % (16 * 16) != 0:
            raise RuntimeError(
                f"invalid sketch token count: {source_tokens.shape[1]}"
            )
        source_grid = (source_tokens.shape[1] // (16 * 16), 16, 16)
        residual = build_padded_control_residual(
            hidden_states,
            source_tokens,
            self.zero_convs[hook_index],
            source_grid=source_grid,
            target_grid=self._current_wan_grid,
            ratio_cap=self.control_ratio_cap,
            control_strength=float(self._control_strength),
        )
        return (hidden_states + residual,) + inputs[1:]

    def get_trainable_parameter_groups(self) -> list[dict]:
        gate_params = [self.control_adapter.modality_gate]
        adapter_params = [
            parameter
            for name, parameter in self.control_adapter.named_parameters()
            if name != "modality_gate"
        ]
        return [
            {"params": adapter_params, "name": "control_adapter"},
            {"params": list(self.zero_convs.parameters()), "name": "zero_convs"},
            {"params": gate_params, "name": "modality_gates"},
        ]

    def checkpoint_metadata(self) -> dict[str, object]:
        return {
            "control_type": "canny",
            "control_key": "sketch_encoded",
            "input_channels": 1,
            "condition_encoder": "SketchConditionEncoder",
            "condition_dim": 256,
            "target_spatial": 16,
            "injection_layers": list(self.control_injection_layers),
            "cfg_control_policy": self.cfg_control_policy,
            "control_ratio_cap": self.control_ratio_cap,
        }
