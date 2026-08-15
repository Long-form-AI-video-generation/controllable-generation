"""Non-owning ControlNet-style hooks for one existing WAN DiT."""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import Mapping
import weakref

import torch
import torch.nn as nn

from .checkpoint_loading import LoadedExpert
from .contracts import (
    DEFAULT_RATIO_CAP,
    INJECTION_LAYERS,
    ActivationState,
    canonicalize_expert_names,
)
from .fusion import FusionDiagnostics, build_expert_residual, fuse_residuals


def _hook_count(module: nn.Module) -> tuple[int, int]:
    return (len(module._forward_pre_hooks), len(module._forward_hooks))


def _grid_length(grid: tuple[int, int, int]) -> int:
    return reduce(mul, grid, 1)


@dataclass(frozen=True)
class ControllerDiagnostics:
    """Diagnostics keyed by injection-layer index for one generation."""

    generation_id: int
    wan_grid: tuple[int, int, int] | None
    by_layer: Mapping[int, FusionDiagnostics]


class MultiControlHookController(nn.Module):
    """Attach independent expert residuals to an already-created WAN model.

    The controller owns only adapters and zero-conv projections.  Its WAN
    reference is a weak reference stored outside ``nn.Module`` registration, so
    the backbone cannot appear in ``state_dict()`` or be moved by ``.to()``.
    """

    def __init__(self, wan: nn.Module, experts: Mapping[str, LoadedExpert]) -> None:
        super().__init__()
        if not experts:
            raise ValueError("at least one loaded expert is required")
        names = canonicalize_expert_names(list(experts))
        dimensions = {experts[name].dit_dim for name in names}
        if len(dimensions) != 1:
            raise ValueError(f"experts disagree on DiT dimension: {dimensions}")
        self.dit_dim = dimensions.pop()
        self.expert_names = names
        self.adapters = nn.ModuleDict({name: experts[name].adapter for name in names})
        self.zero_convs = nn.ModuleDict(
            {name: experts[name].zero_convs for name in names}
        )
        self._control_keys = {name: experts[name].spec.control_key for name in names}
        self._ratio_caps = {
            name: experts[name].spec.standalone_ratio_cap for name in names
        }
        self._block_to_index: dict[int, int] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._baseline_hook_counts: dict[int, tuple[int, int]] = {}
        self._active: ActivationState | None = None
        self._wan_grid: tuple[int, int, int] | None = None
        self._diagnostics: dict[int, FusionDiagnostics] = {}
        self._next_generation_id = 0
        self._closed = False

        self._validate_wan_surface(wan)
        object.__setattr__(self, "_wan_ref", weakref.ref(wan))
        self.eval()
        self._attach_hooks(wan)

    @property
    def wan(self) -> nn.Module:
        wan = self._wan_ref()
        if wan is None:
            raise RuntimeError("the attached WAN model no longer exists")
        return wan

    @property
    def active_state(self) -> ActivationState | None:
        return self._active

    @property
    def wan_grid(self) -> tuple[int, int, int] | None:
        return self._wan_grid

    def _validate_wan_surface(self, wan: nn.Module) -> None:
        if not hasattr(wan, "patch_embedding"):
            raise ValueError("WAN model must expose patch_embedding")
        if not hasattr(wan, "blocks"):
            raise ValueError("WAN model must expose blocks")
        if len(wan.blocks) <= max(INJECTION_LAYERS):
            raise ValueError(
                "WAN model has too few blocks for the integration injection layers"
            )
        config = getattr(wan, "config", None)
        actual_dim = getattr(config, "dim", None)
        if actual_dim is None:
            raise ValueError("WAN model config must expose dim")
        if int(actual_dim) != self.dit_dim:
            raise ValueError(
                f"WAN dim {actual_dim} does not match expert dim {self.dit_dim}"
            )

    def _attach_hooks(self, wan: nn.Module) -> None:
        modules = [wan.patch_embedding] + [wan.blocks[index] for index in INJECTION_LAYERS]
        self._baseline_hook_counts = {id(module): _hook_count(module) for module in modules}
        try:
            self._handles.append(
                wan.patch_embedding.register_forward_hook(self._capture_grid_hook)
            )
            for projection_index, layer_index in enumerate(INJECTION_LAYERS):
                block = wan.blocks[layer_index]
                self._block_to_index[id(block)] = projection_index
                self._handles.append(
                    block.register_forward_pre_hook(self._injection_hook)
                )
        except BaseException:
            for handle in self._handles:
                handle.remove()
            self._handles.clear()
            raise

    def _capture_grid_hook(
        self,
        _module: nn.Module,
        _inputs: tuple[object, ...],
        output: torch.Tensor,
    ) -> None:
        if self._active is None:
            return
        if not isinstance(output, torch.Tensor) or output.ndim < 5:
            raise RuntimeError("WAN patch_embedding must return [B,C,T,H,W] tokens")
        grid = tuple(int(value) for value in output.shape[2:])
        if any(value <= 0 for value in grid):
            raise RuntimeError(f"invalid WAN patch grid {grid}")
        if self._wan_grid is None:
            self._wan_grid = grid
        elif self._wan_grid != grid:
            raise RuntimeError(
                f"WAN patch grid changed within one generation: {self._wan_grid} -> {grid}"
            )

    def _expand_cfg_batch(
        self,
        signal: torch.Tensor,
        batch_size: int,
        name: str,
    ) -> torch.Tensor:
        if signal.ndim != 3:
            raise ValueError(f"{name} adapter signal must be [B,S,C]")
        if signal.shape[0] == batch_size:
            return signal
        if signal.shape[0] == 1:
            return signal.expand(batch_size, -1, -1)
        raise ValueError(
            f"{name} signal batch {signal.shape[0]} is incompatible with WAN batch {batch_size}"
        )

    def _injection_hook(
        self,
        module: nn.Module,
        inputs: tuple[object, ...],
    ) -> tuple[object, ...]:
        state = self._active
        if state is None:
            return inputs
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            raise RuntimeError("WAN block pre-hook did not receive hidden states")
        if self._wan_grid is None:
            raise RuntimeError("WAN patch grid was not captured before injection")
        hidden_states = inputs[0]
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.dit_dim:
            raise RuntimeError("WAN hidden state shape is incompatible with controls")
        projection_index = self._block_to_index[id(module)]
        real_length = _grid_length(self._wan_grid)
        if real_length > hidden_states.shape[1]:
            raise RuntimeError("WAN patch grid exceeds the hidden-state sequence")

        residuals: dict[str, torch.Tensor] = {}
        for name in state.enabled_experts:
            signal = self._expand_cfg_batch(
                state.adapter_signals[name],
                hidden_states.shape[0],
                name,
            )
            token_count = signal.shape[1]
            if token_count % (16 * 16) != 0:
                raise ValueError(f"{name} signal has invalid token count {token_count}")
            source_grid = (token_count // (16 * 16), 16, 16)
            residuals[name] = build_expert_residual(
                hidden_states,
                signal,
                self.zero_convs[name][projection_index],
                source_grid=source_grid,
                target_grid=self._wan_grid,
                ratio_cap=state.ratio_caps[name],
                strength=state.strengths[name],
            )
        fused, diagnostics = fuse_residuals(
            hidden_states,
            residuals,
            real_length=real_length,
            combined_ratio_cap=state.combined_ratio_cap,
            diagnostics=state.diagnostics_enabled,
        )
        if diagnostics is not None:
            self._diagnostics[INJECTION_LAYERS[projection_index]] = diagnostics
        return (hidden_states + fused,) + inputs[1:]

    @torch.inference_mode()
    def compute_adapter_signals(
        self,
        controls: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute requested expert signals once, before the denoising loop."""

        unknown = sorted(set(controls).difference(self.expert_names))
        if unknown:
            raise ValueError(f"controls supplied for unloaded experts: {unknown}")
        signals: dict[str, torch.Tensor] = {}
        for name in self.expert_names:
            if name not in controls:
                continue
            adapter = self.adapters[name]
            adapter.eval()
            signals[name] = adapter({self._control_keys[name]: controls[name]})
        return signals

    def activate_controls(
        self,
        *,
        enabled_experts: tuple[str, ...] | list[str],
        adapter_signals: Mapping[str, torch.Tensor],
        strengths: Mapping[str, float],
        control_artifact_sha256: str,
        combined_ratio_cap: float = DEFAULT_RATIO_CAP,
        diagnostics_enabled: bool = False,
    ) -> ActivationState:
        """Install a fresh immutable activation snapshot for one generation."""

        if self._closed:
            raise RuntimeError("cannot activate a closed controller")
        enabled = canonicalize_expert_names(list(enabled_experts))
        unavailable = sorted(set(enabled).difference(self.expert_names))
        if unavailable:
            raise ValueError(f"requested experts are not loaded: {unavailable}")
        self.deactivate_controls()
        self._next_generation_id += 1
        state = ActivationState(
            enabled_experts=enabled,
            adapter_signals={name: adapter_signals[name] for name in enabled},
            strengths={name: strengths[name] for name in enabled},
            ratio_caps={name: self._ratio_caps[name] for name in enabled},
            combined_ratio_cap=combined_ratio_cap,
            diagnostics_enabled=diagnostics_enabled,
            control_artifact_sha256=control_artifact_sha256,
            generation_id=self._next_generation_id,
        )
        self._active = state
        return state

    def deactivate_controls(self) -> None:
        """Return the attached WAN exactly to its base no-control behavior."""

        self._active = None
        self._wan_grid = None
        self._diagnostics.clear()

    def diagnostics(self) -> ControllerDiagnostics:
        generation_id = 0 if self._active is None else self._active.generation_id
        return ControllerDiagnostics(
            generation_id=generation_id,
            wan_grid=self._wan_grid,
            by_layer=dict(self._diagnostics),
        )

    def close(self) -> None:
        """Remove only this controller's hooks and verify baseline restoration."""

        if self._closed:
            return
        self.deactivate_controls()
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._closed = True
        wan = self.wan
        modules = [wan.patch_embedding] + [wan.blocks[index] for index in INJECTION_LAYERS]
        mismatches = {
            id(module): (self._baseline_hook_counts[id(module)], _hook_count(module))
            for module in modules
            if self._baseline_hook_counts[id(module)] != _hook_count(module)
        }
        if mismatches:
            raise RuntimeError(f"hook counts were not restored: {mismatches}")

    def __enter__(self) -> "MultiControlHookController":
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()
