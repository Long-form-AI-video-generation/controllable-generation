"""WAN-independent hook lifecycle and residual-injection tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from src.control_integration.checkpoint_loading import LoadedExpert, ZeroLinear
from src.control_integration.contracts import EXPERT_SPECS, INJECTION_LAYERS
from src.control_integration.hook_controller import MultiControlHookController


class FakePatchEmbedding(nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden[:, :8].transpose(1, 2).reshape(hidden.shape[0], 4, 2, 2, 2)


class FakeWAN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(dim=4)
        self.patch_embedding = FakePatchEmbedding()
        self.blocks = nn.ModuleList([nn.Identity() for _ in range(30)])

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        _ = self.patch_embedding(hidden)
        for block in self.blocks:
            hidden = block(hidden)
        return hidden


def _expert(name: str) -> LoadedExpert:
    adapter = nn.Identity()
    zero_convs = nn.ModuleList([ZeroLinear(4) for _ in INJECTION_LAYERS])
    for projection in zero_convs:
        projection.proj.weight.data.copy_(torch.eye(4))
        projection.proj.bias.data.fill_(1.0)
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / f"{name}.pt"
        path.write_bytes(b"test")
        return LoadedExpert(
            spec=EXPERT_SPECS[name],
            adapter=adapter,
            zero_convs=zero_convs,
            dit_dim=4,
            checkpoint_path=path,
            checkpoint_sha256="a" * 64,
            global_step=None,
            epoch=None,
            best_val_loss=None,
            control_metadata=None,
        )


class HookControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.wan = FakeWAN()
        self.controller = MultiControlHookController(
            self.wan,
            {"depth": _expert("depth"), "mask": _expert("mask")},
        )
        self.signal = torch.ones(1, 16 * 16, 4)

    def tearDown(self) -> None:
        self.controller.close()

    def test_attaches_one_grid_and_eight_injection_hooks(self) -> None:
        self.assertEqual(len(self.wan.patch_embedding._forward_hooks), 1)
        self.assertEqual(
            sum(len(self.wan.blocks[index]._forward_pre_hooks) for index in INJECTION_LAYERS),
            8,
        )
        self.assertFalse(any(key.startswith("wan.") for key in self.controller.state_dict()))

    def test_deactivated_controller_preserves_base_output(self) -> None:
        hidden = torch.randn(1, 10, 4)
        expected = self.wan(hidden)
        self.controller.deactivate_controls()
        actual = self.wan(hidden)
        self.assertTrue(torch.equal(actual, expected))

    def test_active_signal_is_batched_and_padding_is_not_modified(self) -> None:
        hidden = torch.randn(2, 10, 4)
        self.controller.activate_controls(
            enabled_experts=["depth", "mask"],
            adapter_signals={"depth": self.signal, "mask": self.signal},
            strengths={"depth": 0.5, "mask": 0.5},
            control_artifact_sha256="a" * 64,
            diagnostics_enabled=True,
        )
        result = self.wan(hidden)
        self.assertEqual(result.shape, hidden.shape)
        self.assertTrue(torch.equal(result[:, 8:], hidden[:, 8:]))
        self.assertIsNotNone(self.controller.wan_grid)
        self.assertEqual(set(self.controller.diagnostics().by_layer), set(INJECTION_LAYERS))

    def test_close_is_idempotent_and_restores_counts(self) -> None:
        model_id = id(self.wan)
        self.controller.close()
        self.controller.close()
        self.assertEqual(id(self.wan), model_id)
        self.assertEqual(len(self.wan.patch_embedding._forward_hooks), 0)
        self.assertEqual(
            sum(len(self.wan.blocks[index]._forward_pre_hooks) for index in INJECTION_LAYERS),
            0,
        )
