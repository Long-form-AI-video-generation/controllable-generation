"""CPU-only tests for strict multi-control checkpoint loading."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from src.control_integration.checkpoint_loading import (
    ZeroLinear,
    _instantiate_adapter,
    expected_control_metadata,
    load_expert_checkpoint,
    load_requested_experts,
)
from src.control_integration.contracts import EXPERT_SPECS, INJECTION_LAYERS


class CheckpointLoadingTests(unittest.TestCase):
    def _write_checkpoint(
        self,
        root: Path,
        name: str,
        *,
        dit_dim: int = 16,
        metadata: object = "default",
        include_zero_convs: bool = True,
    ) -> Path:
        spec = EXPERT_SPECS[name]
        adapter = _instantiate_adapter(spec, dit_dim)
        zero_convs = torch.nn.ModuleList(
            [ZeroLinear(dit_dim) for _ in INJECTION_LAYERS]
        )
        payload: dict[str, object] = {
            "model": adapter.state_dict(),
            "global_step": 42,
            "epoch": 3,
            "best_val_loss": 0.5,
        }
        if include_zero_convs:
            payload["zero_convs"] = zero_convs.state_dict()
        if metadata == "default":
            expected = expected_control_metadata(spec)
            if expected is not None:
                payload["control_metadata"] = dict(expected)
        else:
            payload["control_metadata"] = metadata
        path = root / f"{name}.pt"
        torch.save(payload, path)
        return path

    def test_valid_checkpoints_load_cpu_first(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {
                name: self._write_checkpoint(root, name)
                for name in ("depth", "canny", "mask")
            }
            loaded = load_requested_experts(paths)
            self.assertEqual(tuple(loaded), ("depth", "canny", "mask"))
            self.assertEqual({value.dit_dim for value in loaded.values()}, {16})
            self.assertTrue(all(not module.training for value in loaded.values() for module in (value.adapter, value.zero_convs)))
            self.assertTrue(all(parameter.device.type == "cpu" for value in loaded.values() for parameter in value.adapter.parameters()))

    def test_missing_or_bad_state_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing = self._write_checkpoint(
                root,
                "depth",
                include_zero_convs=False,
            )
            with self.assertRaisesRegex(ValueError, "zero_convs"):
                load_expert_checkpoint("depth", missing)

            path = self._write_checkpoint(root, "canny")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            payload["zero_convs"].pop("0.proj.bias")
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "state keys"):
                load_expert_checkpoint("canny", path)

    def test_wrong_metadata_and_reused_path_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wrong_metadata = self._write_checkpoint(
                root,
                "mask",
                metadata={"control_type": "canny"},
            )
            with self.assertRaisesRegex(ValueError, "metadata"):
                load_expert_checkpoint("mask", wrong_metadata)

            valid = self._write_checkpoint(root, "depth")
            with self.assertRaisesRegex(ValueError, "reused"):
                load_requested_experts({"depth": valid, "canny": valid})

    def test_experts_must_agree_on_dit_dimension(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            depth = self._write_checkpoint(root, "depth", dit_dim=16)
            canny = self._write_checkpoint(root, "canny", dit_dim=32)
            with self.assertRaisesRegex(ValueError, "disagree"):
                load_requested_experts({"depth": depth, "canny": canny})
