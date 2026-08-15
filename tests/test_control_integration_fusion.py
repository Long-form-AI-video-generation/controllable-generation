"""Numerical contracts for bounded multi-control residual fusion."""

from __future__ import annotations

import unittest

import torch

from src.control_integration.fusion import (
    build_expert_residual,
    fuse_residuals,
    mean_token_l2,
)
from src.sketch_models.injection import build_padded_control_residual


class FusionTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.hidden = torch.randn(1, 10, 4)
        self.real_length = 8

    def test_no_controls_is_exact_identity_residual(self) -> None:
        fused, diagnostics = fuse_residuals(
            self.hidden,
            {},
            real_length=self.real_length,
            combined_ratio_cap=0.1,
            diagnostics=True,
        )
        self.assertTrue(torch.equal(fused, torch.zeros_like(self.hidden)))
        self.assertEqual(diagnostics.cap_factor, 1.0)

    def test_single_expert_below_global_cap_is_not_attenuated(self) -> None:
        residual = torch.full_like(self.hidden, 0.001)
        residual[:, self.real_length:] = 0
        fused, diagnostics = fuse_residuals(
            self.hidden,
            {"depth": residual},
            real_length=self.real_length,
            combined_ratio_cap=0.1,
            diagnostics=True,
        )
        self.assertTrue(torch.equal(fused, residual))
        self.assertEqual(diagnostics.cap_factor, 1.0)

    def test_aligned_controls_are_globally_capped(self) -> None:
        residual = torch.ones_like(self.hidden)
        residual[:, self.real_length:] = 0
        fused, _ = fuse_residuals(
            self.hidden,
            {"depth": residual, "canny": residual, "mask": residual},
            real_length=self.real_length,
            combined_ratio_cap=0.1,
        )
        self.assertLessEqual(
            float(mean_token_l2(fused[:, :self.real_length])),
            0.1 * float(mean_token_l2(self.hidden[:, :self.real_length])) + 1e-6,
        )
        self.assertTrue(torch.equal(fused[:, self.real_length:], torch.zeros_like(fused[:, self.real_length:])))

    def test_opposing_controls_cancel_and_do_not_produce_nan(self) -> None:
        residual = torch.randn_like(self.hidden)
        residual[:, self.real_length:] = 0
        fused, _ = fuse_residuals(
            self.hidden,
            {"depth": residual, "mask": -residual},
            real_length=self.real_length,
            combined_ratio_cap=0.1,
        )
        self.assertTrue(torch.equal(fused, torch.zeros_like(fused)))
        self.assertFalse(torch.isnan(fused).any())

    def test_order_is_canonical_and_padding_is_enforced(self) -> None:
        first = torch.full_like(self.hidden, 0.01)
        second = torch.full_like(self.hidden, 0.02)
        first[:, self.real_length:] = 0
        second[:, self.real_length:] = 0
        left, _ = fuse_residuals(
            self.hidden,
            {"mask": second, "depth": first},
            real_length=self.real_length,
            combined_ratio_cap=1.0,
        )
        right, _ = fuse_residuals(
            self.hidden,
            {"depth": first, "mask": second},
            real_length=self.real_length,
            combined_ratio_cap=1.0,
        )
        self.assertTrue(torch.equal(left, right))

        first[:, self.real_length:] = 1
        with self.assertRaisesRegex(ValueError, "padding"):
            fuse_residuals(
                self.hidden,
                {"depth": first},
                real_length=self.real_length,
                combined_ratio_cap=0.1,
            )

    def test_safe_projection_matches_single_control_helper(self) -> None:
        control = torch.randn(1, 4, 4)
        projection = torch.nn.Linear(4, 4, bias=True)
        target_grid = (2, 2, 2)
        expected = build_padded_control_residual(
            self.hidden,
            control,
            projection,
            source_grid=(1, 2, 2),
            target_grid=target_grid,
            ratio_cap=0.1,
            control_strength=0.5,
        )
        actual = build_expert_residual(
            self.hidden,
            control,
            projection,
            source_grid=(1, 2, 2),
            target_grid=target_grid,
            ratio_cap=0.1,
            strength=0.5,
        )
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(actual[:, 8:], torch.zeros_like(actual[:, 8:])))
