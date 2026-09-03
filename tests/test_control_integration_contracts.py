"""Tests for dependency-free multi-control contracts."""

from __future__ import annotations

import unittest

from src.control_integration.contracts import (
    CANONICAL_EXPERT_ORDER,
    EXPERT_SPECS,
    ActivationState,
    canonicalize_expert_names,
    combination_name,
    parse_control_combination,
)


SHA256 = "a" * 64


class ExpertContractTests(unittest.TestCase):
    def test_specs_have_the_expected_canonical_mapping(self) -> None:
        self.assertEqual(tuple(EXPERT_SPECS), CANONICAL_EXPERT_ORDER)
        self.assertEqual(EXPERT_SPECS["depth"].control_key, "depth_encoded")
        self.assertEqual(EXPERT_SPECS["canny"].control_key, "sketch_encoded")
        self.assertEqual(EXPERT_SPECS["mask"].control_key, "mask_encoded")

    def test_expert_names_are_canonicalized(self) -> None:
        self.assertEqual(
            canonicalize_expert_names(["mask", "depth"]),
            ("depth", "mask"),
        )
        self.assertEqual(
            parse_control_combination("mask+depth"),
            ("depth", "mask"),
        )
        self.assertEqual(combination_name(["mask", "depth"]), "depth+mask")

    def test_invalid_combination_is_rejected(self) -> None:
        for value in ("", "depth+", "depth+depth", "style"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_control_combination(value)


class ActivationStateTests(unittest.TestCase):
    def _state(self, **overrides: object) -> ActivationState:
        values: dict[str, object] = {
            "enabled_experts": ("depth", "mask"),
            "adapter_signals": {"depth": object(), "mask": object()},
            "strengths": {"depth": 0.5, "mask": 0.5},
            "ratio_caps": {"depth": 0.1, "mask": 0.1},
            "combined_ratio_cap": 0.1,
            "diagnostics_enabled": False,
            "control_artifact_sha256": SHA256,
            "generation_id": 3,
        }
        values.update(overrides)
        return ActivationState(**values)  # type: ignore[arg-type]

    def test_state_is_canonical_and_mapping_immutable(self) -> None:
        state = self._state(enabled_experts=("mask", "depth"))
        self.assertEqual(state.enabled_experts, ("depth", "mask"))
        with self.assertRaises(TypeError):
            state.strengths["depth"] = 1.0  # type: ignore[index]

    def test_state_requires_matching_expert_mappings(self) -> None:
        with self.assertRaisesRegex(ValueError, "adapter signals"):
            self._state(adapter_signals={"depth": object()})
        with self.assertRaisesRegex(ValueError, "strengths and ratio caps"):
            self._state(strengths={"depth": 0.5})

    def test_state_rejects_invalid_hash_or_cap(self) -> None:
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self._state(control_artifact_sha256="not-a-hash")
        with self.assertRaisesRegex(ValueError, "combined_ratio_cap"):
            self._state(combined_ratio_cap=0.0)
