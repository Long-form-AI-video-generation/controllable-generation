"""Tests for multi-control request validation without WAN dependencies."""

from __future__ import annotations

import unittest

from src.control_integration.inference_config import (
    build_inference_config,
    parse_output_size,
    validate_combinations,
)


class InferenceConfigurationTests(unittest.TestCase):
    def test_valid_request_is_normalized(self) -> None:
        config = build_inference_config(
            frame_num=81,
            steps=20,
            fps=16,
            size="480*832",
            control_combinations=["mask+depth", "canny"],
            strengths={"depth": 0.5, "canny": 0.5, "mask": 0.25},
            combined_ratio_cap=0.1,
        )
        self.assertEqual(config.combinations, (("depth", "mask"), ("canny",)))
        self.assertEqual(
            config.strengths,
            {"depth": 0.5, "canny": 0.5, "mask": 0.25},
        )

    def test_invalid_frame_shape_and_runtime_values_fail_fast(self) -> None:
        invalid = dict(
            frame_num=80,
            steps=20,
            fps=16,
            size="480*832",
            control_combinations=["depth"],
            strengths={"depth": 0.5},
            combined_ratio_cap=0.1,
        )
        with self.assertRaisesRegex(ValueError, "4n\\+1"):
            build_inference_config(**invalid)
        invalid["frame_num"] = 81
        invalid["steps"] = 0
        with self.assertRaisesRegex(ValueError, "steps"):
            build_inference_config(**invalid)

    def test_invalid_combinations_and_strengths_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_combinations(["depth+canny", "canny+depth"])
        with self.assertRaisesRegex(ValueError, "unknown control experts"):
            validate_combinations(["depth+style"])

        with self.assertRaisesRegex(ValueError, "missing strengths"):
            build_inference_config(
                frame_num=17,
                steps=1,
                fps=16,
                size="480*832",
                control_combinations=["depth+mask"],
                strengths={"depth": 0.5},
                combined_ratio_cap=0.1,
            )

    def test_size_and_combined_cap_are_strict(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported size"):
            parse_output_size("512*512")
        with self.assertRaisesRegex(ValueError, "combined_ratio_cap"):
            build_inference_config(
                frame_num=17,
                steps=1,
                fps=16,
                size="480*832",
                control_combinations=["depth"],
                strengths={"depth": 0.5},
                combined_ratio_cap=1.5,
            )
