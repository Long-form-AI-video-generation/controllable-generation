import unittest

from src.mask_models.inference_config import (
    strength_tag,
    validate_inference_settings,
)


class MaskInferenceConfigTests(unittest.TestCase):
    def test_valid_frame_counts_and_strengths(self):
        self.assertEqual(
            validate_inference_settings(49, [1.0, 0.5, 0.25]),
            (1.0, 0.5, 0.25),
        )

    def test_invalid_frame_count_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_inference_settings(50, [0.5])

    def test_negative_or_empty_strengths_are_rejected(self):
        with self.assertRaises(ValueError):
            validate_inference_settings(49, [])
        with self.assertRaises(ValueError):
            validate_inference_settings(49, [-0.1])

    def test_output_tag(self):
        self.assertEqual(strength_tag(0.25), "0p25")


if __name__ == "__main__":
    unittest.main()
