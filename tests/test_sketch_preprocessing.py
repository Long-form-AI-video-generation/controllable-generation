import unittest

import cv2
import numpy as np

from src.data.frame_sampling import resolve_frame_interval
from src.sketch_models.preprocessing import (
    CannyConfig,
    extract_canny,
    prepare_canny_sequence,
    select_frame_indices,
    validate_canny_tensor,
)


class SketchPreprocessingTests(unittest.TestCase):
    def test_blank_image_produces_empty_edges(self):
        config = CannyConfig(num_frames=2, output_size=(32, 32))
        frames = np.zeros((2, 48, 48, 3), dtype=np.uint8)
        result = prepare_canny_sequence(frames, config)
        self.assertEqual(result.shape, (1, 1, 2, 32, 32))
        self.assertEqual(np.count_nonzero(result), 0)

    def test_square_edges_survive_nearest_resize(self):
        config = CannyConfig(num_frames=1, output_size=(32, 32))
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        cv2.rectangle(frame, (16, 16), (48, 48), (255, 255, 255), 2)
        edges = extract_canny(frame, config)
        self.assertGreater(np.count_nonzero(edges), 0)
        self.assertTrue(np.isin(np.unique(edges), (0.0, 1.0)).all())

    def test_threshold_validation(self):
        with self.assertRaises(ValueError):
            CannyConfig(low_threshold=200, high_threshold=100)

    def test_frame_selection_is_deterministic_and_pads(self):
        first = select_frame_indices(3, 6, 5)
        second = select_frame_indices(3, 6, 5)
        np.testing.assert_array_equal(first, [3, 4, 5, 5, 5])
        np.testing.assert_array_equal(first, second)

    def test_metadata_interval_is_clamped_to_actual_video_length(self):
        self.assertEqual(resolve_frame_interval(0, 192, 128), (0, 128))
        np.testing.assert_array_equal(
            select_frame_indices(*resolve_frame_interval(0, 192, 128), 8),
            [0, 18, 36, 54, 72, 90, 108, 127],
        )

    def test_interval_rejects_start_beyond_actual_video(self):
        with self.assertRaisesRegex(ValueError, "outside a video"):
            resolve_frame_interval(128, 192, 128)

    def test_rgb_and_bgr_are_explicitly_equivalent(self):
        rgb = np.zeros((40, 40, 3), dtype=np.uint8)
        rgb[8:32, 12:28] = (255, 80, 10)
        bgr = rgb[..., ::-1]
        rgb_edges = extract_canny(
            rgb,
            CannyConfig(num_frames=1, output_size=(40, 40), color_order="RGB"),
        )
        bgr_edges = extract_canny(
            bgr,
            CannyConfig(num_frames=1, output_size=(40, 40), color_order="BGR"),
        )
        np.testing.assert_array_equal(rgb_edges, bgr_edges)

    def test_repeated_calls_are_byte_identical(self):
        rng = np.random.default_rng(42)
        frames = rng.integers(0, 256, (5, 48, 48, 3), dtype=np.uint8)
        config = CannyConfig(num_frames=3, output_size=(24, 24))
        first = prepare_canny_sequence(frames, config)
        second = prepare_canny_sequence(frames, config)
        self.assertEqual(first.tobytes(), second.tobytes())

    def test_invalid_nonbinary_tensor_is_rejected(self):
        invalid = np.full((1, 1, 2, 8, 8), 0.5, dtype=np.float32)
        with self.assertRaises(ValueError):
            validate_canny_tensor(invalid)


if __name__ == "__main__":
    unittest.main()
