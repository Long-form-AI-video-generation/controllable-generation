import unittest
import numpy as np

from scripts.pose.prepare_video_pilot import build_sequence, render_pose
from src.pose_models.extraction.dwpose import CANONICAL_JOINT_NAMES


def result(point=None):
    points = [None] * 17
    points[0] = point
    return {'width': 100, 'height': 50, 'model_info': {'config_sha256': 'test'},
            'candidates': [{'joint_names': CANONICAL_JOINT_NAMES, 'xy_pixels': points}]}


class PoseVideoPilotTests(unittest.TestCase):
    def test_skeleton_map_tracks_source_location_and_channel(self):
        maps = render_pose(result([49.5, 24.5]), 50, 100, 1.0)
        self.assertEqual(maps.shape, (3, 50, 100))
        self.assertEqual(np.unravel_index(maps[0].argmax(), maps[0].shape), (24, 49))
        self.assertEqual(float(maps[1:].max()), 0.0)
        self.assertTrue(np.isfinite(maps).all())

    def test_sampling_preserves_indices_and_empty_frame(self):
        rows = [{'source_frame_index': i, 'result': result(None if i == 0 else [i, i])} for i in range(5)]
        maps, indices, _ = build_sequence(rows, 0, 5, 3, 50, 100, 1.0)
        self.assertEqual(indices, [0, 2, 4])
        self.assertEqual(maps.shape, (3, 3, 50, 100))
        self.assertEqual(float(maps[:, 0].max()), 0.0)
        self.assertGreater(float(maps[:, 1].max()), 0.0)
        with self.assertRaisesRegex(ValueError, 'missing source frames'):
            build_sequence(rows[:-1], 0, 5, 3, 50, 100, 1.0)

    def test_nonfinite_sigma_is_rejected(self):
        for sigma in (float('nan'), float('inf'), 0, -1):
            with self.assertRaises(ValueError):
                render_pose(result([10, 10]), 50, 100, sigma)

    def test_bad_coordinates_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'outside source frame'):
            render_pose(result([100, 2]), 50, 100, 1.0)


if __name__ == '__main__':
    unittest.main()
