import unittest

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

if cv2 is not None and hasattr(cv2, "distanceTransform"):
    from src.sketch_models.evaluate import (
        edge_pair_metrics,
        edge_sequence_metrics,
    )


@unittest.skipUnless(
    cv2 is not None and hasattr(cv2, "distanceTransform"),
    "A complete OpenCV installation is required",
)
class SketchEvaluationTests(unittest.TestCase):
    @staticmethod
    def line(column: int) -> np.ndarray:
        edge = np.zeros((32, 32), dtype=bool)
        edge[4:28, column] = True
        return edge

    def test_exact_match(self):
        metrics = edge_pair_metrics(self.line(10), self.line(10), tolerance=1)
        self.assertAlmostEqual(metrics["f1"], 1.0)
        self.assertAlmostEqual(metrics["symmetric_chamfer"], 0.0)

    def test_shift_within_tolerance(self):
        metrics = edge_pair_metrics(self.line(10), self.line(12), tolerance=2)
        self.assertAlmostEqual(metrics["f1"], 1.0)
        self.assertGreater(metrics["symmetric_chamfer"], 0.0)

    def test_shift_outside_tolerance(self):
        metrics = edge_pair_metrics(self.line(10), self.line(16), tolerance=2)
        self.assertEqual(metrics["f1"], 0.0)

    def test_empty_cases_are_explicit(self):
        empty = np.zeros((32, 32), dtype=bool)
        both = edge_pair_metrics(empty, empty)
        missing = edge_pair_metrics(self.line(10), empty)
        extra = edge_pair_metrics(empty, self.line(10))
        self.assertEqual(both["status"], "both_empty")
        self.assertIsNone(both["f1"])
        self.assertEqual(missing["status"], "empty_prediction")
        self.assertEqual(extra["status"], "empty_reference")

    def test_multiframe_aggregation(self):
        reference = np.stack([self.line(10), self.line(10)])
        prediction = np.stack([self.line(10), self.line(16)])
        metrics = edge_sequence_metrics(reference, prediction, tolerance=2)
        self.assertAlmostEqual(metrics["aggregate"]["f1"], 0.5)
        self.assertEqual(len(metrics["per_frame"]), 2)


if __name__ == "__main__":
    unittest.main()
