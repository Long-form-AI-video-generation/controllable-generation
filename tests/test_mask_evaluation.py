import unittest
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from src.mask_models.evaluate import semantic_metrics, semantic_temporal_agreement, video_health


class MaskEvaluationTests(unittest.TestCase):
    def test_exact_semantic_match(self):
        labels=np.asarray([[[0,0],[12,12]]],dtype=np.uint8)
        result=semantic_metrics(labels,labels)
        self.assertEqual(result["pixel_agreement"],1.0)
        self.assertEqual(result["macro_iou_present"],1.0)

    def test_mismatch_reduces_iou(self):
        reference=np.asarray([[[0,0],[12,12]]],dtype=np.uint8)
        generated=np.asarray([[[0,12],[12,12]]],dtype=np.uint8)
        result=semantic_metrics(reference,generated)
        self.assertLess(result["macro_iou_present"],1.0)
        self.assertAlmostEqual(result["pixel_agreement"],0.75)

    def test_video_health_reports_motion(self):
        frames=np.zeros((2,2,2,3),dtype=np.uint8); frames[1]=10
        self.assertEqual(video_health(frames)["temporal_mad"],10.0)

    def test_semantic_temporal_agreement(self):
        ids=np.zeros((3,2,2),dtype=np.uint8); ids[2]=12
        self.assertEqual(semantic_temporal_agreement(ids),0.5)

    def test_shape_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            semantic_metrics(np.zeros((1,2,2)),np.zeros((2,2,2)))

if __name__ == "__main__": unittest.main()
