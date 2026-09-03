import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

if cv2 is not None and hasattr(cv2, "Canny"):
    from src.sketch_models.prepare_dataset import prepare_record
    from src.sketch_models.preprocessing import CannyConfig


@unittest.skipUnless(
    cv2 is not None and hasattr(cv2, "Canny"),
    "A complete OpenCV installation is required",
)
class SketchPreparationTests(unittest.TestCase):
    def test_resume_validates_and_reuses_existing_tensor(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            videos = root / "videos"
            output = root / "output"
            videos.mkdir()
            (videos / "01.mp4").touch()
            target = output / "01" / (
                "shot_01_shot_0000_controls_encoded.npz"
            )
            target.parent.mkdir(parents=True)
            sketch = np.zeros((1, 1, 2, 8, 8), dtype=np.float32)
            sketch[:, :, :, 2:6, 4] = 1.0
            np.savez_compressed(target, sketch_encoded=sketch)

            result = prepare_record(
                {
                    "video_id": "01",
                    "shot_id": "01_shot_0000",
                    "segment_start_frame": 0,
                    "segment_end_frame": 8,
                },
                videos,
                output,
                CannyConfig(num_frames=2, output_size=(8, 8)),
                resume=True,
            )
            self.assertTrue(result["resumed"])
            self.assertEqual(result["shape"], [1, 1, 2, 8, 8])


if __name__ == "__main__":
    unittest.main()
