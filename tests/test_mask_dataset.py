import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
    from src.data.dataset import ControllableVideoDataset
except ImportError:  # The lightweight local environment may not have PyTorch.
    torch = None
    ControllableVideoDataset = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class MaskDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.encoded = self.root / "encoded" / "01"
        self.encoded.mkdir(parents=True)
        self.videos = self.root / "videos"
        self.videos.mkdir()

        self.metadata = self.root / "shots_metadata.json"
        self.metadata.write_text(json.dumps([{
            "video_id": "01",
            "shot_id": "01_shot_0000",
            "segment_start_frame": 0,
            "segment_end_frame": 8,
            "descriptive_caption": "A character crosses a doorway.",
            "narrative_caption": "",
            "fps": 24.0,
        }]))

        self.manifest = self.root / "split_manifest.json"
        self.manifest.write_text(json.dumps({
            "train": ["01"],
            "val": [],
            "test": [],
        }))
        self.control_path = (
            self.encoded / "shot_01_shot_0000_controls_encoded.npz"
        )

    def tearDown(self):
        self.temporary.cleanup()

    def _dataset(self, **overrides):
        arguments = {
            "encoded_controls_dir": str(self.root / "encoded"),
            "videos_dir": str(self.videos),
            "annotations_path": str(self.metadata),
            "num_frames": 2,
            "resolution": (8, 8),
            "split": "train",
            "load_videos": False,
            "control_key": "mask_encoded",
            "strict": True,
            "split_manifest_path": str(self.manifest),
        }
        arguments.update(overrides)
        return ControllableVideoDataset(**arguments)

    def test_loads_only_one_channel_mask(self):
        mask = np.zeros((1, 1, 2, 8, 8), dtype=np.uint8)
        mask[:, :, :, 2:6, 3] = 12
        np.savez(self.control_path, mask_encoded=mask)

        sample = self._dataset()[0]
        self.assertEqual(set(sample["controls"]), {"mask_encoded"})
        self.assertEqual(tuple(sample["controls"]["mask_encoded"].shape),
                         (1, 2, 8, 8))
        self.assertEqual(sample["controls"]["mask_encoded"].dtype, torch.long)
        self.assertEqual(tuple(sample["video"].shape), (3, 2, 8, 8))

    def test_strict_mode_rejects_missing_mask(self):
        depth = np.zeros((1, 256, 2, 8, 8), dtype=np.float16)
        np.savez(self.control_path, depth_encoded=depth)

        with self.assertRaisesRegex(RuntimeError, "mask_encoded"):
            self._dataset()[0]

    def test_strict_mode_rejects_wrong_channel_count(self):
        mask = np.zeros((1, 2, 2, 8, 8), dtype=np.uint8)
        np.savez(self.control_path, mask_encoded=mask)

        with self.assertRaisesRegex(RuntimeError, "one channel"):
            self._dataset()[0]

    def test_depth_control_key_remains_available(self):
        depth = np.zeros((1, 256, 2, 8, 8), dtype=np.float16)
        np.savez(self.control_path, depth_encoded=depth)

        sample = self._dataset(control_key="depth_encoded")[0]
        self.assertEqual(set(sample["controls"]), {"depth_encoded"})
        self.assertEqual(tuple(sample["controls"]["depth_encoded"].shape),
                         (256, 2, 8, 8))

    def test_float_and_out_of_range_masks_are_rejected(self):
        np.savez(self.control_path, mask_encoded=np.zeros((1,1,2,8,8), dtype=np.float32))
        with self.assertRaisesRegex(RuntimeError, "integer IDs"):
            self._dataset()[0]
        np.savez(self.control_path, mask_encoded=np.full((1,1,2,8,8), 150, dtype=np.uint8))
        with self.assertRaisesRegex(RuntimeError, r"\[0,149\]"):
            self._dataset()[0]


if __name__ == "__main__":
    unittest.main()

