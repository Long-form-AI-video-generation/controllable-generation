"""Shared-frame preprocessing tests that do not need WAN weights."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.control_integration.preprocessing import (
    ReferenceFrameSequence,
    build_matched_controls,
    decode_reference_frames,
)
from src.sketch_models.preprocessing import CannyConfig


class SharedPreprocessingTests(unittest.TestCase):
    def _video(self, root: Path, frame_count: int = 3) -> Path:
        path = root / "reference.avi"
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            8,
            (12, 8),
        )
        for value in range(frame_count):
            writer.write(np.full((8, 12, 3), value * 50, dtype=np.uint8))
        writer.release()
        return path

    def test_sequential_decode_tracks_padding_and_shared_indices(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            sequence = decode_reference_frames(self._video(Path(temporary)), frame_num=5)
        self.assertEqual(sequence.source_indices, (0, 1, 2, 2, 2))
        self.assertEqual(sequence.padded_positions, (3, 4))
        self.assertEqual(sequence.frames_bgr.shape[0], 5)

    def test_canny_uses_the_exact_shared_sequence(self) -> None:
        frames = np.zeros((5, 10, 10, 3), dtype=np.uint8)
        frames[:, :, 5:] = 255
        sequence = ReferenceFrameSequence(
            frames_bgr=frames,
            source_indices=(0, 1, 2, 3, 4),
            padded_positions=(),
            source_video_sha256="a" * 64,
            actual_frame_count=5,
            interval=(0, 5),
        )
        controls = build_matched_controls(
            sequence,
            experts=["canny"],
            canny_config=CannyConfig(num_frames=5, output_size=(8, 8)),
        )
        self.assertEqual(controls["canny"].shape, (1, 1, 5, 8, 8))
        self.assertEqual(controls["canny"].dtype, np.uint8)
        self.assertTrue(np.isin(controls["canny"], (0, 1)).all())

    def test_disabled_extractors_are_not_required(self) -> None:
        sequence = ReferenceFrameSequence(
            frames_bgr=np.zeros((1, 8, 8, 3), dtype=np.uint8),
            source_indices=(0,),
            padded_positions=(),
            source_video_sha256="a" * 64,
            actual_frame_count=1,
            interval=(0, 1),
        )
        with self.assertRaisesRegex(ValueError, "mask requires"):
            build_matched_controls(sequence, experts=["mask"])
