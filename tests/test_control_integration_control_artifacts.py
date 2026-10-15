"""Tests for immutable prepared-control serialization and validation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.control_integration.control_artifacts import (
    array_content_sha256,
    load_prepared_control_bundle,
    write_prepared_control_bundle,
)


def _controls() -> dict[str, np.ndarray]:
    return {
        "depth": np.full((1, 1, 5, 4, 4), 0.5, dtype=np.float32),
        "canny": np.zeros((1, 1, 5, 4, 4), dtype=np.uint8),
        "mask": np.full((1, 1, 5, 4, 4), 12, dtype=np.uint8),
    }


class PreparedControlArtifactTests(unittest.TestCase):
    def test_round_trip_is_canonical_and_verified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "bundle"
            bundle = write_prepared_control_bundle(
                root,
                controls={"mask": _controls()["mask"], "depth": _controls()["depth"]},
                metadata={"source_video_sha256": "b" * 64, "frame_indices": [0, 1, 2, 3, 4]},
            )
            self.assertEqual(tuple(bundle.controls), ("depth", "mask"))
            self.assertEqual(bundle.controls["depth"].dtype, np.float32)
            loaded = load_prepared_control_bundle(
                root,
                expected_experts=("depth", "mask"),
                expected_frame_num=5,
            )
            self.assertEqual(loaded.artifact_sha256, bundle.artifact_sha256)

    def test_content_hash_captures_dtype_shape_and_bytes(self) -> None:
        array = np.zeros((1, 1, 1, 2, 2), dtype=np.uint8)
        changed_value = array.copy()
        changed_value[..., 0, 0] = 1
        self.assertNotEqual(
            array_content_sha256("canny", array),
            array_content_sha256("canny", changed_value),
        )
        self.assertNotEqual(
            array_content_sha256("canny", array),
            array_content_sha256("mask", array),
        )

    def test_mutated_metadata_or_controls_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "bundle"
            write_prepared_control_bundle(root, controls=_controls(), metadata={})
            metadata_path = root / "control_metadata.json"
            metadata = json.loads(metadata_path.read_text())
            metadata["control_names"] = ["depth"]
            metadata_path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "key order"):
                load_prepared_control_bundle(root)

    def test_invalid_control_contract_fails_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "bundle"
            bad = _controls()
            bad["canny"] = bad["canny"].astype(np.float32)
            with self.assertRaisesRegex(ValueError, "binary uint8"):
                write_prepared_control_bundle(root, controls=bad, metadata={})
            self.assertFalse(root.exists())
