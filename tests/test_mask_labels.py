from __future__ import annotations

import unittest

import numpy as np

from src.mask_models.labels import (
    ADE20K_LABELS,
    ADE20K_LABEL_ORDER_SHA256,
    ADE20K_VISUALIZATION_PALETTE,
    ADE20K_VISUALIZATION_PALETTE_SHA256,
    colorize_id_map,
    palette_collisions,
    validate_id2label,
    validate_numpy_id_map,
)


class MaskLabelContractTests(unittest.TestCase):
    def test_pinned_tables_and_hashes(self) -> None:
        self.assertEqual(len(ADE20K_LABELS), 150)
        self.assertEqual(len(ADE20K_VISUALIZATION_PALETTE), 150)
        self.assertEqual(ADE20K_LABELS[0], "wall")
        self.assertEqual(ADE20K_LABELS[12], "person")
        self.assertEqual(ADE20K_LABELS[149], "flag")
        self.assertEqual(
            ADE20K_LABEL_ORDER_SHA256,
            "11a038c43840900c9b815faaca91182dc55ae5f8968deb7bb4d48c217233f4ff",
        )
        self.assertEqual(
            ADE20K_VISUALIZATION_PALETTE_SHA256,
            "ce475c9151035eff51f4cf27fd7d906959b1480f6d15b9423ae2417c1b23a0ed",
        )

    def test_string_keyed_transformers_mapping_is_accepted(self) -> None:
        mapping = {str(index): label for index, label in enumerate(ADE20K_LABELS)}
        self.assertEqual(validate_id2label(mapping), ADE20K_LABELS)

    def test_missing_extra_and_changed_labels_are_rejected(self) -> None:
        missing = {index: label for index, label in enumerate(ADE20K_LABELS)}
        missing.pop(149)
        with self.assertRaisesRegex(ValueError, r"missing=\[149\]"):
            validate_id2label(missing)

        extra = {index: label for index, label in enumerate(ADE20K_LABELS)}
        extra[150] = "extra"
        with self.assertRaisesRegex(ValueError, r"extra=\[150\]"):
            validate_id2label(extra)

        changed = {index: label for index, label in enumerate(ADE20K_LABELS)}
        changed[12] = "human"
        with self.assertRaisesRegex(ValueError, "mismatch at ID 12"):
            validate_id2label(changed)

    def test_visualization_collision_is_explicit(self) -> None:
        self.assertEqual(palette_collisions(), {(140, 140, 140): (6, 48)})

    def test_integer_map_validation_and_colorization(self) -> None:
        labels = np.asarray([[0, 12], [48, 149]], dtype=np.uint8)
        validated = validate_numpy_id_map(labels, expected_ndim=2)
        self.assertIs(validated, labels)

        colored = colorize_id_map(labels)
        self.assertEqual(colored.shape, (2, 2, 3))
        self.assertEqual(colored.dtype, np.uint8)
        np.testing.assert_array_equal(
            colored[0, 1],
            np.asarray(ADE20K_VISUALIZATION_PALETTE[12], dtype=np.uint8),
        )

    def test_invalid_maps_are_rejected(self) -> None:
        with self.assertRaises(TypeError):
            validate_numpy_id_map(np.zeros((2, 2), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, r"\[0,149\]"):
            validate_numpy_id_map(np.asarray([[150]], dtype=np.int16))
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            validate_numpy_id_map(np.empty((0, 2), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "expected a 3D"):
            validate_numpy_id_map(
                np.zeros((2, 2), dtype=np.uint8),
                expected_ndim=3,
            )


if __name__ == "__main__":
    unittest.main()
