import unittest

import numpy as np

from src.mask_models.semantic_groups import (
    SemanticGroup,
    apply_group_lookup,
    build_group_lookup,
    group_for_label,
)
from src.mask_models.temporal_stability import (
    adjacent_agreement,
    summarize_frame_composition,
    summarize_stability,
)


class SemanticGroupTests(unittest.TestCase):
    def test_known_and_unknown_labels(self):
        self.assertEqual(group_for_label("person"), SemanticGroup.PERSON)
        self.assertEqual(group_for_label("car"), SemanticGroup.VEHICLE)
        self.assertEqual(group_for_label("tank"), SemanticGroup.VEHICLE)
        self.assertEqual(group_for_label("wall"), SemanticGroup.STRUCTURE)
        self.assertEqual(group_for_label("pier"), SemanticGroup.STRUCTURE)
        self.assertEqual(group_for_label("ceiling"), SemanticGroup.OVERHEAD)
        self.assertEqual(group_for_label("roof"), SemanticGroup.OVERHEAD)
        self.assertEqual(group_for_label("chair"), SemanticGroup.OBJECT)

    def test_wall_and_ceiling_remain_distinct(self):
        lookup = build_group_lookup({0: "wall", 1: "ceiling"})
        grouped = apply_group_lookup(
            np.asarray([[0, 1]], dtype=np.int64),
            lookup,
        )
        np.testing.assert_array_equal(
            grouped,
            np.asarray(
                [[SemanticGroup.STRUCTURE, SemanticGroup.OVERHEAD]],
                dtype=np.uint8,
            ),
        )

    def test_lookup_accepts_transformers_string_keys(self):
        lookup = build_group_lookup({"0": "wall", "1": "building"})
        labels = np.asarray([[0, 1]], dtype=np.int64)
        grouped = apply_group_lookup(labels, lookup)
        np.testing.assert_array_equal(
            grouped,
            np.asarray([[SemanticGroup.STRUCTURE] * 2], dtype=np.uint8),
        )

    def test_lookup_rejects_missing_ids(self):
        with self.assertRaises(ValueError):
            build_group_lookup({0: "wall", 2: "person"})


class TemporalStabilityTests(unittest.TestCase):
    def test_adjacent_agreement(self):
        maps = np.asarray(
            [
                [[0, 0], [1, 1]],
                [[0, 1], [1, 1]],
                [[1, 1], [1, 1]],
            ]
        )
        np.testing.assert_allclose(adjacent_agreement(maps), [0.75, 0.75])

    def test_grouping_can_remove_within_group_flicker(self):
        raw = np.asarray(
            [
                [[0, 0], [1, 1]],
                [[1, 1], [0, 0]],
            ],
            dtype=np.int64,
        )
        lookup = build_group_lookup({0: "wall", 1: "building"})
        grouped = apply_group_lookup(raw, lookup)
        report = summarize_stability(raw, grouped)
        self.assertEqual(report["raw"]["mean_adjacent_agreement"], 0.0)
        self.assertEqual(report["coarse"]["mean_adjacent_agreement"], 1.0)
        self.assertEqual(report["coarse_agreement_improvement"], 1.0)

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            summarize_stability(np.zeros((2, 2, 2)), np.zeros((2, 2, 3)))

    def test_composition_explains_catch_all_collapse(self):
        raw = np.asarray([[0, 0], [0, 1]], dtype=np.uint8)
        grouped = np.asarray(
            [
                [SemanticGroup.OBJECT, SemanticGroup.OBJECT],
                [SemanticGroup.OBJECT, SemanticGroup.PERSON],
            ],
            dtype=np.uint8,
        )
        result = summarize_frame_composition(
            raw,
            grouped,
            {0: "painting", 1: "person"},
            collapse_threshold=0.7,
        )
        self.assertEqual(result["top_raw_classes"][0]["class_name"], "painting")
        self.assertEqual(
            result["top_raw_classes"][0]["coarse_group_name"],
            "object",
        )
        self.assertEqual(result["object_fraction"], 0.75)
        self.assertTrue(result["collapse_flag"])

    def test_composition_rejects_invalid_threshold(self):
        with self.assertRaises(ValueError):
            summarize_frame_composition(
                np.zeros((2, 2), dtype=np.uint8),
                np.zeros((2, 2), dtype=np.uint8),
                {0: "wall"},
                collapse_threshold=0.0,
            )


if __name__ == "__main__":
    unittest.main()
