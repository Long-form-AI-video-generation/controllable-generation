from __future__ import annotations

import importlib.util
import unittest
from types import SimpleNamespace

import numpy as np

from src.mask_models.labels import ADE20K_LABELS
from src.mask_models.preprocessing import (
    MASK_CONTROL_KEY,
    PREPROCESSING_VERSION,
    SEGFORMER_MODEL_ID,
    SEGFORMER_REVISION,
    SegFormerMaskConfig,
    logits_to_id_maps,
    processor_metadata,
    validate_model_contract,
    validate_rgb_frames,
)


class MaskPreprocessingContractTests(unittest.TestCase):
    def test_default_metadata_is_complete_and_pinned(self) -> None:
        config = SegFormerMaskConfig()
        metadata = config.to_metadata()
        self.assertEqual(metadata["control_key"], MASK_CONTROL_KEY)
        self.assertEqual(metadata["model_id"], SEGFORMER_MODEL_ID)
        self.assertEqual(metadata["revision"], SEGFORMER_REVISION)
        self.assertEqual(metadata["weights_format"], "safetensors")
        self.assertEqual(metadata["num_classes"], 150)
        self.assertEqual(metadata["output_size"], [128, 128])
        self.assertEqual(
            metadata["categorical_conversion"],
            "argmax_after_logit_resize",
        )
        self.assertEqual(
            metadata["neural_representation"],
            "trainable_embedding_150x16",
        )
        self.assertEqual(metadata["preprocessing_version"], PREPROCESSING_VERSION)

    def test_invalid_configuration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SegFormerMaskConfig(num_frames=0)
        with self.assertRaises(ValueError):
            SegFormerMaskConfig(output_size=(0, 128))
        with self.assertRaises(ValueError):
            SegFormerMaskConfig(batch_size=0)
        with self.assertRaisesRegex(ValueError, "unsupported mask extractor"):
            SegFormerMaskConfig(model_id="another/model")
        with self.assertRaisesRegex(ValueError, "unsupported SegFormer revision"):
            SegFormerMaskConfig(revision="main")

    def test_rgb_frame_contract(self) -> None:
        frames = np.zeros((8, 32, 48, 3), dtype=np.uint8)
        self.assertIs(validate_rgb_frames(frames, expected_frames=8), frames)
        with self.assertRaises(TypeError):
            validate_rgb_frames(frames.astype(np.float32))
        with self.assertRaises(ValueError):
            validate_rgb_frames(frames[..., 0])
        with self.assertRaisesRegex(ValueError, "expected 7 frames"):
            validate_rgb_frames(frames, expected_frames=7)

    def test_processor_metadata_is_canonical(self) -> None:
        class FakeProcessor:
            def to_dict(self):
                return {"size": {"height": 640, "width": 640}, "do_resize": True}

        first = processor_metadata(FakeProcessor())
        second = processor_metadata(FakeProcessor())
        self.assertEqual(first, second)
        self.assertEqual(len(first["sha256"]), 64)

    def test_model_contract_accepts_exact_transformers_labels(self) -> None:
        model = SimpleNamespace(
            config=SimpleNamespace(
                num_labels=150,
                id2label={str(i): label for i, label in enumerate(ADE20K_LABELS)},
            )
        )
        validate_model_contract(model)

    def test_model_contract_rejects_changed_vocabulary(self) -> None:
        labels = {str(i): label for i, label in enumerate(ADE20K_LABELS)}
        labels["12"] = "human"
        model = SimpleNamespace(
            config=SimpleNamespace(num_labels=150, id2label=labels)
        )
        with self.assertRaisesRegex(ValueError, "mismatch at ID 12"):
            validate_model_contract(model)


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is unavailable")
class MaskLogitConversionTests(unittest.TestCase):
    def test_logits_are_resized_before_argmax(self) -> None:
        import torch
        import torch.nn.functional as functional

        logits = torch.full((1, 150, 2, 2), -10.0)
        logits[:, 0] = torch.tensor([[[-1.0, 3.0], [3.0, -1.0]]])
        logits[:, 12] = torch.tensor([[[3.0, -1.0], [-1.0, 3.0]]])

        actual = logits_to_id_maps(logits, (5, 7))
        expected = functional.interpolate(
            logits.float(),
            size=(5, 7),
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1).to(torch.uint8)

        self.assertEqual(tuple(actual.shape), (1, 5, 7))
        self.assertEqual(actual.dtype, torch.uint8)
        self.assertTrue(torch.equal(actual, expected))

    def test_invalid_logits_are_rejected(self) -> None:
        import torch

        with self.assertRaisesRegex(ValueError, "150 logit channels"):
            logits_to_id_maps(torch.zeros(1, 149, 2, 2), (4, 4))
        with self.assertRaises(TypeError):
            logits_to_id_maps(torch.zeros(1, 150, 2, 2, dtype=torch.int64), (4, 4))
        bad = torch.zeros(1, 150, 2, 2)
        bad[0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            logits_to_id_maps(bad, (4, 4))


if __name__ == "__main__":
    unittest.main()
