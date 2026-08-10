"""Deterministic raw SegFormer-B5 semantic-mask preprocessing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .labels import (
    ADE20K_LABEL_ORDER_SHA256,
    ADE20K_VISUALIZATION_PALETTE_SHA256,
    NUM_ADE20K_CLASSES,
    validate_id2label,
    validate_numpy_id_map,
)

SEGFORMER_MODEL_ID = "nvidia/segformer-b5-finetuned-ade-640-640"
SEGFORMER_REVISION = "739f5d4692954e4a185eac280dec1ba5a7d52f1d"
PREPROCESSING_VERSION = "segformer-b5-ade20k-raw-v1"
MASK_CONTROL_KEY = "mask_encoded"


def _positive_pair(values: tuple[int, int], name: str) -> tuple[int, int]:
    if len(values) != 2:
        raise ValueError(f"{name} must contain (height,width)")
    result = tuple(int(value) for value in values)
    if any(value <= 0 for value in result):
        raise ValueError(f"{name} values must be positive, got {result}")
    return result


@dataclass(frozen=True)
class SegFormerMaskConfig:
    num_frames: int = 8
    output_size: tuple[int, int] = (128, 128)
    batch_size: int = 4
    model_id: str = SEGFORMER_MODEL_ID
    revision: str = SEGFORMER_REVISION
    preprocessing_version: str = PREPROCESSING_VERSION

    def __post_init__(self) -> None:
        if self.num_frames <= 0 or self.batch_size <= 0:
            raise ValueError("num_frames and batch_size must be positive")
        object.__setattr__(self, "output_size", _positive_pair(self.output_size, "output_size"))
        if self.model_id != SEGFORMER_MODEL_ID:
            raise ValueError(f"unsupported mask extractor {self.model_id!r}")
        if self.revision != SEGFORMER_REVISION:
            raise ValueError(f"unsupported SegFormer revision {self.revision!r}")
        if self.preprocessing_version != PREPROCESSING_VERSION:
            raise ValueError("unsupported preprocessing version")

    def to_metadata(self) -> dict[str, Any]:
        return {
            "control_key": MASK_CONTROL_KEY, "model_id": self.model_id,
            "revision": self.revision, "weights_format": "safetensors",
            "num_classes": NUM_ADE20K_CLASSES, "num_frames": self.num_frames,
            "output_size": list(self.output_size), "batch_size": self.batch_size,
            "resize_logits": "bilinear_align_corners_false_float32",
            "categorical_conversion": "argmax_after_logit_resize",
            "stored_dtype": "uint8", "label_order_sha256": ADE20K_LABEL_ORDER_SHA256,
            "visualization_palette_sha256": ADE20K_VISUALIZATION_PALETTE_SHA256,
            "neural_representation": "trainable_embedding_150x16",
            "preprocessing_version": self.preprocessing_version,
        }


def validate_rgb_frames(frames: Any, *, expected_frames: int | None = None) -> Any:
    import numpy as np
    array = np.asarray(frames)
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError(f"expected RGB frames [T,H,W,3], got {array.shape}")
    if not all(array.shape[:3]) or (expected_frames is not None and len(array) != expected_frames):
        raise ValueError(f"expected {expected_frames} frames, got {len(array)}")
    if array.dtype != np.uint8:
        raise TypeError(f"RGB frames must be uint8, got {array.dtype}")
    return array


def logits_to_id_maps(logits: Any, output_size: tuple[int, int]) -> Any:
    import torch
    import torch.nn.functional as F
    if not isinstance(logits, torch.Tensor):
        raise TypeError("logits must be a torch.Tensor")
    if logits.ndim != 4 or logits.shape[1] != NUM_ADE20K_CLASSES:
        raise ValueError(f"expected 150 logit channels in [B,150,H,W], got {tuple(logits.shape)}")
    if not torch.is_floating_point(logits):
        raise TypeError("logits must be floating point")
    if not torch.isfinite(logits).all():
        raise ValueError("logits contain NaN or Inf")
    return F.interpolate(logits.float(), size=_positive_pair(output_size, "output_size"), mode="bilinear", align_corners=False).argmax(1).to(torch.uint8)


def processor_metadata(processor: Any) -> dict[str, Any]:
    settings = processor.to_dict()
    payload = json.dumps(settings, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str).encode()
    return {"settings": settings, "sha256": hashlib.sha256(payload).hexdigest()}


def validate_model_contract(model: Any) -> None:
    config = getattr(model, "config", None)
    if config is None:
        raise TypeError("model must expose config")
    if int(getattr(config, "num_labels", -1)) != NUM_ADE20K_CLASSES:
        raise ValueError("model must expose 150 labels")
    validate_id2label(getattr(config, "id2label", {}))


def load_segformer(config: SegFormerMaskConfig, *, cache_dir: str | None = None, local_files_only: bool = True):
    from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
    common = {"revision": config.revision, "cache_dir": cache_dir, "local_files_only": local_files_only}
    processor = AutoImageProcessor.from_pretrained(config.model_id, **common)
    model = AutoModelForSemanticSegmentation.from_pretrained(config.model_id, use_safetensors=True, **common)
    validate_model_contract(model)
    return processor, model.eval()


def resolved_weights_sha256(config: SegFormerMaskConfig, *, cache_dir: str | None = None, local_files_only: bool = True) -> str:
    """Hash the exact cached safetensors artifact selected by the pinned revision."""
    from transformers.utils.hub import cached_file
    path = cached_file(config.model_id, "model.safetensors", revision=config.revision, cache_dir=cache_dir, local_files_only=local_files_only)
    if path is None:
        raise FileNotFoundError("pinned SegFormer safetensors weights are unavailable")
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def predict_id_maps(frames: Any, processor: Any, model: Any, config: SegFormerMaskConfig, *, device: str):
    import numpy as np
    import torch
    images = validate_rgb_frames(frames, expected_frames=config.num_frames)
    validate_model_contract(model)
    model.to(device).eval()
    results = []
    with torch.inference_mode():
        for offset in range(0, len(images), config.batch_size):
            inputs = processor(images=list(images[offset:offset + config.batch_size]), return_tensors="pt")
            logits = model(pixel_values=inputs["pixel_values"].to(device)).logits
            results.append(logits_to_id_maps(logits, config.output_size).cpu().numpy())
    result = np.concatenate(results).astype(np.uint8, copy=False)
    validate_numpy_id_map(result, expected_ndim=3)
    if result.shape != (config.num_frames, *config.output_size):
        raise RuntimeError(f"unexpected mask shape {result.shape}")
    return result
