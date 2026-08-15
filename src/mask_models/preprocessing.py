"""Deterministic raw SegFormer-B5 semantic-mask preprocessing."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .labels import (
    ADE20K_LABEL_ORDER_SHA256,
    ADE20K_VISUALIZATION_PALETTE_SHA256,
    NUM_ADE20K_CLASSES,
    validate_id2label,
    validate_numpy_id_map,
)

SEGFORMER_MODEL_ID = "nvidia/segformer-b5-finetuned-ade-640-640"
SEGFORMER_CONFIG_REVISION = "739f5d4692954e4a185eac280dec1ba5a7d52f1d"
SEGFORMER_WEIGHTS_REVISION = "f4c8e8f5b581b6bc8ed5208e4bd139d95f65610f"
SEGFORMER_WEIGHTS_BLOB_SHA256 = (
    "3f451be733bc5f69227886ef4a472a373f3440ad11b59b3b82cf50f500855b62"
)
# Backward-compatible name for callers that only need the configuration ID.
SEGFORMER_REVISION = SEGFORMER_CONFIG_REVISION
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
    config_revision: str = SEGFORMER_CONFIG_REVISION
    weights_revision: str = SEGFORMER_WEIGHTS_REVISION
    preprocessing_version: str = PREPROCESSING_VERSION

    def __post_init__(self) -> None:
        if self.num_frames <= 0 or self.batch_size <= 0:
            raise ValueError("num_frames and batch_size must be positive")
        object.__setattr__(self, "output_size", _positive_pair(self.output_size, "output_size"))
        if self.model_id != SEGFORMER_MODEL_ID:
            raise ValueError(f"unsupported mask extractor {self.model_id!r}")
        if self.config_revision != SEGFORMER_CONFIG_REVISION:
            raise ValueError(f"unsupported SegFormer config revision {self.config_revision!r}")
        if self.weights_revision != SEGFORMER_WEIGHTS_REVISION:
            raise ValueError(f"unsupported SegFormer weights revision {self.weights_revision!r}")
        if self.preprocessing_version != PREPROCESSING_VERSION:
            raise ValueError("unsupported preprocessing version")

    def to_metadata(self) -> dict[str, Any]:
        return {
            "control_key": MASK_CONTROL_KEY, "model_id": self.model_id,
            "config_revision": self.config_revision,
            "weights_revision": self.weights_revision,
            "weights_blob_sha256": SEGFORMER_WEIGHTS_BLOB_SHA256,
            "weights_format": "safetensors",
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

    config_path = _resolved_artifact_path(
        config,
        "config.json",
        revision=config.config_revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    processor_path = _resolved_artifact_path(
        config,
        "preprocessor_config.json",
        revision=config.config_revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    weights_path = _resolved_weights_path(
        config, cache_dir=cache_dir, local_files_only=local_files_only
    )

    with tempfile.TemporaryDirectory(prefix="segformer-b5-safe-") as temporary:
        assembled = Path(temporary)
        _link_or_copy(config_path, assembled / "config.json")
        _link_or_copy(
            processor_path,
            assembled / "preprocessor_config.json",
        )
        _link_or_copy(weights_path, assembled / "model.safetensors")
        processor = AutoImageProcessor.from_pretrained(
            assembled, local_files_only=True
        )
        model, loading_info = AutoModelForSemanticSegmentation.from_pretrained(
            assembled,
            local_files_only=True,
            use_safetensors=True,
            output_loading_info=True,
        )

    incompatible = {
        name: loading_info.get(name, [])
        for name in ("missing_keys", "unexpected_keys", "mismatched_keys")
        if loading_info.get(name)
    }
    if incompatible:
        raise ValueError(
            f"SegFormer safetensors state is incompatible: {incompatible}"
        )
    validate_model_contract(model)
    return processor, model.eval()


def _resolved_artifact_path(
    config: SegFormerMaskConfig,
    filename: str,
    *,
    revision: str,
    cache_dir: str | None = None,
    local_files_only: bool = True,
) -> str:
    from transformers.utils.hub import cached_file
    path = cached_file(config.model_id, filename, revision=revision, cache_dir=cache_dir, local_files_only=local_files_only)
    if path is None:
        raise FileNotFoundError(
            f"pinned SegFormer artifact is unavailable: {filename}"
        )
    return path


def _resolved_weights_path(config: SegFormerMaskConfig, *, cache_dir: str | None = None, local_files_only: bool = True) -> str:
    return _resolved_artifact_path(
        config,
        "model.safetensors",
        revision=config.weights_revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )


def _link_or_copy(source: str, destination: Path) -> None:
    """Assemble a local model directory without normally copying 339 MB."""
    try:
        destination.symlink_to(Path(source).resolve())
        return
    except OSError:
        pass
    try:
        os.link(source, destination)
        return
    except OSError:
        shutil.copy2(source, destination)


def resolved_weights_sha256(config: SegFormerMaskConfig, *, cache_dir: str | None = None, local_files_only: bool = True) -> str:
    """Hash the exact cached safetensors artifact selected by the pinned revision."""
    path = _resolved_weights_path(
        config,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != SEGFORMER_WEIGHTS_BLOB_SHA256:
        raise ValueError(
            f"SegFormer weights hash mismatch: expected "
            f"{SEGFORMER_WEIGHTS_BLOB_SHA256}, got {actual}"
        )
    return actual


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
