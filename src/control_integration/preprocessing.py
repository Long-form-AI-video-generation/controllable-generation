"""Matched reference-video preprocessing for depth, Canny, and masks."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.data.frame_sampling import resolve_frame_interval, select_frame_indices
from src.mask_models.preprocessing import (
    SegFormerMaskConfig,
    predict_id_maps,
    resolved_weights_sha256,
)
from src.sketch_models.preprocessing import CannyConfig, prepare_canny_sequence

from .contracts import canonicalize_expert_names
from .control_artifacts import sha256_file


@dataclass(frozen=True)
class ReferenceFrameSequence:
    """The one source-frame sequence consumed by every requested modality."""

    frames_bgr: np.ndarray
    source_indices: tuple[int, ...]
    padded_positions: tuple[int, ...]
    source_video_sha256: str
    actual_frame_count: int
    interval: tuple[int, int]

    def __post_init__(self) -> None:
        if self.frames_bgr.dtype != np.uint8 or self.frames_bgr.ndim != 4:
            raise ValueError("frames_bgr must be uint8 [T,H,W,3]")
        if self.frames_bgr.shape[-1] != 3:
            raise ValueError("frames_bgr must have three BGR channels")
        if len(self.frames_bgr) != len(self.source_indices):
            raise ValueError("frame count and source_indices must match")
        if any(index < 0 or index >= self.actual_frame_count for index in self.source_indices):
            raise ValueError("source frame index is outside the decoded video")


@dataclass(frozen=True)
class MidasConfig:
    """Pinned local MiDaS identity with no download fallback."""

    repo_dir: Path
    weights_path: Path
    output_size: tuple[int, int] = (128, 128)
    midas_size: tuple[int, int] = (360, 640)
    model_type: str = "DPT_Large"
    input_color_order: str = "BGR"

    def __post_init__(self) -> None:
        object.__setattr__(self, "repo_dir", Path(self.repo_dir).resolve())
        object.__setattr__(self, "weights_path", Path(self.weights_path).resolve())
        if self.input_color_order != "BGR":
            raise ValueError("integration MiDaS input must be BGR")
        if any(value <= 0 for value in (*self.output_size, *self.midas_size)):
            raise ValueError("MiDaS output sizes must be positive")


def decode_reference_frames(
    video_path: str | Path,
    *,
    frame_num: int,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> ReferenceFrameSequence:
    """Decode sequentially once, then select shared deterministic indices."""

    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if frame_num <= 0:
        raise ValueError("frame_num must be positive")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open reference video {path}")
    decoded: list[np.ndarray] = []
    try:
        while True:
            okay, frame = capture.read()
            if not okay:
                break
            if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3:
                raise RuntimeError("reference decoder did not return BGR uint8 frames")
            decoded.append(np.ascontiguousarray(frame))
    finally:
        capture.release()
    actual_count = len(decoded)
    if actual_count <= 0:
        raise RuntimeError(f"could not decode frames from {path}")
    requested_end = actual_count if end_frame is None else int(end_frame)
    start, end = resolve_frame_interval(start_frame, requested_end, actual_count)
    indices = select_frame_indices(start, end, frame_num)
    selected = np.stack([decoded[int(index)] for index in indices])
    padded_positions = ()
    if end - start < frame_num:
        padded_positions = tuple(range(end - start, frame_num))
    return ReferenceFrameSequence(
        frames_bgr=np.ascontiguousarray(selected),
        source_indices=tuple(int(index) for index in indices),
        padded_positions=padded_positions,
        source_video_sha256=sha256_file(path),
        actual_frame_count=actual_count,
        interval=(start, end),
    )


def midas_source_tree_sha256(repo_dir: str | Path) -> str:
    """Hash local MiDaS source code deterministically, excluding cache files."""

    root = Path(repo_dir)
    if not root.is_dir():
        raise FileNotFoundError(root)
    digest = hashlib.sha256()
    files = sorted(
        path
        for path in root.rglob("*.py")
        if ".git" not in path.parts and "__pycache__" not in path.parts
    )
    if not files:
        raise ValueError(f"MiDaS repository has no Python source files: {root}")
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def load_midas_local(config: MidasConfig, *, device: str) -> tuple[torch.nn.Module, Any]:
    """Load the exact local MiDaS code and weights; never download or fall back."""

    if not config.repo_dir.is_dir():
        raise FileNotFoundError(config.repo_dir)
    if not config.weights_path.is_file():
        raise FileNotFoundError(config.weights_path)
    midas = torch.hub.load(
        str(config.repo_dir),
        config.model_type,
        source="local",
        pretrained=False,
    )
    checkpoint = torch.load(
        config.weights_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(checkpoint, Mapping):
        raise ValueError("local MiDaS weights must contain a state-dict mapping")
    midas.load_state_dict(checkpoint, strict=True)
    transforms = torch.hub.load(str(config.repo_dir), "transforms", source="local")
    transform = transforms.dpt_transform
    return midas.to(device).eval(), transform


def extract_depth_frame(
    frame_bgr: np.ndarray,
    *,
    midas: torch.nn.Module,
    transform: Any,
    device: str,
    config: MidasConfig,
) -> np.ndarray:
    """Port of the corrected depth numerical path used by depth inference."""

    if frame_bgr.dtype != np.uint8 or frame_bgr.ndim != 3 or frame_bgr.shape[-1] != 3:
        raise ValueError("depth input must be one BGR uint8 frame")
    input_tensor = transform(frame_bgr).to(device)
    with torch.inference_mode():
        prediction = midas(input_tensor)
        prediction = F.interpolate(
            prediction.unsqueeze(1),
            size=config.midas_size,
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth = prediction.float().cpu().numpy()
    normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    quantized = (normalized * 255.0).astype(np.uint8)
    output_h, output_w = config.output_size
    resized = cv2.resize(
        quantized,
        (output_w, output_h),
        interpolation=cv2.INTER_LINEAR,
    )
    return resized.astype(np.float32) / 255.0


def prepare_depth_sequence(
    frames_bgr: np.ndarray,
    *,
    midas: torch.nn.Module,
    transform: Any,
    device: str,
    config: MidasConfig,
) -> np.ndarray:
    """Produce raw depth control as float32 ``[1,1,T,H,W]``."""

    if frames_bgr.ndim != 4 or frames_bgr.shape[-1] != 3:
        raise ValueError("depth frames must be [T,H,W,3]")
    depths = [
        extract_depth_frame(
            frame,
            midas=midas,
            transform=transform,
            device=device,
            config=config,
        )
        for frame in frames_bgr
    ]
    return np.ascontiguousarray(np.stack(depths)[None, None].astype(np.float32))


def build_matched_controls(
    sequence: ReferenceFrameSequence,
    *,
    experts: Sequence[str],
    canny_config: CannyConfig | None = None,
    mask_config: SegFormerMaskConfig | None = None,
    mask_processor: Any | None = None,
    mask_model: Any | None = None,
    mask_device: str | None = None,
    midas_config: MidasConfig | None = None,
    midas: torch.nn.Module | None = None,
    midas_transform: Any | None = None,
    midas_device: str | None = None,
) -> dict[str, np.ndarray]:
    """Build only requested controls from one already-decoded frame sequence."""

    selected = canonicalize_expert_names(list(experts))
    frame_num = len(sequence.frames_bgr)
    result: dict[str, np.ndarray] = {}
    if "depth" in selected:
        if any(
            value is None
            for value in (midas_config, midas, midas_transform, midas_device)
        ):
            raise ValueError("depth requires local MiDaS config, model, transform, and device")
        result["depth"] = prepare_depth_sequence(
            sequence.frames_bgr,
            midas=midas,
            transform=midas_transform,
            device=midas_device,
            config=midas_config,
        )
    if "canny" in selected:
        if canny_config is None:
            raise ValueError("canny requires CannyConfig")
        if canny_config.num_frames != frame_num:
            raise ValueError("Canny frame count must match the shared sequence")
        canny = prepare_canny_sequence(
            sequence.frames_bgr,
            canny_config,
            frame_indices=range(frame_num),
        )
        result["canny"] = np.ascontiguousarray(canny.astype(np.uint8))
    if "mask" in selected:
        if any(
            value is None
            for value in (mask_config, mask_processor, mask_model, mask_device)
        ):
            raise ValueError("mask requires SegFormer config, processor, model, and device")
        if mask_config.num_frames != frame_num:
            raise ValueError("mask frame count must match the shared sequence")
        frames_rgb = np.ascontiguousarray(sequence.frames_bgr[..., ::-1])
        masks = predict_id_maps(
            frames_rgb,
            mask_processor,
            mask_model,
            mask_config,
            device=mask_device,
        )
        result["mask"] = np.ascontiguousarray(masks[None, None].astype(np.uint8))
    return result


def preprocessing_identities(
    *,
    experts: Sequence[str],
    canny_config: CannyConfig | None = None,
    mask_config: SegFormerMaskConfig | None = None,
    midas_config: MidasConfig | None = None,
    mask_cache_dir: str | None = None,
) -> dict[str, Mapping[str, Any]]:
    """Record exact preprocessing identities for a prepared-control bundle."""

    selected = canonicalize_expert_names(list(experts))
    result: dict[str, Mapping[str, Any]] = {}
    if "depth" in selected:
        if midas_config is None:
            raise ValueError("depth requires MidasConfig")
        result["depth"] = {
            "repo_dir": str(midas_config.repo_dir),
            "weights_path": str(midas_config.weights_path),
            "output_size": list(midas_config.output_size),
            "midas_size": list(midas_config.midas_size),
            "model_type": midas_config.model_type,
            "input_color_order": midas_config.input_color_order,
            "repo_tree_sha256": midas_source_tree_sha256(midas_config.repo_dir),
            "weights_sha256": sha256_file(midas_config.weights_path),
            "operation_order": [
                "BGR_input",
                "midas_dpt_transform",
                "midas_prediction",
                "bicubic_to_midas_size",
                "per_frame_min_max",
                "uint8_quantize",
                "linear_resize",
            ],
        }
    if "canny" in selected:
        if canny_config is None:
            raise ValueError("canny requires CannyConfig")
        result["canny"] = canny_config.to_metadata()
    if "mask" in selected:
        if mask_config is None:
            raise ValueError("mask requires SegFormerMaskConfig")
        result["mask"] = {
            **mask_config.to_metadata(),
            "resolved_weights_sha256": resolved_weights_sha256(
                mask_config,
                cache_dir=mask_cache_dir,
                local_files_only=True,
            ),
            "input_color_order": "RGB_from_shared_BGR",
        }
    return result
