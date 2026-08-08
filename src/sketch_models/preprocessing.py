"""Deterministic Canny preprocessing shared by preparation and inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import cv2
import numpy as np

try:
    from data.frame_sampling import select_frame_indices
except ImportError:
    from src.data.frame_sampling import select_frame_indices


PREPROCESSING_VERSION = "canny-v1"


@dataclass(frozen=True)
class CannyConfig:
    """Complete preprocessing contract for the first sketch-control model."""

    low_threshold: int = 100
    high_threshold: int = 200
    num_frames: int = 8
    output_size: tuple[int, int] = (128, 128)
    color_order: str = "BGR"

    def __post_init__(self) -> None:
        if not 0 <= self.low_threshold < self.high_threshold <= 255:
            raise ValueError(
                "Canny thresholds must satisfy "
                "0 <= low_threshold < high_threshold <= 255"
            )
        if self.num_frames <= 0:
            raise ValueError("num_frames must be positive")
        if len(self.output_size) != 2 or any(v <= 0 for v in self.output_size):
            raise ValueError("output_size must contain two positive integers")
        if self.color_order.upper() not in {"RGB", "BGR"}:
            raise ValueError("color_order must be 'RGB' or 'BGR'")

    def to_metadata(self) -> dict[str, object]:
        metadata = asdict(self)
        metadata["output_size"] = list(self.output_size)
        metadata["color_order"] = self.color_order.upper()
        metadata["preprocessing_version"] = PREPROCESSING_VERSION
        metadata["operation_order"] = [
            "select_frames",
            "declare_color_order",
            "grayscale",
            "canny_at_source_resolution",
            "nearest_neighbor_resize",
            "binary_normalize",
        ]
        return metadata


def _as_uint8(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame)
    if not np.isfinite(frame).all():
        raise ValueError("frame contains NaN or Inf")

    if frame.dtype == np.uint8:
        return frame

    frame = frame.astype(np.float32, copy=False)
    if frame.size and frame.min() >= 0.0 and frame.max() <= 1.0:
        frame = frame * 255.0
    return np.clip(np.rint(frame), 0, 255).astype(np.uint8)


def to_grayscale(frame: np.ndarray, color_order: str) -> np.ndarray:
    """Convert an explicitly RGB- or BGR-ordered frame to uint8 grayscale."""

    frame = _as_uint8(frame)
    order = color_order.upper()
    if order not in {"RGB", "BGR"}:
        raise ValueError("color_order must be 'RGB' or 'BGR'")

    if frame.ndim == 2:
        return frame
    if frame.ndim == 3 and frame.shape[-1] == 1:
        return frame[..., 0]
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(
            "frame must have shape [H,W], [H,W,1], or [H,W,3]"
        )

    conversion = cv2.COLOR_RGB2GRAY if order == "RGB" else cv2.COLOR_BGR2GRAY
    return cv2.cvtColor(frame, conversion)


def extract_canny(frame: np.ndarray, config: CannyConfig) -> np.ndarray:
    """Extract a binary Canny map using the frozen operation order."""

    grayscale = to_grayscale(frame, config.color_order)
    edges = cv2.Canny(
        grayscale,
        config.low_threshold,
        config.high_threshold,
    )

    output_h, output_w = config.output_size
    if edges.shape != (output_h, output_w):
        edges = cv2.resize(
            edges,
            (output_w, output_h),
            interpolation=cv2.INTER_NEAREST,
        )

    return (edges > 0).astype(np.float32)


def prepare_canny_sequence(
    frames: Sequence[np.ndarray] | np.ndarray,
    config: CannyConfig,
    *,
    frame_indices: Iterable[int] | None = None,
) -> np.ndarray:
    """Return a prepared control tensor with shape ``[1,1,T,H,W]``."""

    frames_array = np.asarray(frames)
    if frames_array.ndim not in {3, 4}:
        raise ValueError("frames must have shape [N,H,W] or [N,H,W,C]")
    if len(frames_array) == 0:
        raise ValueError("frames must not be empty")

    if frame_indices is None:
        indices = select_frame_indices(0, len(frames_array), config.num_frames)
    else:
        indices = np.asarray(list(frame_indices), dtype=np.int64)
        if indices.shape != (config.num_frames,):
            raise ValueError(
                f"frame_indices must contain {config.num_frames} values"
            )
        if (indices < 0).any() or (indices >= len(frames_array)).any():
            raise ValueError("frame_indices contain an out-of-range value")

    prepared = np.stack(
        [extract_canny(frames_array[int(index)], config) for index in indices],
        axis=0,
    )
    result = prepared[None, None, ...].astype(np.float32, copy=False)
    validate_canny_tensor(result, expected_frames=config.num_frames)
    return result


def validate_canny_tensor(
    tensor: np.ndarray,
    *,
    expected_frames: int | None = None,
) -> None:
    """Validate the serialized one-channel sketch-control contract."""

    tensor = np.asarray(tensor)
    if tensor.ndim != 5:
        raise ValueError(f"expected [B,1,T,H,W], got shape {tensor.shape}")
    if tensor.shape[0] != 1 or tensor.shape[1] != 1:
        raise ValueError(f"expected batch/channel dimensions [1,1], got {tensor.shape[:2]}")
    if expected_frames is not None and tensor.shape[2] != expected_frames:
        raise ValueError(
            f"expected {expected_frames} frames, got {tensor.shape[2]}"
        )
    if not np.isfinite(tensor).all():
        raise ValueError("Canny tensor contains NaN or Inf")
    unique = np.unique(tensor)
    if not np.isin(unique, (0.0, 1.0)).all():
        raise ValueError(f"Canny tensor must be binary, got values {unique[:8]}")


def to_torch_tensor(prepared: np.ndarray):
    """Convert prepared NumPy control to torch without importing torch globally."""

    validate_canny_tensor(prepared)
    import torch

    return torch.from_numpy(np.ascontiguousarray(prepared)).float()
