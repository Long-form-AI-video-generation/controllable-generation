"""Shared deterministic frame-index selection."""

from __future__ import annotations

import numpy as np


def resolve_frame_interval(
    start_frame: int,
    end_frame: int,
    actual_frame_count: int,
) -> tuple[int, int]:
    """Clamp a metadata interval to the frames present in the video."""

    if actual_frame_count <= 0:
        raise ValueError("actual_frame_count must be positive")
    if start_frame < 0:
        raise ValueError("start_frame must be non-negative")
    if end_frame <= start_frame:
        raise ValueError("end_frame must be greater than start_frame")
    if start_frame >= actual_frame_count:
        raise ValueError(
            f"start_frame {start_frame} is outside a video with "
            f"{actual_frame_count} frames"
        )

    return start_frame, min(end_frame, actual_frame_count)


def select_frame_indices(
    start_frame: int,
    end_frame: int,
    num_frames: int,
) -> np.ndarray:
    """Select deterministic indices from a half-open frame interval."""

    if start_frame < 0:
        raise ValueError("start_frame must be non-negative")
    if end_frame <= start_frame:
        raise ValueError("end_frame must be greater than start_frame")
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")

    available = end_frame - start_frame
    if available >= num_frames:
        return np.linspace(
            start_frame,
            end_frame - 1,
            num_frames,
            dtype=np.int64,
        )

    indices = np.arange(start_frame, end_frame, dtype=np.int64)
    padding = np.full(num_frames - available, indices[-1], dtype=np.int64)
    return np.concatenate([indices, padding])
