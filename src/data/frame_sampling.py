"""Shared deterministic frame-index selection."""

from __future__ import annotations

import numpy as np


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
