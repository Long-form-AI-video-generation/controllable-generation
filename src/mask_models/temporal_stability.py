"""Metrics for comparing adjacent raw and grouped semantic maps."""

from __future__ import annotations

from typing import Any

import numpy as np


def adjacent_agreement(label_maps: np.ndarray) -> np.ndarray:
    """Return the equal-label fraction for every adjacent frame pair."""
    maps = np.asarray(label_maps)
    if maps.ndim != 3:
        raise ValueError(f"Expected [T,H,W], got {maps.shape}")
    if maps.shape[0] < 2:
        return np.empty(0, dtype=np.float64)
    return (maps[1:] == maps[:-1]).mean(axis=(1, 2))


def dominant_fraction(label_map: np.ndarray) -> float:
    values = np.asarray(label_map)
    if values.ndim != 2 or values.size == 0:
        raise ValueError("Expected a nonempty [H,W] label map")
    return float(np.bincount(values.reshape(-1)).max() / values.size)


def summarize_stability(
    raw_maps: np.ndarray,
    grouped_maps: np.ndarray,
) -> dict[str, Any]:
    """Summarize stability while keeping raw and grouped results separate."""
    raw = np.asarray(raw_maps)
    grouped = np.asarray(grouped_maps)
    if raw.shape != grouped.shape:
        raise ValueError(f"Shape mismatch: {raw.shape} != {grouped.shape}")

    raw_agreement = adjacent_agreement(raw)
    grouped_agreement = adjacent_agreement(grouped)
    raw_mean = float(raw_agreement.mean()) if raw_agreement.size else 1.0
    grouped_mean = (
        float(grouped_agreement.mean()) if grouped_agreement.size else 1.0
    )
    return {
        "frame_count": int(raw.shape[0]),
        "raw": {
            "adjacent_agreement": raw_agreement.tolist(),
            "mean_adjacent_agreement": raw_mean,
            "unique_classes_per_frame": [
                int(np.unique(frame).size) for frame in raw
            ],
            "dominant_fraction_per_frame": [
                dominant_fraction(frame) for frame in raw
            ],
        },
        "coarse": {
            "adjacent_agreement": grouped_agreement.tolist(),
            "mean_adjacent_agreement": grouped_mean,
            "unique_groups_per_frame": [
                int(np.unique(frame).size) for frame in grouped
            ],
            "dominant_fraction_per_frame": [
                dominant_fraction(frame) for frame in grouped
            ],
        },
        "coarse_agreement_improvement": grouped_mean - raw_mean,
    }

