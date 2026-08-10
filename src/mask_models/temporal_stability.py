"""Metrics for comparing adjacent raw and grouped semantic maps."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .semantic_groups import SemanticGroup


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


def summarize_frame_composition(
    raw_map: np.ndarray,
    grouped_map: np.ndarray,
    id2label: Mapping[int | str, str],
    *,
    top_k: int = 8,
    collapse_threshold: float = 0.9,
) -> dict[str, Any]:
    """Explain which raw classes produced each coarse frame region."""
    raw = np.asarray(raw_map)
    grouped = np.asarray(grouped_map)
    if raw.shape != grouped.shape or raw.ndim != 2:
        raise ValueError(
            f"Expected matching [H,W] maps, got {raw.shape} and {grouped.shape}"
        )
    if raw.size == 0:
        raise ValueError("label maps must not be empty")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if not 0.0 < collapse_threshold <= 1.0:
        raise ValueError("collapse_threshold must be in (0, 1]")

    labels = {int(key): str(value) for key, value in id2label.items()}
    raw_ids, raw_counts = np.unique(raw, return_counts=True)
    order = np.argsort(raw_counts)[::-1]
    top_raw = []
    for position in order[:top_k]:
        class_id = int(raw_ids[position])
        if class_id not in labels:
            raise KeyError(f"Missing class name for ID {class_id}")
        group_values = grouped[raw == class_id]
        group_id = int(np.bincount(group_values.reshape(-1)).argmax())
        top_raw.append({
            "class_id": class_id,
            "class_name": labels[class_id],
            "fraction": float(raw_counts[position] / raw.size),
            "coarse_group_id": group_id,
            "coarse_group_name": SemanticGroup(group_id).name.lower(),
        })

    group_ids, group_counts = np.unique(grouped, return_counts=True)
    groups = [
        {
            "group_id": int(group_id),
            "group_name": SemanticGroup(int(group_id)).name.lower(),
            "fraction": float(count / grouped.size),
        }
        for group_id, count in sorted(
            zip(group_ids.tolist(), group_counts.tolist()),
            key=lambda item: item[1],
            reverse=True,
        )
    ]
    dominant = groups[0]
    object_fraction = next(
        (
            item["fraction"]
            for item in groups
            if item["group_id"] == int(SemanticGroup.OBJECT)
        ),
        0.0,
    )
    return {
        "top_raw_classes": top_raw,
        "coarse_groups": groups,
        "dominant_group": dominant,
        "object_fraction": float(object_fraction),
        "collapse_flag": bool(dominant["fraction"] >= collapse_threshold),
    }


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
