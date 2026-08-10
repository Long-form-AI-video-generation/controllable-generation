"""Deterministic ADE20K-to-coarse semantic grouping."""

from __future__ import annotations

from enum import IntEnum
from typing import Mapping

import numpy as np


class SemanticGroup(IntEnum):
    UNKNOWN = 0
    PERSON = 1
    VEHICLE = 2
    GROUND = 3
    STRUCTURE = 4
    NATURE = 5
    SKY = 6
    WATER = 7
    OBJECT = 8
    OVERHEAD = 9


_KEYWORDS = {
    SemanticGroup.PERSON: {
        "person",
    },
    SemanticGroup.VEHICLE: {
        "airplane", "bicycle", "boat", "bus", "car", "minibike",
        "ship", "tank", "truck", "van",
    },
    SemanticGroup.GROUND: {
        "carpet", "conveyer belt", "dirt track", "earth", "field",
        "floor", "grass", "land", "path", "pavement", "platform",
        "road", "rug", "sidewalk", "stage", "stairs", "stairway",
        "step", "track",
    },
    SemanticGroup.STRUCTURE: {
        "blind", "bridge", "building", "column",
        "door", "escalator", "fence", "fireplace", "grandstand",
        "house", "hovel", "pier", "railing", "shelf", "skyscraper",
        "stairs", "stairway", "tent", "wall", "windowpane",
    },
    SemanticGroup.OVERHEAD: {
        "awning", "canopy", "ceiling", "roof",
    },
    SemanticGroup.NATURE: {
        "flower", "hill", "mountain", "palm", "plant", "rock", "sand",
        "sea", "snow", "tree",
    },
    SemanticGroup.SKY: {
        "sky",
    },
    SemanticGroup.WATER: {
        "fountain", "lake", "pool", "river", "sea", "swimming pool",
        "water", "waterfall",
    },
}


def normalize_label_name(name: str) -> str:
    """Normalize Transformers label text for stable matching."""
    return name.strip().lower().replace("_", " ")


def group_for_label(name: str) -> SemanticGroup:
    """Map one ADE20K label name to a broad control group."""
    normalized = normalize_label_name(name)
    for group in (
        SemanticGroup.PERSON,
        SemanticGroup.VEHICLE,
        SemanticGroup.WATER,
        SemanticGroup.SKY,
        SemanticGroup.GROUND,
        SemanticGroup.OVERHEAD,
        SemanticGroup.STRUCTURE,
        SemanticGroup.NATURE,
    ):
        if normalized in _KEYWORDS[group]:
            return group
    return SemanticGroup.OBJECT


def build_group_lookup(id2label: Mapping[int | str, str]) -> np.ndarray:
    """Build an indexed lookup table and reject missing class IDs."""
    normalized = {int(key): value for key, value in id2label.items()}
    if not normalized:
        raise ValueError("id2label must not be empty")
    expected = set(range(max(normalized) + 1))
    missing = sorted(expected.difference(normalized))
    if missing:
        raise ValueError(f"id2label has missing IDs: {missing}")
    return np.asarray(
        [int(group_for_label(normalized[index])) for index in sorted(expected)],
        dtype=np.uint8,
    )


def apply_group_lookup(labels: np.ndarray, lookup: np.ndarray) -> np.ndarray:
    """Convert an integer class map without interpolation."""
    values = np.asarray(labels)
    if not np.issubdtype(values.dtype, np.integer):
        raise TypeError("labels must contain integer class IDs")
    if values.size and (values.min() < 0 or values.max() >= len(lookup)):
        raise ValueError("labels contain a class ID outside the lookup table")
    return lookup[values]
