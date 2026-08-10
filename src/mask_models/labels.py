"""Pinned ADE20K label contract and visualization colors.

The semantic-control tensor stores raw SegFormer class IDs.  The RGB palette is
only for diagnostics: it is not lossless and must never be used as the neural
condition representation.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


NUM_ADE20K_CLASSES = 150
LABEL_SOURCE = (
    "https://huggingface.co/nvidia/segformer-b5-finetuned-ade-640-640/"
    "resolve/739f5d4692954e4a185eac280dec1ba5a7d52f1d/config.json"
)
PALETTE_SOURCE = (
    "https://github.com/open-mmlab/mmsegmentation/blob/main/"
    "mmseg/utils/class_names.py"
)

# Exact order and spelling from the pinned NVIDIA SegFormer-B5 configuration.
# The trailing space in ``bed `` is present in that configuration and is kept
# deliberately so provenance checks detect a changed label contract.
ADE20K_LABELS: tuple[str, ...] = (
    "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed ",
    "windowpane", "grass", "cabinet", "sidewalk", "person", "earth", "door",
    "table", "mountain", "plant", "curtain", "chair", "car", "water",
    "painting", "sofa", "shelf", "house", "sea", "mirror", "rug", "field",
    "armchair", "seat", "fence", "desk", "rock", "wardrobe", "lamp",
    "bathtub", "railing", "cushion", "base", "box", "column", "signboard",
    "chest of drawers", "counter", "sand", "sink", "skyscraper", "fireplace",
    "refrigerator", "grandstand", "path", "stairs", "runway", "case",
    "pool table", "pillow", "screen door", "stairway", "river", "bridge",
    "bookcase", "blind", "coffee table", "toilet", "flower", "book", "hill",
    "bench", "countertop", "stove", "palm", "kitchen island", "computer",
    "swivel chair", "boat", "bar", "arcade machine", "hovel", "bus", "towel",
    "light", "truck", "tower", "chandelier", "awning", "streetlight", "booth",
    "television receiver", "airplane", "dirt track", "apparel", "pole", "land",
    "bannister", "escalator", "ottoman", "bottle", "buffet", "poster", "stage",
    "van", "ship", "fountain", "conveyer belt", "canopy", "washer", "plaything",
    "swimming pool", "stool", "barrel", "basket", "waterfall", "tent", "bag",
    "minibike", "cradle", "oven", "ball", "food", "step", "tank", "trade name",
    "microwave", "pot", "animal", "bicycle", "lake", "dishwasher", "screen",
    "blanket", "sculpture", "hood", "sconce", "vase", "traffic light", "tray",
    "ashcan", "fan", "pier", "crt screen", "plate", "monitor", "bulletin board",
    "shower", "radiator", "glass", "clock", "flag",
)

# Standard ADE20K visualization palette used by MMSegmentation/SceneParse150.
# It contains a known duplicate color at IDs 6 and 48, so it is visualization
# metadata only.  Raw class IDs enter the model through a learned embedding.
ADE20K_VISUALIZATION_PALETTE: tuple[tuple[int, int, int], ...] = (
    (120, 120, 120), (180, 120, 120), (6, 230, 230), (80, 50, 50),
    (4, 200, 3), (120, 120, 80), (140, 140, 140), (204, 5, 255),
    (230, 230, 230), (4, 250, 7), (224, 5, 255), (235, 255, 7),
    (150, 5, 61), (120, 120, 70), (8, 255, 51), (255, 6, 82),
    (143, 255, 140), (204, 255, 4), (255, 51, 7), (204, 70, 3),
    (0, 102, 200), (61, 230, 250), (255, 6, 51), (11, 102, 255),
    (255, 7, 71), (255, 9, 224), (9, 7, 230), (220, 220, 220),
    (255, 9, 92), (112, 9, 255), (8, 255, 214), (7, 255, 224),
    (255, 184, 6), (10, 255, 71), (255, 41, 10), (7, 255, 255),
    (224, 255, 8), (102, 8, 255), (255, 61, 6), (255, 194, 7),
    (255, 122, 8), (0, 255, 20), (255, 8, 41), (255, 5, 153),
    (6, 51, 255), (235, 12, 255), (160, 150, 20), (0, 163, 255),
    (140, 140, 140), (250, 10, 15), (20, 255, 0), (31, 255, 0),
    (255, 31, 0), (255, 224, 0), (153, 255, 0), (0, 0, 255),
    (255, 71, 0), (0, 235, 255), (0, 173, 255), (31, 0, 255),
    (11, 200, 200), (255, 82, 0), (0, 255, 245), (0, 61, 255),
    (0, 255, 112), (0, 255, 133), (255, 0, 0), (255, 163, 0),
    (255, 102, 0), (194, 255, 0), (0, 143, 255), (51, 255, 0),
    (0, 82, 255), (0, 255, 41), (0, 255, 173), (10, 0, 255),
    (173, 255, 0), (0, 255, 153), (255, 92, 0), (255, 0, 255),
    (255, 0, 245), (255, 0, 102), (255, 173, 0), (255, 0, 20),
    (255, 184, 184), (0, 31, 255), (0, 255, 61), (0, 71, 255),
    (255, 0, 204), (0, 255, 194), (0, 255, 82), (0, 10, 255),
    (0, 112, 255), (51, 0, 255), (0, 194, 255), (0, 122, 255),
    (0, 255, 163), (255, 153, 0), (0, 255, 10), (255, 112, 0),
    (143, 255, 0), (82, 0, 255), (163, 255, 0), (255, 235, 0),
    (8, 184, 170), (133, 0, 255), (0, 255, 92), (184, 0, 255),
    (255, 0, 31), (0, 184, 255), (0, 214, 255), (255, 0, 112),
    (92, 255, 0), (0, 224, 255), (112, 224, 255), (70, 184, 160),
    (163, 0, 255), (153, 0, 255), (71, 255, 0), (255, 0, 163),
    (255, 204, 0), (255, 0, 143), (0, 255, 235), (133, 255, 0),
    (255, 0, 235), (245, 0, 255), (255, 0, 122), (255, 245, 0),
    (10, 190, 212), (214, 255, 0), (0, 204, 255), (20, 0, 255),
    (255, 255, 0), (0, 153, 255), (0, 41, 255), (0, 255, 204),
    (41, 0, 255), (41, 255, 0), (173, 0, 255), (0, 245, 255),
    (71, 0, 255), (122, 0, 255), (0, 255, 184), (0, 92, 255),
    (184, 255, 0), (0, 133, 255), (255, 214, 0), (25, 194, 194),
    (102, 255, 0), (92, 0, 255),
)


def _validate_constants() -> None:
    if len(ADE20K_LABELS) != NUM_ADE20K_CLASSES:
        raise RuntimeError("ADE20K label table must contain exactly 150 rows")
    if len(ADE20K_VISUALIZATION_PALETTE) != NUM_ADE20K_CLASSES:
        raise RuntimeError("ADE20K palette must contain exactly 150 rows")
    if any(
        len(color) != 3 or any(channel < 0 or channel > 255 for channel in color)
        for color in ADE20K_VISUALIZATION_PALETTE
    ):
        raise RuntimeError("ADE20K palette rows must be uint8 RGB triples")


_validate_constants()


def label_order_sha256(labels: Sequence[str] = ADE20K_LABELS) -> str:
    """Hash the ordered label contract using canonical UTF-8 JSON."""
    payload = json.dumps(
        list(labels),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def visualization_palette_sha256(
    palette: Sequence[Sequence[int]] = ADE20K_VISUALIZATION_PALETTE,
) -> str:
    """Hash palette bytes in row-major RGB order."""
    payload = bytes(channel for color in palette for channel in color)
    return hashlib.sha256(payload).hexdigest()


ADE20K_LABEL_ORDER_SHA256 = label_order_sha256()
ADE20K_VISUALIZATION_PALETTE_SHA256 = visualization_palette_sha256()


def validate_id2label(id2label: Mapping[int | str, str]) -> tuple[str, ...]:
    """Require the exact contiguous label order from the pinned B5 config."""
    normalized: dict[int, str] = {}
    for raw_key, raw_label in id2label.items():
        try:
            key = int(raw_key)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid ADE20K class ID {raw_key!r}") from error
        if key in normalized:
            raise ValueError(f"duplicate ADE20K class ID after normalization: {key}")
        normalized[key] = str(raw_label)

    expected_ids = set(range(NUM_ADE20K_CLASSES))
    actual_ids = set(normalized)
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        raise ValueError(
            f"ADE20K id2label must contain exactly IDs 0..149; "
            f"missing={missing}, extra={extra}"
        )

    actual = tuple(normalized[index] for index in range(NUM_ADE20K_CLASSES))
    mismatches = [
        (index, expected, observed)
        for index, (expected, observed) in enumerate(zip(ADE20K_LABELS, actual))
        if expected != observed
    ]
    if mismatches:
        index, expected, observed = mismatches[0]
        raise ValueError(
            f"ADE20K label mismatch at ID {index}: "
            f"expected {expected!r}, got {observed!r}"
        )
    return actual


def palette_collisions() -> dict[tuple[int, int, int], tuple[int, ...]]:
    """Return duplicate visualization colors and their class IDs."""
    by_color: defaultdict[tuple[int, int, int], list[int]] = defaultdict(list)
    for class_id, color in enumerate(ADE20K_VISUALIZATION_PALETTE):
        by_color[color].append(class_id)
    return {
        color: tuple(class_ids)
        for color, class_ids in by_color.items()
        if len(class_ids) > 1
    }


def validate_numpy_id_map(
    values: Any,
    *,
    expected_ndim: int | None = None,
) -> Any:
    """Validate a nonempty NumPy integer array containing IDs 0..149."""
    import numpy as np

    array = np.asarray(values)
    if expected_ndim is not None and array.ndim != expected_ndim:
        raise ValueError(
            f"expected a {expected_ndim}D ADE20K ID map, got {array.shape}"
        )
    if array.size == 0:
        raise ValueError("ADE20K ID map must not be empty")
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"ADE20K ID map must be integer, got {array.dtype}")
    minimum = int(array.min())
    maximum = int(array.max())
    if minimum < 0 or maximum >= NUM_ADE20K_CLASSES:
        raise ValueError(
            f"ADE20K IDs must be in [0,149], got [{minimum},{maximum}]"
        )
    return array


def colorize_id_map(values: Any) -> Any:
    """Colorize IDs for diagnostics; the result must not be fed to the model."""
    import numpy as np

    array = validate_numpy_id_map(values)
    palette = np.asarray(ADE20K_VISUALIZATION_PALETTE, dtype=np.uint8)
    return palette[array]
