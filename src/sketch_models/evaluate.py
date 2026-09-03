"""Evaluate sketch adherence and video health for one inference comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from src.data.frame_sampling import select_frame_indices


def read_video(path: str) -> np.ndarray:
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {path}")
    frames = []
    try:
        while True:
            okay, frame = capture.read()
            if not okay:
                break
            frames.append(frame)
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    return np.stack(frames)


def canny_frames(
    frames: np.ndarray,
    target_hw: tuple[int, int],
    low: int,
    high: int,
) -> np.ndarray:
    height, width = target_hw
    edges = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)
        edges.append(cv2.Canny(gray, low, high) > 0)
    return np.stack(edges)


def edge_pair_metrics(
    reference: np.ndarray,
    generated: np.ndarray,
    tolerance: int = 3,
) -> dict:
    if reference.shape != generated.shape or reference.ndim != 2:
        raise ValueError("edge maps must have identical [H,W] shapes")
    reference = reference.astype(bool)
    generated = generated.astype(bool)
    reference_count = int(reference.sum())
    generated_count = int(generated.sum())
    if reference_count == 0 and generated_count == 0:
        return {
            "status": "both_empty",
            "precision": None,
            "recall": None,
            "f1": None,
            "symmetric_chamfer": None,
        }
    if reference_count == 0:
        return {
            "status": "empty_reference",
            "precision": 0.0,
            "recall": None,
            "f1": 0.0,
            "symmetric_chamfer": None,
        }
    if generated_count == 0:
        return {
            "status": "empty_prediction",
            "precision": None,
            "recall": 0.0,
            "f1": 0.0,
            "symmetric_chamfer": None,
        }

    kernel = np.ones((2 * tolerance + 1, 2 * tolerance + 1), np.uint8)
    reference_near = cv2.dilate(reference.astype(np.uint8), kernel) > 0
    generated_near = cv2.dilate(generated.astype(np.uint8), kernel) > 0
    precision = float((generated & reference_near).sum()) / generated_count
    recall = float((reference & generated_near).sum()) / reference_count
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)

    distance_to_reference = cv2.distanceTransform(
        (~reference).astype(np.uint8), cv2.DIST_L2, 3
    )
    distance_to_generated = cv2.distanceTransform(
        (~generated).astype(np.uint8), cv2.DIST_L2, 3
    )
    chamfer = 0.5 * (
        float(distance_to_reference[generated].mean())
        + float(distance_to_generated[reference].mean())
    )
    return {
        "status": "ok",
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "symmetric_chamfer": chamfer,
    }


def edge_sequence_metrics(
    reference: np.ndarray,
    generated: np.ndarray,
    tolerance: int = 3,
) -> dict:
    if reference.shape != generated.shape or reference.ndim != 3:
        raise ValueError("edge sequences must have identical [T,H,W] shapes")
    per_frame = [
        edge_pair_metrics(ref, pred, tolerance)
        for ref, pred in zip(reference, generated)
    ]

    def average(name: str):
        values = [item[name] for item in per_frame if item[name] is not None]
        return float(np.mean(values)) if values else None

    return {
        "aggregate": {
            "precision": average("precision"),
            "recall": average("recall"),
            "f1": average("f1"),
            "symmetric_chamfer": average("symmetric_chamfer"),
        },
        "status_counts": {
            status: sum(item["status"] == status for item in per_frame)
            for status in (
                "ok", "both_empty", "empty_reference", "empty_prediction"
            )
        },
        "per_frame": per_frame,
    }


def video_health(frames: np.ndarray) -> dict[str, float]:
    float_frames = frames.astype(np.float32)
    temporal = np.abs(float_frames[1:] - float_frames[:-1])
    return {
        "mean": float(float_frames.mean()),
        "std": float(float_frames.std()),
        "temporal_mad": float(temporal.mean()) if len(temporal) else 0.0,
    }


def evaluate(
    reference: np.ndarray,
    base: np.ndarray,
    controlled: np.ndarray,
    *,
    control_hw: tuple[int, int] = (128, 128),
    canny_low: int = 100,
    canny_high: int = 200,
    tolerance: int = 3,
) -> dict:
    count = min(len(reference), len(base), len(controlled))
    indices = select_frame_indices(0, len(reference), count)
    reference = reference[indices]
    base = base[:count]
    controlled = controlled[:count]

    reference_edges = canny_frames(
        reference, control_hw, canny_low, canny_high
    )
    base_edges = canny_frames(base, control_hw, canny_low, canny_high)
    controlled_edges = canny_frames(
        controlled, control_hw, canny_low, canny_high
    )

    base_all = edge_sequence_metrics(reference_edges, base_edges, tolerance)
    controlled_all = edge_sequence_metrics(
        reference_edges, controlled_edges, tolerance
    )
    if count > 1:
        base_motion = edge_sequence_metrics(
            reference_edges[1:], base_edges[1:], tolerance
        )
        controlled_motion = edge_sequence_metrics(
            reference_edges[1:], controlled_edges[1:], tolerance
        )
    else:
        base_motion = controlled_motion = None

    frame_zero_mad = float(
        np.abs(
            base[0].astype(np.float32) - controlled[0].astype(np.float32)
        ).mean()
    )

    def metric_delta(base_metrics, controlled_metrics, name):
        if base_metrics is None or controlled_metrics is None:
            return None
        base_value = base_metrics["aggregate"][name]
        controlled_value = controlled_metrics["aggregate"][name]
        if base_value is None or controlled_value is None:
            return None
        return controlled_value - base_value

    per_frame_effect = np.abs(
        base.astype(np.float32) - controlled.astype(np.float32)
    ).mean(axis=(1, 2, 3))
    return {
        "frames": count,
        "edge_adherence": {
            "base_all": base_all,
            "controlled_all": controlled_all,
            "base_frames_1_plus": base_motion,
            "controlled_frames_1_plus": controlled_motion,
            "delta_f1_all": metric_delta(base_all, controlled_all, "f1"),
            "delta_f1_frames_1_plus": metric_delta(
                base_motion, controlled_motion, "f1"
            ),
            "delta_chamfer_all": metric_delta(
                base_all, controlled_all, "symmetric_chamfer"
            ),
        },
        "frame_zero": {
            "base_controlled_mad": frame_zero_mad,
            "reported_separately": True,
        },
        "video_health": {
            "base": video_health(base),
            "controlled": video_health(controlled),
        },
        "base_controlled_pixel_mad": {
            "per_frame": per_frame_effect.tolist(),
            "all_frames": float(per_frame_effect.mean()),
            "frames_1_plus": (
                float(per_frame_effect[1:].mean()) if count > 1 else None
            ),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--controlled", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--control-height", type=int, default=128)
    parser.add_argument("--control-width", type=int, default=128)
    parser.add_argument("--canny-low", type=int, default=100)
    parser.add_argument("--canny-high", type=int, default=200)
    parser.add_argument("--tolerance", type=int, default=3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = evaluate(
        read_video(args.reference),
        read_video(args.base),
        read_video(args.controlled),
        control_hw=(args.control_height, args.control_width),
        canny_low=args.canny_low,
        canny_high=args.canny_high,
        tolerance=args.tolerance,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
