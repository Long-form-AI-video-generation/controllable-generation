"""Audit raw SegFormer labels against coarse groups on adjacent frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .semantic_groups import (
    SemanticGroup,
    apply_group_lookup,
    build_group_lookup,
)
from .temporal_stability import summarize_stability


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare adjacent-frame stability of raw and coarse masks"
    )
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--video-ids", nargs="+", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames-per-video", type=int, default=24)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def read_adjacent_window(
    video_path: Path,
    start_frame: int,
    end_frame: int,
    frame_count: int,
) -> tuple[np.ndarray, list[int]]:
    """Read one centered run of consecutive RGB frames."""
    capture = cv2.VideoCapture(str(video_path))
    actual = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    stop = min(end_frame, actual)
    if actual <= 0 or start_frame >= stop:
        capture.release()
        raise RuntimeError(f"Invalid frame interval for {video_path}")

    available = stop - start_frame
    count = min(frame_count, available)
    first = start_frame + max(0, (available - count) // 2)
    last = first + count
    capture.set(cv2.CAP_PROP_POS_FRAMES, first)

    frames: list[np.ndarray] = []
    indices: list[int] = []
    for index in range(first, last):
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"Could not decode frame {index} from {video_path}")
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        indices.append(index)
    capture.release()
    return np.stack(frames), indices


@torch.inference_mode()
def predict_labels(
    frames: np.ndarray,
    processor: Any,
    model: torch.nn.Module,
    device: torch.device,
    output_size: tuple[int, int],
) -> np.ndarray:
    """Predict categorical maps in small batches to bound memory."""
    predictions: list[np.ndarray] = []
    for offset in range(0, len(frames), 4):
        images = [Image.fromarray(frame) for frame in frames[offset:offset + 4]]
        inputs = processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        logits = model(pixel_values=pixel_values).logits
        logits = F.interpolate(
            logits,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        predictions.append(logits.argmax(dim=1).byte().cpu().numpy())
    return np.concatenate(predictions)


def palette(size: int) -> np.ndarray:
    """Create a deterministic high-contrast RGB palette."""
    colors = np.zeros((size, 3), dtype=np.uint8)
    for index in range(size):
        colors[index] = (
            (37 * index + 53) % 256,
            (97 * index + 101) % 256,
            (193 * index + 17) % 256,
        )
    colors[0] = (30, 30, 30)
    return colors


def labelled_panel(image: np.ndarray, label: str) -> np.ndarray:
    panel = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    panel = cv2.copyMakeBorder(panel, 30, 0, 0, 0, cv2.BORDER_CONSTANT)
    cv2.putText(
        panel,
        label,
        (8, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return panel


def save_contact_sheet(
    path: Path,
    frames: np.ndarray,
    raw: np.ndarray,
    coarse: np.ndarray,
    indices: list[int],
) -> None:
    raw_colors = palette(150)
    coarse_colors = palette(len(SemanticGroup))
    sample_positions = sorted({0, len(frames) // 2, len(frames) - 1})
    rows = []
    for position in sample_positions:
        source = cv2.resize(frames[position], (384, 216))
        raw_rgb = cv2.resize(
            raw_colors[raw[position]], (384, 216), interpolation=cv2.INTER_NEAREST
        )
        coarse_rgb = cv2.resize(
            coarse_colors[coarse[position]],
            (384, 216),
            interpolation=cv2.INTER_NEAREST,
        )
        rows.append(
            np.hstack(
                [
                    labelled_panel(source, f"frame {indices[position]} | source"),
                    labelled_panel(raw_rgb, "raw 150 classes"),
                    labelled_panel(coarse_rgb, "coarse groups"),
                ]
            )
        )
    cv2.imwrite(str(path), np.vstack(rows))


def main() -> None:
    args = parse_args()
    if args.frames_per_video < 2:
        raise ValueError("--frames-per-video must be at least 2")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

    processor = AutoImageProcessor.from_pretrained(
        args.model_id,
        local_files_only=True,
    )
    model = AutoModelForSemanticSegmentation.from_pretrained(
        args.model_id,
        local_files_only=True,
        use_safetensors=True,
    )
    device = torch.device(args.device)
    model.to(device).eval()
    lookup = build_group_lookup(model.config.id2label)

    records = json.loads(args.metadata.read_text())
    by_video = {str(record["video_id"]): record for record in records}
    reports = []

    for video_id in args.video_ids:
        if video_id not in by_video:
            raise KeyError(f"No metadata record for video {video_id}")
        record = by_video[video_id]
        frames, indices = read_adjacent_window(
            args.videos_dir / f"{video_id}.mp4",
            int(record["segment_start_frame"]),
            int(record["segment_end_frame"]),
            args.frames_per_video,
        )
        raw = predict_labels(
            frames,
            processor,
            model,
            device,
            (args.height, args.width),
        )
        coarse = apply_group_lookup(raw, lookup)
        report = summarize_stability(raw, coarse)
        report.update({"video_id": video_id, "frame_indices": indices})
        reports.append(report)
        save_contact_sheet(
            args.output_dir / f"video_{video_id}_contact_sheet.jpg",
            frames,
            raw,
            coarse,
            indices,
        )
        print(
            f"video={video_id} "
            f"raw={report['raw']['mean_adjacent_agreement']:.4f} "
            f"coarse={report['coarse']['mean_adjacent_agreement']:.4f} "
            f"delta={report['coarse_agreement_improvement']:+.4f}"
        )

    raw_mean = float(np.mean([
        item["raw"]["mean_adjacent_agreement"] for item in reports
    ]))
    coarse_mean = float(np.mean([
        item["coarse"]["mean_adjacent_agreement"] for item in reports
    ]))
    output = {
        "settings": {
            "model_id": args.model_id,
            "frames_per_video": args.frames_per_video,
            "output_size": [args.height, args.width],
            "video_ids": args.video_ids,
            "group_names": {int(group): group.name.lower() for group in SemanticGroup},
        },
        "aggregate": {
            "raw_mean_adjacent_agreement": raw_mean,
            "coarse_mean_adjacent_agreement": coarse_mean,
            "coarse_agreement_improvement": coarse_mean - raw_mean,
        },
        "videos": reports,
    }
    report_path = args.output_dir / "temporal_stability_report.json"
    report_path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
