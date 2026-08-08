"""Create deterministic one-channel Canny controls from source videos."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from src.data.frame_sampling import select_frame_indices
from src.sketch_models.preprocessing import (
    CannyConfig,
    prepare_canny_sequence,
    validate_canny_tensor,
)


def _read_selected_frames(
    video_path: Path,
    start_frame: int,
    end_frame: int,
    num_frames: int,
) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video_path}")

    frames = []
    try:
        for index in select_frame_indices(start_frame, end_frame, num_frames):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            okay, frame = capture.read()
            if not okay:
                raise RuntimeError(
                    f"Could not decode frame {index} from {video_path}"
                )
            frames.append(frame)
    finally:
        capture.release()
    return np.stack(frames)


def _output_path(output_dir: Path, record: dict) -> Path:
    video_id = str(record["video_id"])
    shot_id = str(record["shot_id"])
    return (
        output_dir
        / video_id
        / f"shot_{shot_id}_controls_encoded.npz"
    )


def prepare_record(
    record: dict,
    videos_dir: Path,
    output_dir: Path,
    config: CannyConfig,
    *,
    resume: bool = False,
    overwrite: bool = False,
) -> dict:
    video_id = str(record["video_id"])
    video_path = videos_dir / f"{video_id}.mp4"
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    output_path = _output_path(output_dir, record)
    if output_path.exists() and resume:
        with np.load(output_path, allow_pickle=False) as data:
            if set(data.files) != {"sketch_encoded"}:
                raise ValueError(f"Invalid keys in existing {output_path}")
            sketch = np.asarray(data["sketch_encoded"])
        validate_canny_tensor(sketch)
        expected = (1, 1, config.num_frames, *config.output_size)
        if sketch.shape != expected:
            raise ValueError(
                f"Existing {output_path} has {sketch.shape}, expected {expected}"
            )
        return {
            "video_id": video_id,
            "shot_id": str(record["shot_id"]),
            "path": str(output_path.relative_to(output_dir)),
            "shape": list(sketch.shape),
            "sha256": hashlib.sha256(sketch.tobytes()).hexdigest(),
            "nonzero_fraction": float(np.count_nonzero(sketch) / sketch.size),
            "resumed": True,
        }
    if output_path.exists() and not overwrite:
        raise FileExistsError(output_path)

    frames = _read_selected_frames(
        video_path,
        int(record["segment_start_frame"]),
        int(record["segment_end_frame"]),
        config.num_frames,
    )
    sketch = prepare_canny_sequence(frames, config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary_path, sketch_encoded=sketch)
    temporary_path.replace(output_path)

    return {
        "video_id": video_id,
        "shot_id": str(record["shot_id"]),
        "path": str(output_path.relative_to(output_dir)),
        "shape": list(sketch.shape),
        "sha256": hashlib.sha256(sketch.tobytes()).hexdigest(),
        "nonzero_fraction": float(np.count_nonzero(sketch) / sketch.size),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos-dir", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--canny-low", type=int, default=100)
    parser.add_argument("--canny-high", type=int, default=200)
    behavior = parser.add_mutually_exclusive_group()
    behavior.add_argument("--resume", action="store_true")
    behavior.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    from tqdm import tqdm

    args = build_parser().parse_args()
    videos_dir = Path(args.videos_dir).resolve()
    metadata_path = Path(args.metadata).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "sketch_preprocessing_manifest.json"
    if manifest_path.exists() and not (args.overwrite or args.resume):
        raise FileExistsError(
            f"{manifest_path} already exists; pass --resume to validate/skip "
            "existing outputs or --overwrite to regenerate"
        )

    config = CannyConfig(
        low_threshold=args.canny_low,
        high_threshold=args.canny_high,
        num_frames=args.num_frames,
        output_size=(args.height, args.width),
        color_order="BGR",
    )
    records = json.loads(metadata_path.read_text(encoding="utf-8"))
    records = sorted(records, key=lambda item: str(item["shot_id"]))
    results = [
        prepare_record(
            record,
            videos_dir,
            output_dir,
            config,
            resume=args.resume,
            overwrite=args.overwrite,
        )
        for record in tqdm(records, desc="Preparing Canny controls")
    ]
    manifest = {
        "format_version": 1,
        "control_key": "sketch_encoded",
        "metadata_path": str(metadata_path),
        "metadata_sha256": hashlib.sha256(
            metadata_path.read_bytes()
        ).hexdigest(),
        "preprocessing": config.to_metadata(),
        "records": results,
    }
    temporary_manifest = manifest_path.with_suffix(".tmp.json")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_manifest.replace(manifest_path)
    print(f"Prepared {len(results)} sketch controls")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
