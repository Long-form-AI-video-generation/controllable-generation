#!/usr/bin/env python3
"""Build one reviewed smoke-test frame into an audited, non-training pose record."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.pose_models.extraction.dwpose_onnx import sha256_file
from src.pose_models.record_builder import (
    SUBJECT_POLICY, UNAVAILABLE_SCORE_POLICY, build_pose_record, save_pose_record,
    _unique_object, _reject_constant,
)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=_unique_object, parse_constant=_reject_constant)


def video_metadata(path):
    """Verify constant presentation-time spacing; average FPS alone is insufficient."""
    try:
        result = subprocess.run([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height:frame=best_effort_timestamp_time",
            "-show_streams", "-show_frames", "-of", "json", str(path),
        ], check=True, capture_output=True, text=True)
    except FileNotFoundError as error:
        raise ValueError("ffprobe is required to verify constant frame rate") from error
    except subprocess.CalledProcessError as error:
        raise ValueError(f"ffprobe failed: {error.stderr.strip()}") from error
    return metadata_from_probe(json.loads(result.stdout))


def metadata_from_probe(probe):
    streams = probe.get("streams", [])
    frames = probe.get("frames", [])
    if len(streams) != 1 or len(frames) < 2:
        raise ValueError("expected one video stream with at least two timestamped frames")
    width, height = streams[0].get("width"), streams[0].get("height")
    if any(type(v) is not int or v < 2 for v in (width, height)):
        raise ValueError("invalid video dimensions")
    try:
        times = [float(frame["best_effort_timestamp_time"]) for frame in frames]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("every decoded frame needs a presentation timestamp") from error
    if not all(math.isfinite(t) for t in times):
        raise ValueError("non-finite frame timestamps")
    step = (times[-1] - times[0]) / (len(times) - 1)
    if step <= 0 or any(abs((b - a) - step) > 1e-5 for a, b in zip(times, times[1:])):
        raise ValueError("video does not have constant frame timing; pose-pilot-v1 requires it")
    return {"width": width, "height": height, "frame_count": len(times),
            "fps": 1.0 / step, "timing": "constant_fps"}


def development_partition(manifest, video_id):
    if not isinstance(manifest, dict) or any(
        not isinstance(manifest.get(key), list) for key in ("train", "val", "test")
    ):
        raise ValueError("split manifest must have train, val, and test ID lists")
    matches = [key for key in ("train", "val", "test")
               for value in manifest[key] if str(value) == video_id]
    if len(matches) != 1 or matches[0] == "test":
        raise ValueError("video must occur exactly once in train/val, and never in test")
    return matches[0]


def prepare_request(args, metadata):
    payload = load_json(args.smoke_result)
    source_info = payload["source"]
    result = payload["result"]
    if source_info.get("kind") != "video_frame":
        raise ValueError("smoke result must come from a source video frame")
    if Path(source_info["path"]).resolve() != args.video.resolve():
        raise ValueError("smoke-result source path differs from the supplied video")
    for key in ("width", "height"):
        if result.get(key) != metadata[key] or source_info.get(key) != metadata[key]:
            raise ValueError(f"smoke-result {key} differs from the probed video")
    index = source_info["source_frame_index"]
    if type(index) is not int or not 0 <= index < metadata["frame_count"]:
        raise ValueError("source_frame_index is outside the video")
    if not re.fullmatch(r"[0-9a-f]{40}", args.extractor_revision):
        raise ValueError("extractor revision must be the full 40-character producing commit SHA")
    annotations = load_json(args.annotations)
    if not isinstance(annotations, list) or not any(
        isinstance(item, dict) and str(item.get("video_id")) == args.video_id
        and item.get("shot_id") == args.shot_id for item in annotations
    ):
        raise ValueError("video_id and shot_id pair not found in annotations")
    partition = development_partition(load_json(args.split_manifest), args.video_id)
    info = result["model_info"]
    weights = {}
    for role, path in (("detector", args.detector_model), ("pose", args.pose_model)):
        digest = sha256_file(path)
        if digest != info[role]["sha256"]:
            raise ValueError(f"{role} weights differ from smoke-result model metadata")
        weights[role] = {"filename": path.name, "sha256": digest}
    configuration = {
        "purpose": "pipeline_smoke_only", "training_eligible": False,
        "score_policy": UNAVAILABLE_SCORE_POLICY,
        "score_policy_note": "No confidence conversion selected; all record confidences null and valid flags false.",
        "subject_policy": SUBJECT_POLICY, "candidate_id": args.candidate_id,
        "subject_id": "subject-0", "source_frame_index": index,
        "validity_threshold": 0.5,
        "threshold_note": "Schema placeholder only; unavailable confidence makes every joint invalid.",
        "extractor_revision": args.extractor_revision, "model_info": info,
        "smoke_result_sha256": sha256_file(args.smoke_result),
        "source_binding": "legacy smoke source path plus current video hash; source unchanged since inference is assumed",
        "timing_verification": "ffprobe decoded presentation timestamps; absolute spacing tolerance 0.00001 seconds",
        "visual_review_note": args.review_note,
        "limitations": ["Single-frame format check; no temporal validation.",
                        "Subject selection is an explicit human decision; candidate IDs are not track IDs.",
                        "Annotations confirm the ID pair; shot interval membership is not inferred."],
    }
    # Hash the exact retained JSON bytes, including the final newline.
    weights_text = json.dumps(weights, sort_keys=True, indent=2, allow_nan=False) + "\n"
    config_text = json.dumps(configuration, sort_keys=True, indent=2, allow_nan=False) + "\n"
    request = {
        "source": {**metadata, "video_id": args.video_id, "shot_id": args.shot_id,
                   "video_sha256": sha256_file(args.video),
                   "annotations_sha256": sha256_file(args.annotations),
                   "split_manifest_sha256": sha256_file(args.split_manifest), "partition": partition},
        "sampling": {"start_frame": index, "end_frame": index + 1, "num_frames": 1,
                     "policy": "shared_select_frame_indices_v1"},
        "extraction": {"extractor_id": "dwpose-onnx", "extractor_revision": args.extractor_revision,
                       "mapping_revision": "coco17-first17-v1",
                       "preprocessing_revision": "dwpose-onnx-source-pixels-v1",
                       "subject_policy_revision": SUBJECT_POLICY,
                       "weights_sha256": hashlib.sha256(weights_text.encode("utf-8")).hexdigest(),
                       "config_sha256": hashlib.sha256(config_text.encode("utf-8")).hexdigest(),
                       "validity_threshold": 0.5},
        "score_policy": UNAVAILABLE_SCORE_POLICY,
        "observations": [{"source_frame_index": index, "frame_result": result,
                          "selection": {"subject_status": "selected", "subject_id": "subject-0",
                                        "candidate_id": args.candidate_id}}],
    }
    return request, weights_text, config_text


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("smoke-result", "video", "annotations", "split-manifest",
                 "detector-model", "pose-model", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("video-id", "shot-id", "candidate-id", "extractor-revision", "review-note"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args(argv)
    try:
        request, weights_text, config_text = prepare_request(args, video_metadata(args.video))
        record = build_pose_record(request)
        args.output_dir.mkdir(parents=True, exist_ok=False)
        (args.output_dir / "weights_manifest.json").write_text(weights_text, encoding="utf-8")
        (args.output_dir / "extraction_config.json").write_text(config_text, encoding="utf-8")
        (args.output_dir / "request.json").write_text(
            json.dumps(request, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        (args.output_dir / "smoke_result.json").write_bytes(args.smoke_result.read_bytes())
        save_pose_record(record, args.output_dir / "record.json")
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"Reviewed record build failed: {error}", file=sys.stderr)
        return 1
    print(f"Saved validated pipeline-test record: {args.output_dir / 'record.json'}")
    print("Training eligible: NO. All confidence values are unavailable; all joints are invalid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
