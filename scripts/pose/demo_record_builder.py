"""Make a synthetic, model-free example using the real frame wrapper and builder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.pose_models.extraction.dwpose import CANONICAL_JOINT_NAMES, DWPoseFrameExtractor
from src.pose_models.record_builder import (
    SCORE_POLICY, SUBJECT_POLICY, build_pose_record, save_pose_record,
)


class _Detector:
    def __init__(self, count):
        self.count = count

    def detect(self, image):
        return [{"box_xyxy": [0, 0, 160, 100], "score_raw": 0.9}
                for _ in range(self.count)]


class _Estimator:
    def estimate(self, image, boxes):
        points = [[159, 99], None, [400, 10], [10, 10], [10, 10]] + [[20, 30]] * 12
        return [{"joint_names": list(CANONICAL_JOINT_NAMES), "xy": points,
                 "scores_raw": [0.5, 0.9, 0.8, 0.1, None] + [0.8] * 12,
                 "coordinate_space": "original_pixels", "to_original": None}
                for _ in boxes]


def make_synthetic_request():
    """Placeholder hashes and observations are synthetic; never use for training."""
    image = np.zeros((100, 160, 3), dtype=np.uint8)
    observations = []
    for index, count, status in ((10, 1, "selected"), (11, 0, "absent"), (12, 2, "ambiguous")):
        wrapper = DWPoseFrameExtractor(_Detector(count), _Estimator(), {"kind": "simulated"})
        observations.append({
            "source_frame_index": index, "frame_result": wrapper.extract_frame(image),
            "selection": {"subject_status": status,
                          "subject_id": "subject-0" if status == "selected" else None,
                          "candidate_id": "candidate_0" if status == "selected" else None},
        })
    return {
        "source": {
            "video_id": "synthetic-video", "shot_id": "synthetic-shot",
            "video_sha256": "a" * 64, "annotations_sha256": "b" * 64,
            "split_manifest_sha256": "c" * 64, "partition": "train",
            "width": 160, "height": 100, "frame_count": 20, "fps": 16.0,
            "timing": "constant_fps",
        },
        "sampling": {"start_frame": 10, "end_frame": 13, "num_frames": 5,
                     "policy": "shared_select_frame_indices_v1"},
        "extraction": {
            "extractor_id": "synthetic-wrapper-demo", "extractor_revision": "synthetic-v1",
            "mapping_revision": "coco17-v1", "preprocessing_revision": "identity-synthetic-v1",
            "subject_policy_revision": SUBJECT_POLICY, "weights_sha256": "d" * 64,
            "config_sha256": "e" * 64, "validity_threshold": 0.5,
        },
        "score_policy": SCORE_POLICY, "observations": observations,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, help="new directory for synthetic artifacts")
    args = parser.parse_args(argv)
    try:
        request = make_synthetic_request()
        record = build_pose_record(request)
        args.output_dir.mkdir(parents=True, exist_ok=False)
        (args.output_dir / "synthetic_request.json").write_text(
            json.dumps(request, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        save_pose_record(record, args.output_dir / "synthetic_record.json")
    except (ValueError, OSError) as error:
        print(f"Synthetic demo failed: {error}", file=sys.stderr)
        return 1
    print(f"Saved synthetic request and validated record in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
