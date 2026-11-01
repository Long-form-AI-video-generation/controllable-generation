"""Checks for the one-frame audited pipeline-test command."""

import argparse
import contextlib
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.pose.build_reviewed_record import (
    development_partition, main, metadata_from_probe, prepare_request,
)
from scripts.pose.demo_record_builder import make_synthetic_request
from src.pose_models.record_builder import build_pose_record
from src.pose_models.record_validation import load_pose_record


class TimingAndMembershipTests(unittest.TestCase):
    def test_accepts_uniform_timestamps_and_rejects_variable_rate(self):
        probe = {"streams": [{"width": 160, "height": 100}],
                 "frames": [{"best_effort_timestamp_time": str(t)} for t in (0, .04, .08)]}
        self.assertAlmostEqual(metadata_from_probe(probe)["fps"], 25)
        probe["frames"][2]["best_effort_timestamp_time"] = ".09"
        with self.assertRaisesRegex(ValueError, "constant frame"):
            metadata_from_probe(probe)
        probe["frames"][2] = {}
        with self.assertRaisesRegex(ValueError, "timestamp"):
            metadata_from_probe(probe)

    def test_requires_unique_development_membership(self):
        manifest = {"train": ["02"], "val": [], "test": []}
        self.assertEqual(development_partition(manifest, "02"), "train")
        for bad in ({"train": [], "val": [], "test": ["02"]},
                    {"train": ["02", "02"], "val": [], "test": []},
                    {"train": ["02"], "val": [], "test": ["02"]}):
            with self.assertRaises(ValueError):
                development_partition(bad, "02")


class ReviewedRecordTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.args = argparse.Namespace(
            smoke_result=root / "frame_result.json", video=root / "02.mp4",
            annotations=root / "annotations.json", split_manifest=root / "split.json",
            detector_model=root / "det.onnx", pose_model=root / "pose.onnx",
            output_dir=root / "record-bundle", video_id="02", shot_id="02_shot_0000",
            candidate_id="candidate_0", extractor_revision="d" * 40,
            review_note="Main subject selected; leg landmarks inaccurate; pipeline test only.",
        )
        self.args.video.write_bytes(b"synthetic video placeholder")
        self.args.detector_model.write_bytes(b"detector test bytes")
        self.args.pose_model.write_bytes(b"pose test bytes")
        self.args.annotations.write_text(json.dumps([{"video_id": "02", "shot_id": "02_shot_0000"}]))
        self.args.split_manifest.write_text(json.dumps({"train": ["02"], "val": [], "test": []}))
        self.result = make_synthetic_request()["observations"][0]["frame_result"]
        self.result["candidates"][0]["joint_scores_raw"][0] = 1.052
        self.result["model_info"] = {
            role: {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for role, path in (("detector", self.args.detector_model), ("pose", self.args.pose_model))
        }
        self.metadata = {"width": 160, "height": 100, "frame_count": 20,
                         "fps": 25.0, "timing": "constant_fps"}
        self.payload = {"source": {"kind": "video_frame", "path": str(self.args.video),
                                   "source_frame_index": 10, "width": 160, "height": 100},
                        "result": self.result}
        self.write_smoke()

    def write_smoke(self):
        self.args.smoke_result.write_text(json.dumps(self.payload))

    def test_retains_raw_scores_and_hashes_exact_configuration_bytes(self):
        before = deepcopy(self.payload)
        request, weights, config = prepare_request(self.args, self.metadata)
        record = build_pose_record(request)
        self.assertEqual(request["observations"][0]["frame_result"], before["result"])
        self.assertIsNone(record["frames"][0]["joints"][0]["confidence"])
        self.assertFalse(any(j["valid"] for j in record["frames"][0]["joints"]))
        self.assertEqual(record["extraction"]["weights_sha256"], hashlib.sha256(weights.encode()).hexdigest())
        self.assertEqual(record["extraction"]["config_sha256"], hashlib.sha256(config.encode()).hexdigest())
        self.assertFalse(json.loads(config)["training_eligible"])

    def test_rejects_wrong_video_dimensions_or_weights(self):
        self.payload["source"]["path"] = "/wrong/02.mp4"
        self.write_smoke()
        with self.assertRaisesRegex(ValueError, "path"):
            prepare_request(self.args, self.metadata)
        self.payload["source"]["path"] = str(self.args.video)
        self.payload["source"]["width"] = 161
        self.write_smoke()
        with self.assertRaisesRegex(ValueError, "width"):
            prepare_request(self.args, self.metadata)
        self.payload["source"]["width"] = 160
        self.write_smoke()
        self.args.pose_model.write_bytes(b"changed weights")
        with self.assertRaisesRegex(ValueError, "weights"):
            prepare_request(self.args, self.metadata)

    def test_rejects_missing_annotation_or_candidate(self):
        self.args.shot_id = "wrong-shot"
        with self.assertRaisesRegex(ValueError, "annotations"):
            prepare_request(self.args, self.metadata)
        self.args.shot_id = "02_shot_0000"
        self.args.candidate_id = "candidate_999"
        request, _, _ = prepare_request(self.args, self.metadata)
        with self.assertRaisesRegex(ValueError, "candidate_id"):
            build_pose_record(request)

    def test_command_writes_valid_bundle_and_refuses_overwrite(self):
        argv = [part for name, value in vars(self.args).items()
                for part in ("--" + name.replace("_", "-"), str(value))]
        with patch("scripts.pose.build_reviewed_record.video_metadata", return_value=self.metadata), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(argv), 0)
            record = load_pose_record(self.args.output_dir / "record.json")
            self.assertEqual(record["sampling"]["num_frames"], 1)
            self.assertEqual(main(argv), 1)
        raw = json.loads((self.args.output_dir / "smoke_result.json").read_text())
        self.assertEqual(raw, self.payload)
        config = json.loads((self.args.output_dir / "extraction_config.json").read_text())
        self.assertEqual(config["smoke_result_sha256"],
                         hashlib.sha256((self.args.output_dir / "smoke_result.json").read_bytes()).hexdigest())
        request = json.loads((self.args.output_dir / "request.json").read_text())
        self.assertEqual(build_pose_record(request), record)


if __name__ == "__main__":
    unittest.main()
