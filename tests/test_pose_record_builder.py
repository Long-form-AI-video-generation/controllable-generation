"""Contract tests for the builder, including integration with PR #16's validator."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.pose.demo_record_builder import make_synthetic_request
from src.pose_models.record_builder import (
    PoseRecordBuildError, SCORE_POLICY, UNAVAILABLE_SCORE_POLICY,
    build_pose_record, frame_observation, save_pose_record,
)

HAS_VALIDATOR = importlib.util.find_spec("src.pose_models.record_validation") is not None


class FrameConversionTests(unittest.TestCase):
    def setUp(self):
        observation = make_synthetic_request()["observations"][0]
        self.result = observation["frame_result"]
        self.selection = observation["selection"]

    def convert(self):
        return frame_observation(10, self.result, self.selection, width=160, height=100,
                                 validity_threshold=0.5, score_policy=SCORE_POLICY)

    def test_normalization_missing_and_threshold(self):
        joints = self.convert()["joints"]
        self.assertEqual(joints[0], {"xy": [1.0, 1.0], "confidence": 0.5, "valid": True})
        self.assertEqual(joints[1], {"xy": None, "confidence": 0.9, "valid": False})
        self.assertEqual(joints[2], {"xy": None, "confidence": 0.8, "valid": False})
        self.assertEqual(joints[3]["xy"], [10 / 159, 10 / 99])
        self.assertFalse(joints[3]["valid"])
        self.assertIsNone(joints[4]["confidence"])
        self.assertFalse(joints[4]["valid"])

    def test_origin_is_available(self):
        self.result["candidates"][0]["xy_pixels"][0] = [0, 0]
        self.assertEqual(self.convert()["joints"][0]["xy"], [0, 0])

    def test_does_not_mutate_inputs(self):
        before = deepcopy((self.result, self.selection))
        self.convert()
        self.assertEqual((self.result, self.selection), before)

    def test_raw_scores_are_never_clipped(self):
        for score in (-0.2, 1.2, float("nan"), float("inf"), True, "0.5"):
            with self.subTest(score=score):
                self.result["candidates"][0]["joint_scores_raw"][0] = score
                with self.assertRaisesRegex(PoseRecordBuildError, "score_raw"):
                    self.convert()

    def test_unavailable_confidence_preserves_inputs_and_never_marks_valid(self):
        self.result["candidates"][0]["joint_scores_raw"][0] = 1.052
        before = deepcopy(self.result)
        frame = frame_observation(10, self.result, self.selection, width=160, height=100,
                                  validity_threshold=0.5, score_policy=UNAVAILABLE_SCORE_POLICY)
        self.assertEqual(frame["joints"][0]["xy"], [1.0, 1.0])
        self.assertTrue(all(j["confidence"] is None and not j["valid"] for j in frame["joints"]))
        self.assertEqual(self.result, before)

    def test_unavailable_confidence_still_rejects_malformed_raw_scores(self):
        for score in (float("nan"), float("inf"), True, "0.5", 10 ** 400):
            with self.subTest(score=score):
                self.result["candidates"][0]["joint_scores_raw"][0] = score
                with self.assertRaisesRegex(PoseRecordBuildError, "score_raw"):
                    frame_observation(10, self.result, self.selection, width=160, height=100,
                                      validity_threshold=0.5,
                                      score_policy=UNAVAILABLE_SCORE_POLICY)

    def test_ambiguous_has_no_observed_joints(self):
        self.selection = {"subject_status": "ambiguous", "subject_id": None, "candidate_id": None}
        self.assertEqual(self.convert()["joints"],
                         [{"xy": None, "confidence": None, "valid": False}] * 17)

    def test_rejects_missing_candidate_and_wrong_dimensions(self):
        self.selection["candidate_id"] = "candidate_9"
        with self.assertRaisesRegex(PoseRecordBuildError, "candidate_id"):
            self.convert()
        self.selection["candidate_id"] = "candidate_0"
        self.result["width"] = 161
        with self.assertRaisesRegex(PoseRecordBuildError, "width"):
            self.convert()

    def test_rejects_bad_selected_payloads(self):
        for key, value in (("joint_names", []), ("xy_pixels", []),
                           ("invalid_reasons", [None] * 17)):
            with self.subTest(key=key):
                original = deepcopy(self.result)
                self.result["candidates"][0][key] = value
                with self.assertRaises(PoseRecordBuildError):
                    self.convert()
                self.result = original

    def test_rejects_failed_extraction_instead_of_marking_absent(self):
        self.result["status"] = "error"
        with self.assertRaisesRegex(PoseRecordBuildError, "status"):
            self.convert()


@unittest.skipUnless(HAS_VALIDATOR, "Requires record_validation.py from PR #16")
class RecordIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.request = make_synthetic_request()

    def test_full_path_and_repeated_sampling(self):
        from src.pose_models.record_validation import validate_pose_record
        before = deepcopy(self.request)
        record = build_pose_record(self.request)
        self.assertIsNone(validate_pose_record(record))
        self.assertEqual([f["source_frame_index"] for f in record["frames"]], [10, 11, 12, 12, 12])
        self.assertEqual([f["subject_status"] for f in record["frames"]],
                         ["selected", "absent", "ambiguous", "ambiguous", "ambiguous"])
        self.assertEqual(record["frames"][2], record["frames"][3])
        self.assertIsNot(record["frames"][2], record["frames"][3])
        self.assertEqual(self.request, before)

    def test_unavailable_confidence_record_is_valid_but_has_no_valid_joints(self):
        from src.pose_models.record_validation import validate_pose_record
        self.request["score_policy"] = UNAVAILABLE_SCORE_POLICY
        self.request["observations"][0]["frame_result"]["candidates"][0]["joint_scores_raw"][0] = 1.052
        record = build_pose_record(self.request)
        self.assertIsNone(validate_pose_record(record))
        self.assertFalse(any(j["valid"] for f in record["frames"] for j in f["joints"]))

    def test_requires_exact_sampling_observations(self):
        for index in (9, 13, True):
            with self.subTest(index=index):
                request = deepcopy(self.request)
                request["observations"][0]["source_frame_index"] = index
                with self.assertRaises(PoseRecordBuildError):
                    build_pose_record(request)
        self.request["observations"].append(deepcopy(self.request["observations"][0]))
        with self.assertRaisesRegex(PoseRecordBuildError, "duplicate"):
            build_pose_record(self.request)

    def test_matches_even_spacing_not_arbitrary_in_range_indices(self):
        request = self.request
        request["sampling"].update(end_frame=20, num_frames=3)
        with self.assertRaisesRegex(PoseRecordBuildError, "shared-sampler"):
            build_pose_record(request)
        for observation, index in zip(request["observations"], (10, 14, 19)):
            observation["source_frame_index"] = index
        self.assertEqual([f["source_frame_index"] for f in build_pose_record(request)["frames"]],
                         [10, 14, 19])

    def test_rejects_identity_switch(self):
        from src.pose_models.record_validation import PoseRecordValidationError
        other = deepcopy(self.request["observations"][0])
        other["source_frame_index"] = 11
        other["selection"]["subject_id"] = "different-subject"
        self.request["observations"][1] = other
        with self.assertRaises(PoseRecordValidationError):
            build_pose_record(self.request)

    def test_rejects_test_partition_and_unresolved_interval(self):
        self.request["source"]["partition"] = "test"
        with self.assertRaisesRegex(PoseRecordBuildError, "train/val"):
            build_pose_record(self.request)
        self.request["source"]["partition"] = "train"
        self.request["sampling"]["end_frame"] = 21
        with self.assertRaisesRegex(PoseRecordBuildError, "resolved interval"):
            build_pose_record(self.request)

    def test_rejects_changed_model_info(self):
        self.request["observations"][1]["frame_result"]["model_info"] = {"other": "model"}
        with self.assertRaisesRegex(PoseRecordBuildError, "model_info"):
            build_pose_record(self.request)

    def test_save_validates_before_writing_and_refuses_overwrite(self):
        from src.pose_models.record_validation import PoseRecordValidationError, load_pose_record
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "record.json"
            record = build_pose_record(self.request)
            save_pose_record(record, output)
            self.assertEqual(load_pose_record(output), record)
            with self.assertRaises(FileExistsError):
                save_pose_record(record, output)
            output.unlink()
            record["frames"][0]["joints"][0]["valid"] = False
            with self.assertRaises(PoseRecordValidationError):
                save_pose_record(record, output)
            self.assertFalse(output.exists())

    def test_cli_success_and_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            request_path = Path(directory) / "request.json"
            output = Path(directory) / "record.json"
            request_path.write_text(json.dumps(self.request), encoding="utf-8")
            command = [sys.executable, "-m", "src.pose_models.record_builder",
                       str(request_path), str(output)]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(output.exists())
            output.unlink()
            for bad_json in ('{"source":{},"source":{}}', '{"x": NaN}', '{broken'):
                request_path.write_text(bad_json, encoding="utf-8")
                result = subprocess.run(command, capture_output=True, text=True)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
