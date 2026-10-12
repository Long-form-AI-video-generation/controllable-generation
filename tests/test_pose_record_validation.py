"""Tests for ``pose-pilot-v1`` pose record validation.

The supplied example records are not committed. To include them, point
``POSE_FIXTURES_DIR`` at the folder that holds them, e.g.::

    POSE_FIXTURES_DIR=/path/to/examples python -m unittest discover -s tests -p 'test_pose_*.py' -v

Without it, the example tests are skipped and every other test still runs
against records built in this file.
"""

from __future__ import annotations

import contextlib
import copy
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Callable

from src.pose_models import (
    PoseRecordValidationError,
    load_pose_record,
    validate_pose_record,
)
from src.pose_models.record_validation import CANONICAL_JOINT_NAMES, main


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = os.environ.get("POSE_FIXTURES_DIR")

VALID_EXAMPLES = (
    "valid_full.json",
    "valid_partial.json",
    "valid_absent.json",
    "valid_ambiguous.json",
    "valid_repeated.json",
)
INVALID_EXAMPLES = {
    "invalid_joint_count.json": r"frames\[0\]\.joints: must contain exactly 17 joints, got 16",
    "invalid_coordinate.json": r"frames\[0\]\.joints\[0\]\.xy\[0\]: must be in \[0\.0, 1\.0\], got 1\.01",
    "invalid_validity.json": r"frames\[0\]\.joints\[0\]\.valid: must be False",
    "invalid_repeated.json": r"frames\[2\]: repeats source_frame_index 101",
}


def _joint(xy: list[float] | None, confidence: float | None, valid: bool) -> dict[str, Any]:
    return {"xy": xy, "confidence": confidence, "valid": valid}


def _unavailable_joint() -> dict[str, Any]:
    return _joint(None, None, False)


def _selected_frame(index: int = 100, subject_id: str = "subject-0") -> dict[str, Any]:
    # Joints run from the bottom-left corner to the top-right corner, so
    # both borders are covered.
    step = 1 / (len(CANONICAL_JOINT_NAMES) - 1)
    return {
        "source_frame_index": index,
        "subject_status": "selected",
        "subject_id": subject_id,
        "joints": [
            _joint([i * step, 1 - i * step], 0.9, True)
            for i in range(len(CANONICAL_JOINT_NAMES))
        ],
    }


def _unavailable_frame(index: int = 100, status: str = "absent") -> dict[str, Any]:
    return {
        "source_frame_index": index,
        "subject_status": status,
        "subject_id": None,
        "joints": [_unavailable_joint() for _ in CANONICAL_JOINT_NAMES],
    }


def _record(frames: list[dict[str, Any]] | None = None, end_frame: int = 101) -> dict[str, Any]:
    """Build a valid record; by default one selected frame, like valid_full."""

    frames = [_selected_frame()] if frames is None else frames
    return {
        "schema_version": "pose-pilot-v1",
        "joint_names": list(CANONICAL_JOINT_NAMES),
        "coordinate_space": "original_frame_normalized_pixel_centers",
        "source": {
            "video_id": "synthetic-video",
            "shot_id": "synthetic-shot",
            "video_sha256": "a" * 64,
            "annotations_sha256": "b" * 64,
            "split_manifest_sha256": "c" * 64,
            "partition": "train",
            "width": 832,
            "height": 480,
            "frame_count": 200,
            "fps": 16.0,
            "timing": "constant_fps",
        },
        "sampling": {
            "start_frame": 100,
            "end_frame": end_frame,
            "num_frames": len(frames),
            "policy": "shared_select_frame_indices_v1",
        },
        "extraction": {
            "extractor_id": "synthetic-fixture",
            "extractor_revision": "fixture-v1",
            "mapping_revision": "body17-v1",
            "preprocessing_revision": "identity-v1",
            "subject_policy_revision": "single-subject-v1",
            "weights_sha256": "d" * 64,
            "config_sha256": "e" * 64,
            "validity_threshold": 0.5,
        },
        "frames": frames,
    }


def _three_frame_record() -> dict[str, Any]:
    return _record(
        [_selected_frame(100), _selected_frame(101), _selected_frame(101)],
        end_frame=102,
    )


Mutation = Callable[[dict[str, Any]], None]


def _set(*keys: Any, value: Any) -> Mutation:
    def mutate(record: dict[str, Any]) -> None:
        target = record
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = value

    return mutate


def _delete(*keys: Any) -> Mutation:
    def mutate(record: dict[str, Any]) -> None:
        target = record
        for key in keys[:-1]:
            target = target[key]
        del target[keys[-1]]

    return mutate


def _joint0(field: str, value: Any) -> Mutation:
    return _set("frames", 0, "joints", 0, field, value=value)


class _RecordTestCase(unittest.TestCase):
    def assertRejected(
        self,
        mutation: Mutation,
        pattern: str,
        base: Callable[[], dict[str, Any]] = _record,
    ) -> None:
        record = base()
        mutation(record)
        with self.assertRaisesRegex(PoseRecordValidationError, pattern):
            validate_pose_record(record)

    def assertAccepted(
        self,
        mutation: Mutation,
        base: Callable[[], dict[str, Any]] = _record,
    ) -> None:
        record = base()
        mutation(record)
        self.assertIsNone(validate_pose_record(record))


@unittest.skipUnless(FIXTURES_DIR, "set POSE_FIXTURES_DIR to run the supplied example records")
class SuppliedExampleTests(unittest.TestCase):
    def _example(self, name: str) -> Path:
        path = Path(FIXTURES_DIR or "") / name
        self.assertTrue(path.is_file(), f"missing supplied example {path}")
        return path

    def test_valid_examples_are_accepted_and_returned_unchanged(self) -> None:
        for name in VALID_EXAMPLES:
            with self.subTest(example=name):
                path = self._example(name)
                expected = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(load_pose_record(path), expected)

    def test_invalid_examples_are_rejected_for_the_documented_reason(self) -> None:
        for name, pattern in INVALID_EXAMPLES.items():
            with self.subTest(example=name):
                with self.assertRaisesRegex(PoseRecordValidationError, pattern):
                    load_pose_record(self._example(name))


class AcceptedRecordTests(_RecordTestCase):
    def test_baseline_records_are_valid(self) -> None:
        for build in (
            _record,
            _three_frame_record,
            lambda: _record([_unavailable_frame(status="absent")]),
            lambda: _record([_unavailable_frame(status="ambiguous")]),
        ):
            with self.subTest(build=build):
                self.assertIsNone(validate_pose_record(build()))

    def test_boundary_and_optional_values_are_accepted(self) -> None:
        cases: dict[str, Mutation] = {
            "partition test": _set("source", "partition", value="test"),
            "partition val": _set("source", "partition", value="val"),
            "integer fps": _set("source", "fps", value=24),
            "minimum frame size": lambda r: r["source"].update(width=2, height=2),
            "end_frame equals frame_count": lambda r: r["sampling"].update(
                start_frame=199, end_frame=200
            )
            or r["frames"][0].update(source_frame_index=199),
            "threshold 0": _set("extraction", "validity_threshold", value=0),
            "zero xy and confidence with threshold 0": lambda r: (
                r["extraction"].update(validity_threshold=0),
                r["frames"][0]["joints"][0].update(xy=[0, 0], confidence=0, valid=True),
            ),
            "threshold 1 with confidence 1": lambda r: (
                r["extraction"].update(validity_threshold=1.0),
                [joint.update(confidence=1.0) for joint in r["frames"][0]["joints"]],
            ),
            "confidence equal to threshold is valid": _joint0("confidence", 0.5),
            "low confidence keeps coordinates": lambda r: r["frames"][0]["joints"][0].update(
                confidence=0.1, valid=False
            ),
            "unavailable xy keeps confidence": lambda r: r["frames"][0]["joints"][0].update(
                xy=None, valid=False
            ),
            "null confidence is invalid": lambda r: r["frames"][0]["joints"][0].update(
                confidence=None, valid=False
            ),
            "selected frame with all joints invalid": lambda r: [
                joint.update(confidence=0.0, valid=False)
                for joint in r["frames"][0]["joints"]
            ],
        }
        for name, mutation in cases.items():
            with self.subTest(case=name):
                self.assertAccepted(mutation)

    def test_selected_and_unavailable_frames_can_be_mixed(self) -> None:
        record = _record(
            [
                _selected_frame(100),
                _unavailable_frame(101, "absent"),
                _unavailable_frame(102, "ambiguous"),
                _selected_frame(103),
            ],
            end_frame=104,
        )
        self.assertIsNone(validate_pose_record(record))

    def test_validation_does_not_mutate_the_record(self) -> None:
        record = _three_frame_record()
        snapshot = copy.deepcopy(record)
        validate_pose_record(record)
        self.assertEqual(record, snapshot)


class StructureTests(_RecordTestCase):
    def test_record_must_be_an_object(self) -> None:
        for value in (None, [], "record", 1):
            with self.subTest(value=value):
                with self.assertRaisesRegex(PoseRecordValidationError, "record: must be an object"):
                    validate_pose_record(value)

    def test_missing_fields_are_rejected_at_every_level(self) -> None:
        cases = {
            "record": ("schema_version",),
            "source": ("source", "fps"),
            "sampling": ("sampling", "policy"),
            "extraction": ("extraction", "validity_threshold"),
            r"frames\[0\]": ("frames", 0, "subject_id"),
            r"frames\[0\]\.joints\[0\]": ("frames", 0, "joints", 0, "valid"),
        }
        for path, keys in cases.items():
            with self.subTest(level=path):
                self.assertRejected(_delete(*keys), rf"^{path}: missing required fields")

    def test_unknown_fields_are_rejected_at_every_level(self) -> None:
        cases = {
            "record": (),
            "source": ("source",),
            "sampling": ("sampling",),
            "extraction": ("extraction",),
            r"frames\[0\]": ("frames", 0),
            r"frames\[0\]\.joints\[0\]": ("frames", 0, "joints", 0),
        }
        for path, keys in cases.items():
            with self.subTest(level=path):
                self.assertRejected(_set(*keys, "extra", value=1), rf"^{path}: unknown fields \['extra'\]")

    def test_nested_values_must_have_the_right_container_type(self) -> None:
        cases: dict[str, tuple[Mutation, str]] = {
            "source list": (_set("source", value=[]), "source: must be an object"),
            "frames object": (_set("frames", value={}), "frames: must be an array"),
            "frame list": (_set("frames", 0, value=[]), r"frames\[0\]: must be an object"),
            "joints object": (_set("frames", 0, "joints", value={}), r"joints: must be an array"),
            "joint list": (
                _set("frames", 0, "joints", 0, value=[0.5, 0.5, 0.9]),
                r"joints\[0\]: must be an object",
            ),
            "joint_names string": (_set("joint_names", value="nose"), "joint_names: must be an array"),
        }
        for name, (mutation, pattern) in cases.items():
            with self.subTest(case=name):
                self.assertRejected(mutation, pattern)


class HeaderTests(_RecordTestCase):
    def test_literals_must_match_exactly(self) -> None:
        cases: dict[str, tuple[Mutation, str]] = {
            "schema_version": (_set("schema_version", value="pose-pilot-v2"), "schema_version"),
            "coordinate_space": (
                _set("coordinate_space", value="detector_crop_pixels"),
                "coordinate_space",
            ),
            "timing": (_set("source", "timing", value="variable_fps"), r"source\.timing"),
            "policy": (_set("sampling", "policy", value="uniform"), r"sampling\.policy"),
            "partition": (_set("source", "partition", value="Train"), r"source\.partition"),
        }
        for name, (mutation, pattern) in cases.items():
            with self.subTest(field=name):
                self.assertRejected(mutation, pattern)

    def test_joint_names_must_match_canonical_order(self) -> None:
        swapped = list(CANONICAL_JOINT_NAMES)
        swapped[1], swapped[2] = swapped[2], swapped[1]
        for name, value in {
            "left/right swapped": swapped,
            "missing joint": list(CANONICAL_JOINT_NAMES[:-1]),
            "extra neck": [*CANONICAL_JOINT_NAMES, "neck"],
        }.items():
            with self.subTest(case=name):
                self.assertRejected(_set("joint_names", value=value), "joint_names: must exactly match")

    def test_identifiers_must_be_nonempty_strings(self) -> None:
        for keys in (
            ("source", "video_id"),
            ("source", "shot_id"),
            ("extraction", "extractor_id"),
            ("extraction", "subject_policy_revision"),
        ):
            for value in ("", 123, None):
                with self.subTest(field=keys, value=value):
                    self.assertRejected(_set(*keys, value=value), "must be a nonempty string")

    def test_hashes_must_be_lowercase_sha256(self) -> None:
        for keys in (
            ("source", "video_sha256"),
            ("source", "split_manifest_sha256"),
            ("extraction", "weights_sha256"),
        ):
            for value in ("A" * 64, "a" * 63, "a" * 65, "g" * 64, "a" * 64 + "\n", None):
                with self.subTest(field=keys, value=value):
                    self.assertRejected(_set(*keys, value=value), "SHA-256")

    def test_integer_fields_reject_other_types_and_small_values(self) -> None:
        for keys, minimum in (
            (("source", "width"), 2),
            (("source", "height"), 2),
            (("source", "frame_count"), 1),
            (("sampling", "num_frames"), 1),
        ):
            for value in (True, "832", 832.0, None, minimum - 1):
                with self.subTest(field=keys, value=value):
                    self.assertRejected(_set(*keys, value=value), r"must be (an integer|>=)")

    def test_fps_must_be_finite_and_positive(self) -> None:
        for value in (0, -16.0, float("nan"), float("inf"), True, "16"):
            with self.subTest(value=value):
                self.assertRejected(_set("source", "fps", value=value), r"source\.fps")

    def test_sampling_interval_must_be_ordered_and_inside_the_video(self) -> None:
        cases: dict[str, tuple[Mutation, str]] = {
            "negative start": (_set("sampling", "start_frame", value=-1), "start_frame: must be >= 0"),
            "empty interval": (_set("sampling", "end_frame", value=100), "greater than start_frame"),
            "reversed interval": (_set("sampling", "end_frame", value=99), "greater than start_frame"),
            "past frame_count": (_set("sampling", "end_frame", value=201), "frame_count 200"),
        }
        for name, (mutation, pattern) in cases.items():
            with self.subTest(case=name):
                self.assertRejected(mutation, pattern)

    def test_validity_threshold_must_be_a_finite_unit_number(self) -> None:
        for value in (-0.1, 1.5, float("nan"), float("-inf"), True, "0.5", None):
            with self.subTest(value=value):
                self.assertRejected(
                    _set("extraction", "validity_threshold", value=value),
                    r"extraction\.validity_threshold",
                )


class FrameTests(_RecordTestCase):
    def test_frames_must_be_nonempty_and_match_num_frames(self) -> None:
        self.assertRejected(
            lambda r: r.update(frames=[]) or r["sampling"].update(num_frames=1),
            "frames: must be nonempty",
        )
        self.assertRejected(
            _set("sampling", "num_frames", value=2),
            "frames: must contain sampling.num_frames=2 frames, got 1",
        )

    def test_source_frame_index_must_be_inside_the_interval(self) -> None:
        for value, pattern in (
            (99, r"inside the sampling interval \[100, 101\)"),
            (101, r"inside the sampling interval \[100, 101\)"),
            (True, "must be an integer"),
            (100.0, "must be an integer"),
        ):
            with self.subTest(value=value):
                self.assertRejected(_set("frames", 0, "source_frame_index", value=value), pattern)

    def test_source_frame_indices_must_be_nondecreasing(self) -> None:
        record = _record([_selected_frame(101), _selected_frame(100)], end_frame=102)
        with self.assertRaisesRegex(PoseRecordValidationError, "nondecreasing, got 100 after 101"):
            validate_pose_record(record)

    def test_repeated_indices_must_repeat_the_whole_observation(self) -> None:
        cases: dict[str, Mutation] = {
            "coordinate": _set("frames", 2, "joints", 5, "xy", 0, value=0.25),
            "confidence": _set("frames", 2, "joints", 5, "confidence", value=0.8),
            "status": lambda r: r["frames"].__setitem__(2, _unavailable_frame(101)),
        }
        for name, mutation in cases.items():
            with self.subTest(changed=name):
                self.assertRejected(
                    mutation,
                    r"frames\[2\]: repeats source_frame_index 101",
                    base=_three_frame_record,
                )

    def test_subject_status_and_subject_id_must_agree(self) -> None:
        cases: dict[str, tuple[Mutation, str]] = {
            "unknown status": (
                _set("frames", 0, "subject_status", value="occluded"),
                "subject_status: must be one of",
            ),
            "selected null id": (_set("frames", 0, "subject_id", value=None), "nonempty string"),
            "selected empty id": (_set("frames", 0, "subject_id", value=""), "nonempty string"),
            "selected numeric id": (_set("frames", 0, "subject_id", value=0), "nonempty string"),
        }
        for name, (mutation, pattern) in cases.items():
            with self.subTest(case=name):
                self.assertRejected(mutation, pattern)

        for status in ("absent", "ambiguous"):
            with self.subTest(status=status):
                record = _record([_unavailable_frame(status=status)])
                record["frames"][0]["subject_id"] = "subject-0"
                with self.assertRaisesRegex(PoseRecordValidationError, "subject_id: must be null"):
                    validate_pose_record(record)

    def test_selected_frames_must_share_one_subject_id(self) -> None:
        record = _record(
            [_selected_frame(100), _unavailable_frame(101), _selected_frame(102, "subject-1")],
            end_frame=103,
        )
        with self.assertRaisesRegex(PoseRecordValidationError, r"frames\[2\]\.subject_id: must match"):
            validate_pose_record(record)

    def test_unavailable_frames_must_hold_exactly_unavailable_joints(self) -> None:
        for status in ("absent", "ambiguous"):
            for field, value in (("xy", [0.5, 0.5]), ("confidence", 0.0), ("valid", True)):
                with self.subTest(status=status, field=field):
                    record = _record([_unavailable_frame(status=status)])
                    record["frames"][0]["joints"][3][field] = value
                    with self.assertRaisesRegex(
                        PoseRecordValidationError,
                        r"joints\[3\]: must be exactly .* when the subject is not selected",
                    ):
                        validate_pose_record(record)


class JointTests(_RecordTestCase):
    def test_joint_count_must_be_seventeen(self) -> None:
        for count in (0, 16, 18):
            with self.subTest(count=count):
                self.assertRejected(
                    lambda r, n=count: r["frames"][0].update(
                        joints=[_joint([0.5, 0.5], 0.9, True) for _ in range(n)]
                    ),
                    rf"must contain exactly 17 joints, got {count}",
                )

    def test_xy_must_be_two_finite_unit_numbers(self) -> None:
        cases: dict[str, tuple[Any, str]] = {
            "one value": ([0.5], "exactly 2 numbers, got 1"),
            "three values": ([0.5, 0.5, 0.5], "exactly 2 numbers, got 3"),
            "empty": ([], "exactly 2 numbers, got 0"),
            "object": ({"x": 0.5, "y": 0.5}, "must be an array"),
            "string": ("0.5,0.5", "must be an array"),
            "numeric string": (["0.5", 0.5], r"xy\[0\]: must be a number"),
            "boolean": ([True, 0.5], r"xy\[0\]: must be a number"),
            "null element": ([0.5, None], r"xy\[1\]: must be a number"),
            "nested": ([[0.5], 0.5], r"xy\[0\]: must be a number"),
            "negative": ([-0.01, 0.5], r"xy\[0\]: must be in"),
            "above one": ([0.5, 1.01], r"xy\[1\]: must be in"),
            "pixel units": ([416, 240], r"xy\[0\]: must be in"),
            "nan": ([float("nan"), 0.5], r"xy\[0\]: must be finite"),
            "infinity": ([0.5, float("inf")], r"xy\[1\]: must be finite"),
        }
        for name, (value, pattern) in cases.items():
            with self.subTest(case=name):
                self.assertRejected(_joint0("xy", value), pattern)

    def test_confidence_must_be_a_finite_unit_number_or_null(self) -> None:
        for value, pattern in (
            (1.01, "must be in"),
            (-0.1, "must be in"),
            (float("nan"), "must be finite"),
            (True, "must be a number"),
            ("0.9", "must be a number"),
            ([0.9], "must be a number"),
        ):
            with self.subTest(value=value):
                self.assertRejected(_joint0("confidence", value), rf"confidence: {pattern}")

    def test_valid_must_be_a_literal_boolean(self) -> None:
        for value in (1, 0, "true", None):
            with self.subTest(value=value):
                self.assertRejected(_joint0("valid", value), r"valid: must be a boolean")

    def test_valid_must_follow_the_threshold_rule(self) -> None:
        cases: dict[str, Mutation] = {
            "measured joint marked invalid": _joint0("valid", False),
            "low confidence marked valid": _joint0("confidence", 0.49),
            "missing xy marked valid": _joint0("xy", None),
            "null confidence marked valid": _joint0("confidence", None),
            "threshold raised above confidence": _set("extraction", "validity_threshold", value=0.95),
        }
        for name, mutation in cases.items():
            with self.subTest(case=name):
                self.assertRejected(mutation, r"joints\[0\]\.valid: must be (True|False)")


class LoadPoseRecordTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.directory = Path(self._temporary.name)

    def _write(self, content: str | bytes, name: str = "record.json") -> Path:
        path = self.directory / name
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")
        return path

    def test_returns_the_record_as_a_dictionary(self) -> None:
        record = _three_frame_record()
        path = self._write(json.dumps(record, indent=2))
        self.assertEqual(load_pose_record(path), record)
        self.assertEqual(load_pose_record(str(path)), record)

    def test_non_finite_json_literals_are_rejected(self) -> None:
        for literal in ("NaN", "Infinity", "-Infinity"):
            for field in ("confidence", "xy"):
                with self.subTest(literal=literal, field=field):
                    text = json.dumps(_record(), indent=2)
                    target = '"confidence": 0.9' if field == "confidence" else "0.0625"
                    replacement = (
                        f'"confidence": {literal}' if field == "confidence" else literal
                    )
                    path = self._write(text.replace(target, replacement, 1))
                    with self.assertRaisesRegex(
                        PoseRecordValidationError,
                        rf"non-finite JSON number {literal}",
                    ):
                        load_pose_record(path)

    def test_overflowing_numbers_are_rejected(self) -> None:
        text = json.dumps(_record()).replace('"fps": 16.0', '"fps": 1e400', 1)
        with self.assertRaisesRegex(PoseRecordValidationError, r"source\.fps: must be finite"):
            load_pose_record(self._write(text))

    def test_duplicate_keys_are_rejected(self) -> None:
        text = json.dumps(_record())
        cases = {
            "top level": text.replace(
                '"schema_version": "pose-pilot-v1"',
                '"schema_version": "pose-pilot-v0", "schema_version": "pose-pilot-v1"',
                1,
            ),
            "joint": text.replace('"valid": true', '"valid": false, "valid": true', 1),
        }
        for name, content in cases.items():
            with self.subTest(location=name):
                with self.assertRaisesRegex(PoseRecordValidationError, "duplicate JSON object key"):
                    load_pose_record(self._write(content))

    def test_file_and_json_errors_raise_validation_errors(self) -> None:
        cases: dict[str, tuple[Callable[[], Path], str]] = {
            "missing file": (lambda: self.directory / "missing.json", "cannot read file"),
            "directory": (lambda: self.directory, "cannot read file"),
            "non-utf8": (lambda: self._write(b"\xff\xfe{}"), "cannot read file"),
            "empty file": (lambda: self._write(""), "invalid JSON at line 1 column 1"),
            "truncated": (lambda: self._write('{"schema_version": '), "invalid JSON"),
            "trailing comma": (lambda: self._write('{"a": 1,}'), "invalid JSON"),
            "top-level array": (lambda: self._write("[]"), "record: must be an object"),
            "invalid record": (lambda: self._write("{}"), "record: missing required fields"),
        }
        for name, (make_path, pattern) in cases.items():
            with self.subTest(case=name):
                path = make_path()
                with self.assertRaisesRegex(PoseRecordValidationError, pattern) as raised:
                    load_pose_record(path)
                self.assertIn(str(path), str(raised.exception))

    def test_error_is_a_value_error(self) -> None:
        self.assertTrue(issubclass(PoseRecordValidationError, ValueError))


class CommandLineTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.directory = Path(self._temporary.name)

    def _run(self, argv: list[str]) -> tuple[int, str, str]:
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main(argv)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_valid_file_exits_zero(self) -> None:
        path = self.directory / "valid.json"
        path.write_text(json.dumps(_record()), encoding="utf-8")
        code, stdout, stderr = self._run([str(path)])
        self.assertEqual(code, 0)
        self.assertIn(f"OK: {path}", stdout)
        self.assertEqual(stderr, "")

    def test_invalid_file_exits_one_with_reason(self) -> None:
        record = _record()
        record["frames"][0]["joints"][0]["xy"] = [1.01, 1.0]
        path = self.directory / "invalid.json"
        path.write_text(json.dumps(record), encoding="utf-8")
        code, stdout, stderr = self._run([str(path)])
        self.assertEqual(code, 1)
        self.assertEqual(stdout, "")
        self.assertIn("INVALID:", stderr)
        self.assertIn("frames[0].joints[0].xy[0]", stderr)

    def test_missing_argument_is_a_usage_error(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                main([])
        self.assertEqual(raised.exception.code, 2)


class PackageTests(unittest.TestCase):
    def test_package_exports_the_public_api(self) -> None:
        import src.pose_models as package

        self.assertIs(package.validate_pose_record, validate_pose_record)
        self.assertIs(package.load_pose_record, load_pose_record)
        self.assertIs(package.PoseRecordValidationError, PoseRecordValidationError)
        with self.assertRaises(AttributeError):
            package.not_a_name  # noqa: B018

    def test_package_does_not_import_model_libraries(self) -> None:
        # A fresh interpreter, so modules imported by other tests do not count.
        script = (
            "import sys, src.pose_models, src.pose_models.record_validation; "
            "print(sorted({'torch', 'numpy', 'cv2'} & set(sys.modules)))"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(result.stdout.strip(), "[]")


if __name__ == "__main__":
    unittest.main()
