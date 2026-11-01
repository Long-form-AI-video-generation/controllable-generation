"""Assemble reviewed frame-wrapper results into validated pose-pilot-v1 records.

The record validator is included in this pipeline. This module does not select a subject,
calibrate scores, decode video, or run a model. Keep the input request as the audit
sidecar: it retains raw scores, diagnostics, and explicit per-frame selections.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import sys

from src.data.frame_sampling import resolve_frame_interval, select_frame_indices
from src.pose_models.extraction.dwpose import CANONICAL_JOINT_NAMES


class PoseRecordBuildError(ValueError):
    """A build request or frame observation cannot be converted safely."""


SCORE_POLICY = "require_unit_interval_v1"
UNAVAILABLE_SCORE_POLICY = "unavailable_confidence_v1"
SCORE_POLICIES = (SCORE_POLICY, UNAVAILABLE_SCORE_POLICY)
SUBJECT_POLICY = "explicit_frame_selection_v1"


def _object(value, field, keys=None):
    if not isinstance(value, dict):
        raise PoseRecordBuildError(f"{field}: must be an object")
    if keys is not None and set(value) != set(keys):
        raise PoseRecordBuildError(f"{field}: expected fields {sorted(keys)}")
    return value


def _integer(value, field, minimum=0):
    if type(value) is not int or value < minimum:
        raise PoseRecordBuildError(f"{field}: must be an integer >= {minimum}")
    return value


def _number(value, field, low, high):
    if type(value) not in (int, float):
        raise PoseRecordBuildError(f"{field}: must be a finite number in [{low}, {high}]")
    try:
        valid = math.isfinite(value) and low <= value <= high
    except OverflowError:
        valid = False
    if not valid:
        raise PoseRecordBuildError(f"{field}: must be a finite number in [{low}, {high}]")
    return value


def _validator():
    try:
        from src.pose_models.record_validation import validate_pose_record
    except ModuleNotFoundError as error:
        if error.name != "src.pose_models.record_validation":
            raise
        raise PoseRecordBuildError(
            "record validator unavailable: install the complete pose-record pipeline checkout"
        ) from error
    return validate_pose_record


def frame_observation(source_frame_index, frame_result, selection, *, width, height,
                      validity_threshold, score_policy):
    """Convert one explicit selection; do not infer identity from candidate order."""
    _integer(source_frame_index, "source_frame_index")
    _integer(width, "width", 2)
    _integer(height, "height", 2)
    _number(validity_threshold, "validity_threshold", 0, 1)
    if score_policy not in SCORE_POLICIES:
        raise PoseRecordBuildError(f"score_policy: must explicitly be one of {SCORE_POLICIES}")
    result = _object(frame_result, "frame_result")
    for name, expected in (("width", width), ("height", height)):
        _integer(result.get(name), f"frame_result.{name}", 2)
        if result[name] != expected:
            raise PoseRecordBuildError(f"frame_result.{name}: differs from source")
    candidates = result.get("candidates")
    if not isinstance(candidates, list):
        raise PoseRecordBuildError("frame_result.candidates: must be a list")
    expected_status = "completed" if candidates else "no_detections"
    if result.get("status") != expected_status:
        raise PoseRecordBuildError("frame_result.status: inconsistent with candidates")
    by_id = {}
    for candidate in candidates:
        candidate = _object(candidate, "candidate")
        identifier = candidate.get("candidate_id")
        if not isinstance(identifier, str) or not identifier.strip() or identifier in by_id:
            raise PoseRecordBuildError("candidate_id: must be nonempty and unique per frame")
        by_id[identifier] = candidate

    selection = _object(selection, "selection", ("subject_status", "subject_id", "candidate_id"))
    status = selection["subject_status"]
    if status not in ("selected", "absent", "ambiguous"):
        raise PoseRecordBuildError("selection.subject_status: unsupported status")
    joints = [{"xy": None, "confidence": None, "valid": False} for _ in range(17)]
    if status != "selected":
        if selection["subject_id"] is not None or selection["candidate_id"] is not None:
            raise PoseRecordBuildError("selection: absent/ambiguous requires null IDs")
    else:
        subject_id = selection["subject_id"]
        if not isinstance(subject_id, str) or not subject_id.strip():
            raise PoseRecordBuildError("selection.subject_id: must be a nonempty string")
        candidate_id = selection["candidate_id"]
        if not isinstance(candidate_id, str) or candidate_id not in by_id:
            raise PoseRecordBuildError("selection.candidate_id: not found in this frame")
        candidate = by_id[candidate_id]
        if candidate.get("joint_names") != CANONICAL_JOINT_NAMES:
            raise PoseRecordBuildError("candidate.joint_names: must use canonical COCO-17 order")
        for key in ("xy_pixels", "joint_scores_raw", "invalid_reasons"):
            if not isinstance(candidate.get(key), list) or len(candidate[key]) != 17:
                raise PoseRecordBuildError(f"candidate.{key}: must contain 17 entries")
        for index, (point, score, reason) in enumerate(zip(
            candidate["xy_pixels"], candidate["joint_scores_raw"], candidate["invalid_reasons"]
        )):
            field = f"candidate.joints[{index}]"
            if point is None:
                if reason not in ("model_missing", "outside_frame"):
                    raise PoseRecordBuildError(f"{field}: missing point requires an invalid reason")
                xy = None
            else:
                if not isinstance(point, list) or len(point) != 2 or reason is not None:
                    raise PoseRecordBuildError(f"{field}: expected in-frame [x, y] with null reason")
                x = _number(point[0], f"{field}.x", 0, width - 1)
                y = _number(point[1], f"{field}.y", 0, height - 1)
                xy = [x / (width - 1), y / (height - 1)]
            if score is not None:
                if score_policy == SCORE_POLICY:
                    _number(score, f"{field}.score_raw (no clipping or calibration)", 0, 1)
                else:
                    try:
                        finite_score = type(score) in (int, float) and math.isfinite(score)
                    except OverflowError:
                        finite_score = False
                    if not finite_score:
                        raise PoseRecordBuildError(f"{field}.score_raw: must be finite or null")
            confidence = score if score_policy == SCORE_POLICY else None
            joints[index] = {
                "xy": xy,
                "confidence": confidence,
                "valid": xy is not None and confidence is not None and confidence >= validity_threshold,
            }
    return {"source_frame_index": source_frame_index, "subject_status": status,
            "subject_id": selection["subject_id"], "joints": joints}


def build_pose_record(request):
    """Build from metadata and one observation per unique shared-sampler index.

    Metadata and hashes must describe the retained source/configuration artifacts.
    Caller-reviewed selections establish identity; candidate IDs alone do not.
    """
    validate = _validator()
    request = _object(request, "request", (
        "source", "sampling", "extraction", "score_policy", "observations"
    ))
    source = _object(request["source"], "source")
    sampling = _object(request["sampling"], "sampling")
    extraction = _object(request["extraction"], "extraction")
    if request["score_policy"] not in SCORE_POLICIES:
        raise PoseRecordBuildError(f"score_policy: must explicitly be one of {SCORE_POLICIES}")
    if extraction.get("subject_policy_revision") != SUBJECT_POLICY:
        raise PoseRecordBuildError(f"subject_policy_revision: must be '{SUBJECT_POLICY}'")
    if source.get("partition") not in ("train", "val"):
        raise PoseRecordBuildError("source.partition: development pilot accepts train/val only")
    count = _integer(source.get("frame_count"), "source.frame_count", 1)
    start = _integer(sampling.get("start_frame"), "sampling.start_frame")
    end = _integer(sampling.get("end_frame"), "sampling.end_frame", 1)
    num_frames = _integer(sampling.get("num_frames"), "sampling.num_frames", 1)
    try:
        resolved = resolve_frame_interval(start, end, count)
        if resolved != (start, end):
            raise ValueError("supply the resolved interval; end_frame exceeds source.frame_count")
        indices = [int(i) for i in select_frame_indices(start, end, num_frames)]
    except ValueError as error:
        raise PoseRecordBuildError(f"sampling: {error}") from error
    record = {
        "schema_version": "pose-pilot-v1",
        "joint_names": list(CANONICAL_JOINT_NAMES),
        "coordinate_space": "original_frame_normalized_pixel_centers",
        "source": deepcopy(source), "sampling": deepcopy(sampling),
        "extraction": deepcopy(extraction),
        "frames": [{"source_frame_index": i, "subject_status": "absent", "subject_id": None,
                    "joints": [{"xy": None, "confidence": None, "valid": False}
                               for _ in range(17)]} for i in indices],
    }
    # Check metadata before using dimensions, thresholds, or declaring output valid.
    validate(record)
    observations = request["observations"]
    if not isinstance(observations, list):
        raise PoseRecordBuildError("observations: must be a list")
    by_index = {}
    model_info = None
    for observation in observations:
        observation = _object(observation, "observation", (
            "source_frame_index", "frame_result", "selection"
        ))
        index = _integer(observation["source_frame_index"], "observation.source_frame_index")
        if index in by_index:
            raise PoseRecordBuildError(f"observations: duplicate source index {index}")
        result = _object(observation["frame_result"], f"observations[{index}].frame_result")
        info = _object(result.get("model_info"), f"observations[{index}].model_info")
        if model_info is not None and info != model_info:
            raise PoseRecordBuildError("observations: model_info differs between frames")
        model_info = deepcopy(info)
        by_index[index] = frame_observation(
            index, result, observation["selection"], width=source["width"], height=source["height"],
            validity_threshold=extraction["validity_threshold"], score_policy=request["score_policy"],
        )
    if set(by_index) != set(indices):
        raise PoseRecordBuildError("observations: indices must exactly match unique shared-sampler indices")
    # A repeated sampled index reuses the complete observation, never another model call.
    record["frames"] = [deepcopy(by_index[i]) for i in indices]
    validate(record)
    return record


def save_pose_record(record, path):
    """Validate before writing; refuse to overwrite an existing file."""
    _validator()(record)
    payload = json.dumps(record, indent=2, allow_nan=False) + "\n"
    with Path(path).open("x", encoding="utf-8") as handle:
        handle.write(payload)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise PoseRecordBuildError(f"JSON: duplicate key '{key}'")
        result[key] = value
    return result


def _reject_constant(value):
    raise PoseRecordBuildError(f"JSON: non-finite number {value}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path, help="retained build-request JSON")
    parser.add_argument("output", type=Path, help="new validated record JSON")
    args = parser.parse_args(argv)
    try:
        request = json.loads(args.request.read_text(encoding="utf-8"),
                             object_pairs_hook=_unique_object, parse_constant=_reject_constant)
        record = build_pose_record(request)
        save_pose_record(record, args.output)
    except (ValueError, OSError) as error:
        print(f"Pose record build failed: {error}", file=sys.stderr)
        return 1
    print(f"Saved validated pose record: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
