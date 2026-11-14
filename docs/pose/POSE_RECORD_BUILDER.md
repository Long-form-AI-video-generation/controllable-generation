# Minimal pose-record builder

The builder connects the merged frame wrapper to the existing `pose-pilot-v1`
record validator. It prepares a record from explicit frame selections, checks the
record, and saves JSON. It uses the shared frame sampler. It does not run a model
or establish extraction accuracy.

## Included components

This branch includes the frame wrapper merged through PR #17, the validator and
original commits from PR #16, the ONNX backend, and this builder. The validator
still enforces the agreed `pose-pilot-v1` schema. No second validator is used.
See [the server runbook](SERVER_RECORD_SMOKE.md) for the one-frame real-data check.

## Input and decisions

`build_pose_record(request)` takes a dictionary with exactly these keys:

- `source`: source metadata from the [record contract](PILOT_RECORD_CONTRACT.md).
- `sampling`: the resolved interval, requested frame count, and shared sampler policy.
- `extraction`: extraction metadata from the same contract.
- `score_policy`: explicitly `require_unit_interval_v1` or `unavailable_confidence_v1`.
- `observations`: one observation for each unique index returned by the shared sampler.

An observation contains `source_frame_index`, the wrapper's complete `frame_result`,
and a `selection` with exactly `subject_status`, `subject_id`, and `candidate_id`.

Example selection for a reviewed subject:

```json
{"subject_status": "selected", "subject_id": "subject-0", "candidate_id": "candidate_0"}
```

For absent or ambiguous subjects, both IDs must be null. The builder records 17
unavailable joints. It requires a selection even when only one candidate exists.
Candidate IDs belong to one frame; they are not track IDs. Reusing `subject-0`
asserts a reviewed identity and does not prove that the same character was tracked.
Use `subject_policy_revision: explicit_frame_selection_v1` for this builder.

The strict `require_unit_interval_v1` policy preserves a measured score only if it is finite and already in
`[0,1]`. Null remains null. An out-of-range score fails with an error; the builder
does not clip, apply a sigmoid, or treat the score as a calibrated probability.
This policy supports bounded scores without selecting a production DWPose
confidence conversion. The real frame-47 run returned four scores above 1, with
a maximum of about 1.052. Applying a sigmoid or clipping has not been justified.

The explicit `unavailable_confidence_v1` policy retains all raw scores in the
request, but stores null confidence for **all** selected joints in the final
record, including scores that happen to fall within [0,1]. Coordinates retain
their measured values; every joint is invalid. Malformed or non-finite raw scores
still fail. This supports a structural pipeline test only. It does not produce
training-ready controls or establish that the joint locations are correct.

## What happens to the observations

1. Check metadata with the existing validator. Accept train/val for this development
   pilot; final held-out test processing is outside this builder's current scope.
2. Compute the exact source indices with `src/data/frame_sampling.py`. Reject an
   unresolved interval or observations that do not match the sampled indices.
3. Apply the explicit selection for each unique source frame.
4. Divide original-image x by `width - 1`, and y by `height - 1`. Keep missing points
   null. Keep low-confidence coordinates available but mark them invalid.
5. With the strict policy, set `valid` only when coordinates and confidence exist
   and confidence meets the recorded threshold, including equality. With the
   unavailable-confidence policy, every joint has null confidence and is invalid.
6. Repeat the complete observation when the sampler repeats an index. Do not rerun
   extraction for a repeated frame.
7. Validate the assembled record. Validate again before writing. Refuse to overwrite
   an existing output file.

All frame results must agree on image dimensions and model metadata. Model failures
must raise an error, rather than becoming absent-subject observations. The caller
retains responsibility for accurate video metadata, constant frame rate, actual
artifact hashes, frozen-split membership, and visual identity review.

Keep the input request as an audit sidecar. It retains raw scores, invalid reasons,
model information, and explicit candidate selections that do not fit inside the
agreed record schema. Keep the extraction configuration artifact too: it must
describe the score policy, subject policy, threshold, and model/preprocessing
settings. Supply its actual SHA-256 as `extraction.config_sha256`. The builder
checks hash syntax through the validator; it cannot verify an external artifact
that was not supplied. Do not substitute placeholder hashes for real artifacts.

## Run the synthetic check

Run from the repository root with
Python 3.10+ and NumPy. No ONNX Runtime, OpenCV, weights, or video files are needed.

```bash
python scripts/pose/demo_record_builder.py /tmp/pose-builder-demo
python -m src.pose_models.record_validation /tmp/pose-builder-demo/synthetic_record.json
```

The demo uses the actual frame wrapper with stand-in detector/estimator outputs.
It writes a retained request and a validated five-frame record. Indices are
`10, 11, 12, 12, 12`; statuses are selected, absent, ambiguous, ambiguous, ambiguous.
Its placeholder hashes, model information, and threshold are synthetic test data.
Use a new output directory each time.

Rebuild from a retained request:

```bash
python -m src.pose_models.record_builder /tmp/pose-builder-demo/synthetic_request.json /tmp/rebuilt-pose-record.json
```

The command rejects malformed JSON, duplicate object keys, and non-finite JSON
constants. It exits with code 0 on success and code 1 on input or file errors.

Run the focused tests:

```bash
python -m unittest discover -s tests -p 'test_pose_record_builder.py' -v
```

The server command builds from a retained real smoke-test result with an explicit
candidate selection and visual-review note. It hashes the current video,
annotations, split, weight manifest, and configuration, verifies development-split
membership and model hashes, and uses ffprobe frame timestamps to verify constant
frame rate. It refuses an existing output directory. Legacy smoke results have
no inference-time video hash: a matching source path is verified, but the video
being unchanged since extraction remains an explicit assumption in the config.
This does not establish temporal identity or training eligibility.
