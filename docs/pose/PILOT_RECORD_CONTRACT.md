# Pilot pose record contract

Version: `pose-pilot-v1`. This P02 contract defines raw observations for the first pose extraction pilot and the P03 validator assignment. It is provisional until reviewed extraction examples establish P08's production format. Actual pilot videos and eligibility counts remain pending. The project lead authorizes this engineering baseline so CPU-only implementation can proceed alongside sample selection.

Raw joints are separate from RGB skeletons, heatmaps, model tensors and WAN tokens. A structurally valid record is not automatically eligible for training or evidence of correct extraction. No extractor, confidence threshold, renderer, encoder or model quality is selected by this contract.

## Joint and coordinate convention

The canonical joint order is the following 17-joint body convention:

```text
nose, left_eye, right_eye, left_ear, right_ear,
left_shoulder, right_shoulder, left_elbow, right_elbow,
left_wrist, right_wrist, left_hip, right_hip,
left_knee, right_knee, left_ankle, right_ankle
```

Left/right refer to the subject's anatomical side, not the viewer's side. Eye/ear/nose landmarks are body-estimator anchors; detailed face and hand keypoints are outside this version. Neck or other derived joints are not silently inserted. An extractor exposing a different convention must supply an explicit versioned mapping; unknown joints remain unavailable rather than being fabricated. This is a storage convention for comparisons, not a claim that all candidate extractors output this ordering directly.

Coordinates are `[x,y]` in the original decoded frame, normalized by `(width-1,height-1)`. The top-left pixel center is `[0,0]`; bottom-right is `[1,1]`. Both borders are valid. Frame dimensions must each be at least two. Undo detector crops, resize and letterbox transforms before storing coordinates. Renderer and tensor preparation apply their own declared transforms later. Do not swap axes or store detector-crop coordinates as full-frame coordinates. Do not clip out-of-frame detections into apparently valid border joints: store `xy: null` for an unavailable location and preserve any finite confidence.

## Record fields

All fields below are required. Reject unknown fields at the record, source, sampling, extraction, frame and joint levels for this version; changes require an explicit schema version. JSON objects have unordered keys. Arrays have ordered elements. Reject duplicate JSON object keys in the file entry point rather than silently retaining the last value.

| Field | Type and meaning |
|---|---|
| `schema_version` | Exactly `pose-pilot-v1` |
| `joint_names` | Array exactly matching the canonical joint order |
| `coordinate_space` | Exactly `original_frame_normalized_pixel_centers` |
| `source` | Object described below |
| `sampling` | Object described below |
| `extraction` | Object described below |
| `frames` | Nonempty ordered array of frame records; length equals `sampling.num_frames` |

`source` contains:

- `video_id`, `shot_id`: nonempty strings, preserving authoritative IDs as strings.
- `video_sha256`, `annotations_sha256`, `split_manifest_sha256`: 64-character lowercase hexadecimal SHA-256 strings. Fixture hashes identify synthetic inputs; real artifacts require real hashes.
- `partition`: `train`, `val` or `test`. The schema can represent test records, but the development pilot must select only train/val. Validation cannot establish membership without the authoritative manifest.
- `width`, `height`: integers ≥2; `frame_count`: positive integer; `fps`: finite positive number.
- `timing`: exactly `constant_fps`. This pilot assumes constant frame rate; source frame `i` has nominal time `i/fps`. Variable-frame-rate sources require a later explicit timestamp contract and must not be silently treated as constant-rate.

`sampling` contains:

- `start_frame`, `end_frame`: integer half-open interval satisfying `0 <= start_frame < end_frame <= source.frame_count`. Use the resolved/clamped interval from the shared helper.
- `num_frames`: positive integer, independent of the generation `4n+1` requirement.
- `policy`: exactly `shared_select_frame_indices_v1`. This names the behavior at audited commit `9ab622b4c25803f4397618c3117594a5d2bdf25a`: select evenly across `[start,end)` when sufficient frames exist; otherwise keep available frames and repeat the final frame. Record original frame indices, not positions in a preview or resampled video.

`extraction` contains:

- `extractor_id`, `extractor_revision`, `mapping_revision`, `preprocessing_revision`, `subject_policy_revision`: nonempty strings identifying the estimator/detector pipeline, pinned model/code identity, canonical mapping, crop/resize inversion and subject-selection policy. The wrapper must retain detailed settings in a separate hashed configuration.
- `weights_sha256`, `config_sha256`: 64-character lowercase SHA-256 strings. For multiple weight files, `weights_sha256` hashes a retained manifest of their filenames and hashes; for one file it may hash that file. Define the manifest convention with the wrapper. The validator checks syntax, not artifact contents.
- `validity_threshold`: finite number in `[0,1]`, chosen and recorded per extractor before comparison. Fixture value `0.5` is only a test parameter, not the chosen production threshold. Confidence calibration differs across estimators.

## Frame and joint fields

Each frame has exactly `source_frame_index`, `subject_status`, `subject_id`, `joints`.

`source_frame_index` is an integer inside the declared half-open interval. Indices are nondecreasing; repeated indices must repeat the entire frame observation identically. The validator checks range/order/repeat consistency without reimplementing NumPy's shared selection algorithm. Preparation and server review must separately compare the entire index list to `src/data/frame_sampling.py` output. A structurally valid arbitrary index list is not proof of correct sampling.

`subject_status` is one of:

| Status | Interpretation and subject ID |
|---|---|
| `selected` | One intended subject has been selected; `subject_id` is a nonempty clip-local string |
| `absent` | No usable intended subject detected; `subject_id` is null |
| `ambiguous` | A subject cannot be selected reliably, including unresolved competing detections; `subject_id` is null |

Multiple detections can still yield `selected` if the declared selection policy resolves them. Every selected frame in a record uses the same subject ID. This detects inconsistent labels, not undetected identity switches; temporal identity still needs visual review. Absent/ambiguous frames contain 17 unavailable joints and are retained as failure records, not rendered into a valid blank control.

`joints` is an array of exactly 17 objects, corresponding to `joint_names`. Each joint has exactly:

- `xy`: a two-number array of finite values in `[0,1]`, or null.
- `confidence`: a finite number in `[0,1]`, or null.
- `valid`: a literal boolean.

For selected frames, `valid` must equal `(xy is not null and confidence is not null and confidence >= validity_threshold)`. A measured low-confidence joint can retain its in-frame coordinates but is invalid. A location that is unavailable can retain a finite measured confidence but is invalid. Null confidence means unmeasured/unavailable confidence and implies invalidity. No interpolation, smoothing or inferred joints are encoded as observed valid joints in v1.

For absent/ambiguous frames, all joints are exactly `{"xy": null, "confidence": null, "valid": false}`. A selected frame with all joints invalid is structurally allowed: it records extraction failure and does not pass training eligibility automatically. NaN/Infinity, booleans used as numbers, numeric strings and malformed lists are rejected. Zero is a valid coordinate or confidence, not a missing-data sentinel.

Confidence can also be unavailable because an extractor's raw score has no agreed
conversion to this format. The explicit `unavailable_confidence_v1` builder policy
then stores null confidence for every selected joint and marks every joint invalid.
The retained request keeps the raw scores unchanged, and the hashed configuration
declares this policy. This is a pipeline test, not score calibration or a usable
training control. It does not establish anatomical accuracy or hide the original
extraction output. The v1 field sets and validity rule are unchanged.

## Examples and migration

[Synthetic examples](../../tests/fixtures/pose_records/README.md) include full, partial, absent, ambiguous and repeated-frame observations. They require no videos, server access, models or third-party Python packages. The fixtures are engineering inputs, not real annotations or pose accuracy evidence.

P03 validates these raw JSON records only. It does not change the current tensor loader, combined registry or checkpoint format. P08 may change joints, validity or representation after extraction demos; use a new schema version and explicit migration/regeneration rather than silently reinterpreting v1 records. Hash strings and declared metadata do not themselves prove provenance, membership, anatomical accuracy or eligibility.
