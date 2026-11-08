# Synthetic pose extraction case review

Date: 2026-11-08. Scope: four generated anime images used to check extraction
behavior. The project lead ran the server commands and supplied the terminal
results and overlay screenshots. This is a static-image software and visual
check, not training-data preparation, a dataset accuracy benchmark, or a test of
video consistency. It replaces the previously proposed three-clip pilot for
this bounded case-handling task; the video pilot remains separate future work.

## Inputs and execution

Generated images, originals, prompts, JPEG hashes and case descriptions are
retained locally under `artifacts/pose-case-inputs/2026-11-08/`.
The files uploaded to the server are `full-body.jpg`, `partial-body.jpg`,
`multiple-people.jpg`, and `no-person.jpg`. Expected scene contents were specified
before extraction; no manual joint-coordinate ground truth was supplied.

Server input directory:
`/home/jovyan/jedidiah/controllable-generation/artifacts/pose-case-inputs/`.
Server output directory:
`/home/jovyan/jedidiah/controllable-generation/artifacts/pose-case-results/2026-11-08/`.
Each case has `frame_result.json`, `pose_overlay.png`, and `run.log`. The parent
directory also retains `code-commit.txt` and `input-sha256.txt`. Their contents
have not been supplied in the conversation; use those files for the actual
server code pin and input-byte verification rather than guessing a revision.

Reported setup: CPUExecutionProvider, ONNX Runtime 1.23.2 and OpenCV 4.13.0,
using the pinned DWPose detector and pose weights. The script checks those pins.
Reported extraction configuration SHA-256:
`77db7f47e737c86a087c19efb05d07ccd031a48fb74bae9cba6e11b7f5717e70`.

## Findings

| Case | Reported result | Visual review | Conclusion for this input |
|---|---|---|---|
| Full body | `completed`, one candidate; 17 positions available | `candidate_0` encloses the intended person. Shoulders, arms, hips, knees and ankles appear broadly aligned in the supplied screenshot. Fine landmark accuracy is not measured. | The extractor returns a plausible body pose for this clear example. |
| Partial body | `no_detections`, zero candidates | The source visibly contains a person whose lower body is hidden by a counter. No box or skeleton was drawn. | Detector-level miss. The empty result is handled without a crash, but partially visible people are not established as supported. |
| Multiple people | `completed`, three candidates; 17 positions each | Three separate skeletons broadly follow the three people. Main foreground woman in blue is `candidate_1`; red-shirt person is `candidate_0`, green-shirt person is `candidate_2`. | Multiple detections are retained and the intended person can be explicitly selected in this image. Automatic subject selection or tracking was not tested. |
| No person | `no_detections`, zero candidates | Empty furnished room; no boxes or skeletons. | Expected negative result for this example; no false positive was returned. |

Raw detector scores, distinct from per-joint confidence:

- Full body: candidate_0 = 0.8489909172058105.
- Multiple people: candidate_0 = 0.9505822658538818,
  candidate_1 = 0.9424570202827454, candidate_2 = 0.9128261208534241.

The highest detector score belongs to the red-shirt person, not our intended
blue-shirt subject. Choosing the first or highest-scoring candidate would select
the wrong person for this task. Candidate IDs belong to one image only.

## Completion and limits

All four extraction calls completed without crashing and wrote their results.
The supplied overlays have been reviewed. No image-case records were passed
through the video-record builder or validator during this run; these inputs are
static images and must not be given fabricated video provenance.

The partial-body example returned no detector boxes, so the pose estimator was
not exercised for that person. This result does not establish how the estimator
or record builder handles hidden joints. Separating detector failure from pose
estimation would require a deliberate follow-up, such as a reviewed person box;
do not assume such a run happened or silently add a full-image fallback.

Joint availability means returned in-frame coordinates, not verified anatomical
visibility or accuracy. Prior unresolved raw-score conversion and training
eligibility limitations still apply. No new pose training, calibration, temporal
tracking, or dataset-wide performance claim follows from these four examples.

This completes the bounded extraction-case run and qualitative review. Preserve
the detector miss as a finding rather than treating every case as a quality pass.

## Follow-up detector diagnosis

The user clarified that the goal is automatic preprocessing that can handle these
cases later. A manual person box would be diagnostic only, not the deployed
solution. A server comparison reused the pinned detector weights and image files,
ran RGB and BGR detector inputs, and inspected retained detection counts above
0.3, 0.1 and 0.05. The low proposal floor for this diagnostic was 0.01. Pose
estimation was not run in the comparison.

| Image | RGB count above 0.3 | BGR count above 0.3 | Highest RGB score | Highest BGR score |
|---|---:|---:|---:|---:|
| Full body | 1 | 1 | 0.848991 | 0.936208 |
| Partial body | 0 | 1 | 0.034433 | 0.916218 |
| Multiple people | 3 | 3 | 0.950582 | 0.960161 |
| No person | 0 | 0 | No retained proposal above 0.01 | No retained proposal above 0.01 |

The counts in each column also held at 0.1 and 0.05. The comparison supports a
detector color-order correction rather than lowering thresholds to fit this one
positive. It does not establish detection accuracy from scores alone: inspect
the saved BGR boxes and the corrected full-pipeline overlay before accepting the
partial-body result. The no-person result is one negative case, not a measured
false-positive rate.

Server diagnostic directory:
`/home/jovyan/jedidiah/controllable-generation/artifacts/pose-detector-diagnostics/2026-11-08/`.
It retains `report.json` with the code pin, model metadata, input hashes and
low-floor detections, plus box overlays for each order/threshold combination.

The correction converts RGB source images to BGR internally for YOLOX and leaves
the source array available to the pose estimator unchanged. It versions and
records the detector color policy in the configuration manifest. Production
thresholds stay 0.1 for proposals and 0.3 for accepted people. Full-pipeline
regression on the same four cases, partial-body pose inspection, and any further
pose-color/hidden-joint investigation are pending; do not report them as complete.
