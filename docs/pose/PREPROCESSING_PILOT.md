# One-shot pose preprocessing pilot

Status: diagnostic data-path pilot completed on the server; training eligibility is not established.

## Scope and representation

Branch: `feat/pose-preprocessing`, based on organization `dev` after PR #18 merged.
The older `feat/pose-preprocessing-pilot` branch is retained for the historical server runs.
The annotated-shot run used implementation commit `4bc2f45`.

The pilot renders grouped Gaussian keypoint maps (head, torso, limbs) for all
person candidates. It retains every available in-frame position without a raw-score
threshold, confidence conversion, temporal tracking, interpolation, or smoothing.
These maps are not `pose-pilot-v1` records. They do not satisfy its selected-subject
and validity contract. Their `pose_encoded` key is a loader diagnostic interface;
the three channels are raw maps, not the legacy 256-channel learned pose features.
No trained pose adapter or end-to-end WAN forward pass was tested.
A packaged legacy encoder/adapter dimensional smoke command is now available;
its server execution remains pending until an actual report is supplied.

Source coordinates use an explicit pixel-centre resize transform. Spatial size is
128 × 128, matching the loader's direct resize for this check. That resize changes
aspect ratio for the 1280 × 720 source; it is not a final training resize policy.
Gaussian sigma is 2 output pixels. The three map channels use a maximum across
all points in each group, so individual joints and people cannot be recovered
from the maps alone. Raw predictions remain the audit source.

## Server verification reported on 2026-11-14

- Source: `/mnt/d1/snet/controllable-generation/veo-data/02.mp4`.
- Video ID `02`, shot ID `02_shot_0000`.
- Original metadata: `/mnt/d1/snet/jedidiah/data_depth_repro_docx/shots_metadata.json`.
- Frozen split: `/mnt/d1/snet/jedidiah/data_depth_repro/split_manifest.json`.
- Partition `train`; full annotations contain 524 shots, of which 314 are training shots.
- Selected source indices: `[0, 27, 54, 81, 109, 136, 163, 191]`.
- Candidate counts: `[8, 4, 8, 7, 2, 7, 5, 7]`.
- Three renderer/sampler tests passed.
- The real `ControllableVideoDataset` constructor discovered exactly one sample.
- `dataset[0]` loaded the correct video ID, shot ID and caption.
- Video and control shapes were each `[3, 8, 128, 128]`.
- Saved selected indices matched the shared frame sampler.

Server output directory:
`/home/jovyan/jedidiah/controllable-generation/artifacts/pose-preprocessing-pilot/02-annotated-shot-v1/`.

Artifacts include `manifest.json`, `raw_predictions.json`, and
`controls/02/shot_02_shot_0000_controls_encoded.npz`. The manifest stores actual
source, annotation, split and raw-prediction hashes, extractor model information,
code revision and rendering policy. Hash values were not supplied in the terminal
report, so this document does not claim independent verification of those values.

## Visual review

The user supplied `annotated-shot-overlay.png`, generated from the fully loaded
sample. Points broadly align with people across all eight inspected frames.
The main woman's head and upper-body estimates generally follow her position.
Overlapping people and cropped lower-body estimates remain uncertain.
This is a qualitative alignment check at 128 × 128, not a labelled landmark
accuracy benchmark or proof of temporal stability between sampled frames.

The source image is preserved locally under
`artifacts/pose-preprocessing-pilot/02-annotated-shot-v1/review/` with an evidence
manifest. Overlay SHA-256: `818799525e031171b3eba5a92cb01175bc773fd139661343c6f42988116bad6b`.

## Remaining work

Before using these outputs for training, establish the intended people policy,
visibility/score handling, spatial and temporal sampling, and the exact pose
encoder/adapter interface. The current eight-frame loader diagnostic is not a
WAN generation frame-count or model-training contract. Keep this evidence when
judging any later training results.

The preprocessing PR has not yet been opened. This pilot can be reviewed as a
diagnostic preprocessing contribution with the above limits stated explicitly.

## Agreed PR scope

Prepare one annotated-shot bundle reproducibly, verify spatial/frame/caption
alignment through the full dataset loader, and check one sample through the
existing pose encoder and generic adapter interface. Keep the PR in draft while
server compatibility checks are pending. The broader PR can be developed further
within that scope; full-dataset expansion and expert training are follow-up work.

The current diagnostic baseline explicitly includes all detected people and all
available in-frame joints. No new confidence threshold has been chosen from the
small example set. This supports the user's decision to retain DWPose as the
baseline and revisit extraction if later model results warrant it. It does not
change the selected-subject `pose-pilot-v1` contract or silently mark estimates
as verified observations.

## Verify the saved bundle and dimensional interface

Run from the new branch with the server's `controllable-gen` environment:

```bash
python -m scripts.pose.check_preprocessing_pilot \
  --bundle ~/jedidiah/controllable-generation/artifacts/pose-preprocessing-pilot/02-annotated-shot-v1 \
  --videos-dir /mnt/d1/snet/controllable-generation/veo-data \
  --annotations /mnt/d1/snet/jedidiah/data_depth_repro_docx/shots_metadata.json \
  --split-manifest /mnt/d1/snet/jedidiah/data_depth_repro/split_manifest.json \
  --report ~/jedidiah/controllable-generation/artifacts/pose-preprocessing-pilot/02-annotated-shot-v1/model-input-check.json
```

The command hashes the source artifacts, verifies frozen development membership,
checks the annotation interval and shared frame indices, and rebuilds the maps
from retained predictions. Old bundles without an NPZ byte hash are accepted only
when their numerical maps match the reconstruction (absolute and relative
tolerances 1e-6). New extractions also record the NPZ hash.

It constructs the full dataset, loads the correct shot/caption, checks input
shapes and ranges, then runs `PoseEncoder(out_channels=256)` followed by
`ControlAdapter(control_dim=256, hidden_dim=512, dit_dim=2048, num_controls=1)`.
The default device is CPU, using four PyTorch threads. No model weights are
updated. Initialization seed is 0; no checkpoint is loaded.

Expected tensor shapes for this existing bundle:

- Loaded control: `[1, 3, 8, 128, 128]`.
- Encoder features: `[1, 256, 8, 64, 64]`.
- Generic adapter output: `[1, 2048, 2048]` (2048 tokens, width 2048).

These are the legacy components' default interface dimensions. The smoke does
not choose a WAN checkpoint, validate its actual DiT width or injection hooks,
prove training gradient flow, measure generation quality, or establish final
production people/visibility/resize/sampling policies. A successful finite output
with random weights establishes dimensional compatibility only.

Local review: 80 pose tests and 43 DWPose tests passed (123 total). This includes
artifact tampering, stale indices, incorrect split, reconstruction, coordinate,
channel grouping and invalid Gaussian-width checks. Heavyweight torch/OpenCV
execution is pending on the server; it was not silently skipped and claimed as
passed locally.
