# One real pose-record pipeline check

Goal: turn the reviewed main subject in video 02, frame 47, into one structurally
valid `pose-pilot-v1` record with retained raw results and real artifact hashes.
This is a one-frame integration check. It does not complete an extraction pilot,
calibrate confidence, validate temporal tracking, or produce a training control.

## What is established

- The DWPose backend and wrapper passed 42 tests on the server.
- After fixing detector-box boundaries, frames 47 and 191 completed on CPU.
- Human review selected candidate_0 in both of those individual frames. Candidate
  IDs are frame-local and must not be reused as tracking IDs.
- Several leg points are inaccurate or attached to clothing. The source is
  cropped, and all 17 returned points are not proof of 17 correct visible joints.
- Frame 47 returned raw joint scores from about 0.217 to 1.052; four exceeded 1.
  No probability calibration or bounded conversion has been established.

For this check, `unavailable_confidence_v1` preserves raw scores in the audit
request and stores null confidence for every joint in the record. All valid
flags are false. The fixed threshold 0.5 is a required schema placeholder only;
it has no effect when confidence is null. A production conversion, quality
criteria, and treatment of visually wrong joints remain future decisions.

## Server checkout

Use a persistent, detached review worktree. The earlier `/tmp` worktree disappeared
and the server's local `dev` was accidentally fast-forwarded to d88bf1b. Do not
reset that checkout or push its `dev` while preparing this check. The new worktree
does not repair the old branch; that is a separate, deliberate cleanup task.

Run in the server terminal. The parenthesized block stops if any setup or test
command fails, without exiting the interactive terminal.

```bash
(
  cd ~/jedidiah/controllable-generation || exit 1
  conda activate controllable-gen || exit 1
  git fetch origin feat/pose-record-pipeline || exit 1
  git worktree add --detach ~/pose-record-pipeline-review origin/feat/pose-record-pipeline || exit 1
  cd ~/pose-record-pipeline-review || exit 1
  python -m unittest discover -s tests -p 'test_pose_*.py' || exit 1
  python -m unittest discover -s tests -p 'test_dwpose_*.py' || exit 1
  command -v ffprobe || exit 1
)
```

If `~/pose-record-pipeline-review` already exists, inspect it before reusing it;
do not delete it or rerun worktree creation blindly. The builder itself needs
Python, NumPy, and the `ffprobe` executable. It does not import OpenCV or run ONNX.
If ffprobe is unavailable, install it in the server environment before proceeding;
do not assert constant frame rate from the reported average FPS alone.

## Build from the existing result

The reviewed frame was extracted with commit d88bf1b. Resolve its full SHA for
`extractor_revision`; the current builder commit is a different piece of code.
The command verifies the source path and current file hashes. Because the legacy
smoke output has no video hash from inference time, unchanged source bytes since
that run are an explicit assumption, retained in the hashed configuration.

```bash
(
  cd ~/pose-record-pipeline-review || exit 1
  conda activate controllable-gen || exit 1
  python scripts/pose/build_reviewed_record.py \
    --smoke-result ~/jedidiah/controllable-generation/artifacts/pose-smoke-review/video-02-frame-47_result.json \
    --video /mnt/d1/snet/controllable-generation/veo-data/02.mp4 \
    --video-id 02 \
    --shot-id 02_shot_0000 \
    --annotations /mnt/d1/snet/jedidiah/data_depth_repro_docx/shots_metadata.json \
    --split-manifest /mnt/d1/snet/jedidiah/data_depth_repro/split_manifest.json \
    --detector-model /mnt/d1/snet/jedidiah/models/DWPose/yolox_l.onnx \
    --pose-model /mnt/d1/snet/jedidiah/models/DWPose/dw-ll_ucoco_384.onnx \
    --extractor-revision "$(git rev-parse d88bf1b)" \
    --candidate-id candidate_0 \
    --review-note 'Main foreground woman selected in frame 47. Leg/ankle points are inaccurate; source is cropped. Pipeline test only.' \
    --output-dir ~/jedidiah/controllable-generation/artifacts/pose-record-02-frame-47 || exit 1

  python -m src.pose_models.record_validation \
    ~/jedidiah/controllable-generation/artifacts/pose-record-02-frame-47/record.json || exit 1
)
```

The output directory must be new. It contains `record.json`, `request.json`,
`smoke_result.json`, `weights_manifest.json`, and `extraction_config.json`.
The manifests are hashed as their exact UTF-8 JSON bytes, including the final
newline. The request preserves raw scores, all candidate results, and the explicit
subject choice. The smoke result retains the original source metadata.

Success means the CLI and independent validator both exit 0, the expected frame
and candidate appear in the request, the audit files match their record hashes,
and the configuration states `training_eligible: false`. The unit tests cover
these boundaries with stand-in data. Actual execution of this command on the
server, including real ffprobe timing verification, remains required before
reporting a completed real record.

## Contributions and remaining work

The validator was merged through PR #16; the frame wrapper was merged in PR #17.
The integration branch includes the latest dev containing both contributions.
The integration PR adds the ONNX backend, boundary fix,
builder, retained fixtures, explicit unavailable-confidence policy, and this
one-frame audit workflow. It builds on the merged validator PR.
Review and merge the integration through the organization's `dev` branch.

After this pipeline check, resolve confidence representation and visual quality
handling before preparing training controls. Validate identity and pose quality
over multiple frames and clips. Repair the server's local `dev` separately after
preserving its current commits and generated artifacts.
