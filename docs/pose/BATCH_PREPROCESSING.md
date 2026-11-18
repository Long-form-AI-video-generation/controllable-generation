# Bounded diagnostic pose batches

This runner extends the annotated-shot pilot to an explicit list of training and
validation shots. It produces **diagnostic raw maps**, not a training-approved
pose dataset. It does not train, run WAN or establish pose accuracy.

## Configure a small run

Save an actual configuration under an ignored local output area, for example
`artifacts/pose-batch/config.json`. Keep real paths, captions and run evidence out
of commits. Relative paths below resolve against the configuration file's directory.
All six input/output paths are required. Replace the example paths and shot IDs
with real development data before running.

```json
{
  "videos_dir": "/path/to/videos",
  "annotations": "/path/to/shots_metadata.json",
  "split_manifest": "/path/to/frozen_split_manifest.json",
  "output_dir": "/path/to/ignored/artifacts/pose-batch/run-01",
  "detector_model": "/path/to/DWPose/yolox_l.onnx",
  "pose_model": "/path/to/DWPose/dw-ll_ucoco_384.onnx",
  "detector_sha256": "7860ae79de6c89a3c1eb72ae9a2756c0ccfbe04b7791bb5880afabd97855a411",
  "pose_sha256": "724f4ff2439ed61afb86fb8a1951ec39c6220682803b4a8bd4f598cd913b1843",
  "onnxruntime_version": "1.23.2",
  "opencv_version": "4.13.0",
  "num_frames": 8,
  "size": 128,
  "sigma": 2.0,
  "max_shots": 4,
  "shots": [
    {"video_id": "02", "shot_id": "02_shot_0000"}
  ]
}
```

The model hashes identify the existing pinned DWPose files. Versions must match
your installed, verified environment; the example uses the pilot baseline.
`max_shots` is a required explicit bound, not automatic dataset selection. Add
chosen train/validation IDs to `shots`; duplicate requests and test shots are
rejected. The frozen split is mandatory and checked for overlapping video IDs.
Each selected annotation must have a nonempty narrative or descriptive caption;
the batch does not accept the loader's generic fallback as a meaningful caption.

## Run and resume

Run from the source checkout, with the established environment active. On a server,
use that server's actual checkout location in `cd`. The new code must already be
available there. The following uses a generic checkout placeholder:

```bash
cd /path/to/controllable-generation
conda activate controllable-gen
python -m scripts.pose.prepare_batch --config artifacts/pose-batch/config.json --dry-run
python -m scripts.pose.prepare_batch --config artifacts/pose-batch/config.json
python -m scripts.pose.prepare_batch --config artifacts/pose-batch/config.json --resume
```

Dry-run validates configuration, metadata, shot membership, captions, input presence
and source-video/model-file hashes without decoding videos, initializing ONNX or creating output.
It prints the selected plan. It is not a runtime compatibility or pose-quality check.

Real execution initializes DWPose once and processes shots sequentially on CPU.
The CLI exits nonzero if any shot fails. Read the printed summary and retained
attempt receipts; a failed inference is never turned into an absent-person success.

A normal run requires clean tracked source and no untracked Python implementation
files under `src/` or `scripts/`. The source checkout is resolved from the script,
not the shell's current directory. `--allow-dirty-code` is a development option:
it records the commit plus a digest of changes; dirty results cannot later be
resumed with different code, including a newly committed version. Use a fresh root.
Untracked notes and installed skills do not make source dirty.

## Output and recovery contract

```text
run-01/
  batch.json                         immutable settings, all source hashes, code identity
  bundles/<video_id>/<shot_id>/
    manifest.json                    diagnostic contract and batch fingerprint
    raw_predictions.json
    controls/<video_id>/shot_<shot_id>_controls_encoded.npz
  attempts/
    <run_id>-0001.json                immutable per-shot receipt
    <run_id>-summary.json             counts and timings for this invocation
```

Maps remain float32 `[3,T,H,W]` under `pose_encoded`, with head/torso/limb channels,
all detected people, no joint-score visibility filtering and direct pixel-centre
resize. Every manifest keeps `training_eligible: false`. Captions remain in the
original annotations and the per-shot receipts. Raw predictions preserve candidates.

- A bundle is written into a sibling `.<shot_id>.incomplete-*` directory. Saved
  maps are rebuilt from the retained predictions and source hashes are checked
  before an atomic directory rename exposes the completed shot.
- A caught failure adds `failure.json` to its incomplete directory. SIGKILL cannot
  write a final failure record; its incomplete directory remains identifiable.
  Old failed attempts are preserved when a later attempt succeeds.
- The immutable batch record is also initialized through a temporary directory.
  JSON receipts are published atomically without replacing an existing file.
- Every distinct selected video is hashed into the original batch identity, including
  failed and not-yet-started shots. Changed source bytes reject resume; within a run,
  each shot checks its original hash before extraction and again before publication.
- Resume verifies source hashes, retained predictions, control bytes, frame
  indices, shot identity, map shape and the batch fingerprint before skipping.
  It reinitializes no extractor when every selected shot is already verified.
- Missing completed bundles are retried; corrupt or incompatible completed bundles
  are reported as failures and never overwritten. There is deliberately no force
  overwrite flag. Preserve the old run and use a fresh output root for changed
  settings, changed inputs/code, or intentional reprocessing of corrupt finals.
- Process locks prevent cooperating writers from running the same batch or writing
  the same shot simultaneously. Locks release when their process exits. File locks,
  hard links and atomic same-filesystem directory renames require a filesystem
  supporting those operations. This does not promise power-loss durability.
- Summary counts describe **this invocation**: completed, verified/skipped, failed
  and unprocessed. A graceful interrupt also records its interrupted shot. Receipts
  retain caption, partition, frame indices, candidate counts, elapsed time and
  successful output bytes. Earlier invocations remain in `attempts/`.

The runner's bundles deliberately remain separate. To use the full loader and
random encoder/adapter diagnostic, run the existing checker on each bundle with
its original annotation and split files and a fresh report path. That needs the
real video, OpenCV and PyTorch; the batch's artifact verification alone does not
establish full loader or model compatibility for the batch.

## First server review

Before expanding the selected list, verify accounting and inspect aligned source
frames/pose overlays for representative completed shots, especially no-person,
multiple-person and cropped-person cases when available on development data.
Overlays are not generated by this runner. Use the retained predictions/maps for
that review; candidate counts or successful shapes do not measure pose accuracy.
Check actual caption quality, runtime and storage before a larger run. Keep
production resize, identity/visibility policy, WAN temporal/interface requirements
and training eligibility as separate decisions.

## Focused local checks

```bash
python -m unittest discover -s tests -p 'test_pose_*.py'
python -m unittest discover -s tests -p 'test_dwpose_*.py'
```

Batch tests use synthetic frames and a fake extractor to exercise failures,
interruptions, source drift, split enforcement, provenance and resume. They do not
claim real model inference or dataset-wide coverage.
