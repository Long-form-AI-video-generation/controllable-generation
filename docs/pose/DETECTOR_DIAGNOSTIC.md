# Diagnose detector dropouts in saved diagnostic batches

Run this after batch extraction/resume, when visual review finds subjects present
in frames with no returned detections. This is a bounded development diagnostic;
it does not change thresholds, run pose estimation, fill missing maps or rewrite
the baseline controls. A separate source worktree is recommended so the original
batch can still resume with its original code identity.

```bash
cd /path/to/diagnostic-review-checkout
conda activate controllable-gen
python -m scripts.pose.diagnose_detector \
  --batch-config /path/to/private/batch/config.json \
  --video-id DEVELOPMENT_VIDEO_A --video-id DEVELOPMENT_VIDEO_B \
  --max-frames 16 --top-k 20 \
  --output-dir /path/to/ignored/artifacts/detector-review-v1 \
  --dry-run
```

Remove `--dry-run` for detector inference. The requested IDs must already belong
to the existing train/validation batch. Selected bundles and source bytes are
verified before inference. The default 16-frame bound allows two eight-frame
shots. A normal run requires clean code; `--allow-dirty-code` records source changes
for local development only. The pinned factory loads and verifies both model
sessions, but only the detector runs here. Each frame invokes that detector once.

The command wraps the detector session temporarily, captures its ordinary output,
and restores the original session even on failure. Its independent explanation of
filtering must reproduce the ordinary accepted boxes and scores exactly. Existing
production detector/pose code and defaults are not modified.

Saved output, in a fresh directory published atomically after checks:

- `manifest.json`: code/model identities, baseline manifest hashes, compact per-frame
  counts, maximum raw person score, and agreement with stored baseline detections.
- Per-frame JSON: Top-K raw person proposals, uncalibrated objectness/class/product
  scores, unclipped/clipped source boxes and rejection reason. Top-K is explicitly
  truncated; it does not establish that no other proposals exist.
- Per-frame NPZ: **all** decoded boxes, person scores, objectness/person-class
  components and filter-stage codes, before Top-K selection.
- Per-shot PNG: source frame beside the Top-5 raw proposals. Green = accepted;
  amber = below final cutoff; red = below initial cutoff; purple = suppressed by
  NMS; gray = empty after clipping. High-ranking duplicates can overlap. These are
  proposals, not verified people or pose joints. Numerical details are in JSON.

The initial cutoff uses the existing NumPy comparison; the final cutoff uses the
same Python-float conversion as the production detector. Equality near a cutoff
can therefore depend on float32 representation. The audit preserves that behavior.

Read the results in this order:

1. Check `baseline_rerun_matches_stored`. If false, investigate reproducibility
   before attributing the old dropout to a filter stage.
2. Locate proposals over the visible subject. A high global score could belong
   to a different region; scores alone do not identify the intended subject.
3. Inspect their rejection reasons. A good subject box below the final cutoff
   suggests a threshold-related dropout. Low or misplaced proposals suggest a
   detector/generalization/localization issue worth investigating further.
4. Keep any later threshold comparison separate, on development data, with both
   visible-person and no-person/false-positive cases. This diagnostic selects no
   new universal threshold and establishes no pose accuracy or training eligibility.

Baseline sources/controls remain private and unchanged. Keep diagnostic JSON,
NPZ, images and logs under an ignored artifact directory, outside commits and PRs.
