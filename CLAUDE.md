# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A ControlNet-style adapter for **WAN 2.2 TI2V-5B** that adds controllable video generation across 6 modalities (depth, sketch/edges, motion/optical-flow, style, pose, mask). The WAN backbone, VAE, and T5 are **frozen**; only a `ControlAdapter` plus four zero-initialized projection layers are trained. Controls are injected into the frozen DiT via **forward pre-hooks** so the pretrained model is unchanged at initialization (zero-convs output exactly 0) and control influence grows during training. Dataset is AnimeShooter (anime YouTube clips segmented into shots).

`README.md` describes intent well but several commands/filenames in it are stale — trust the code and the notes below over the README.

## Pipeline & commands

Four stages, run in order. There is no `requirements.txt` and no test runner; everything is invoked as plain scripts.

**1. Extract raw control signals** (`src/data/extract_control.py`)
Runs MiDaS (depth), Canny (edges), Farneback (flow), CLIP (style), YOLOv8/MediaPipe (pose), SAM/DeepLabV3 (mask) per shot → `control_signals/<video_id>/shot_<id>_controls.npz`.
The README shows `--videos_dir/--shots_json/--output_dir` flags, but `main()` has **no argparse** — paths are hardcoded constants at the top of `main()`. Edit them or import `process_dataset(...)` directly.

**2. Encode raw signals → 256-dim feature volumes** (`src/models/encode_controls.py`, class `ControlEncoderProcessor`)
```bash
python src/models/encode_controls.py --control_dir <dir>/control_signals --output_dir <dir>/encoded_controls --num_frames 8 --resolution 256 256
```
Output `*_encoded.npz` holds 6 keys (`depth_encoded` … `style_encoded`), each `(1, 256, T, H, W)` float16. A file is kept only if ≥4 modalities encode successfully; missing motion/pose are zero-filled.

**3. Train** (`src/models/train.py`, class `MultiVideoTrainer`) — **this is the canonical trainer.**
```bash
python src/models/train.py     # config dict + data_dir are edited inside main(), not via CLI
```
On startup it scans `checkpoint_dir` for the latest `checkpoint_*.pt` and **prompts interactively** (`input()`) to resume. Logs go to `<checkpoint_dir>/training_log.jsonl`. Watch `zero_conv_mean_weight_norm` (starts 0, rises slowly) and the `gate_*` values (should diverge from ~0.5).

**4. Inference / comparison** (`tests/test-allcontrols.py`) — main inference entry point.
```bash
python tests/test-allcontrols.py \
  --checkpoint checkpoints/multi_video/checkpoint_best.pt \
  --ref_video <ref.mp4> --ref_image <first_frame.png> \
  --prompt "..." --size 480*832 --frame_num 81 --steps 40
```
`--ref_video`, `--ref_image`, `--prompt`, `--checkpoint` are all required. `--frame_num` must be `4n+1` (17/33/49/81…). Writes `base.mp4`, `controlled.mp4`, `comparison.mp4` to `--output_dir` (default `results/`). (README's `--output result`/`--steps 50`/`*_output.mp4` names are wrong.)

## Architecture (the parts that span files)

- **Zero-conv injection via hooks** (`wan_controllable.py`). `ControllableWAN` registers `forward_pre_hook`s on WAN DiT blocks `[0, 8, 16, 24]`. The control signal is computed once by `ControlAdapter`, stashed in `self._control_signal`, and each hook adds `zero_convs[i](ctrl)` to that block's input (with spatial/sequence interpolation to match token count). Setting `_control_signal = None` cleanly disables all controls — this is how the "base, no-control" run is produced. WAN itself is never modified.
- **Adapter** (`control_adapter.py`). Per-modality `Linear→SiLU→LayerNorm→Dropout`, each scaled by a learned `sigmoid(modality_gate)`, concatenated and fused to `dit_dim`. Returns the pre-zero-conv signal `(B, T·16·16, dit_dim)`; the zero-conv is applied per-layer inside the hook, not here.
- **Sorted-key contract.** The adapter sorts control keys **alphabetically** → fixed order `depth, mask, motion, pose, sketch, style`. `modality_gates` and `get_modality_weights()` follow this exact order. The adapter requires **exactly 6** keys; inference zero-pads any missing modality before calling it.
- **Encoders** (`encoders.py`). Six lightweight 3D-CNN encoders (`Conv3D`/`GroupNorm`/`SiLU`, residual blocks, spatial-only downsampling). Each ends in an `output_proj` initialized near-zero (Xavier gain 0.02). These are used **only in stage 2** (offline encoding); they are not part of the trained model and not in checkpoints.
- **Flow-matching training** (`train.py`). `noisy = (1-t)·latent + t·noise`, `target = noise - latent`, `FlowMatchEulerDiscreteScheduler`. Loss = flow MSE + 0.1·timestep-weighted MSE (+ small frame-diff smoothness + gate-entropy terms). LR groups: adapter `1e`, zero_convs `2×`, modality_gates `20×`.
- **Trainable scope.** Only `control_adapter` + `zero_convs` have `requires_grad`. Checkpoints save just `{'model': adapter, 'zero_convs': ...}` (+ optimizer/step/config). `export_for_inference()` also writes `.safetensors`.
- **Inference DiT swap.** `test-allcontrols.py` builds the official `wan.WanTI2V` pipeline, then hot-swaps `wan_pipeline.model = ctrl_model.wan` (the hooked DiT). Controls are toggled by `activate_adapter`/`deactivate_adapter` (which set/clear `_control_signal`) around `wan_pipeline.generate(...)`, same seed for both runs.

## Code map: canonical vs legacy

Several files are superseded experiments — prefer the canonical ones:

| Use | Avoid (legacy/experimental) |
|---|---|
| `src/models/train.py` (flow-matching, March 2026) | `src/models/train_controllable_wan.py` (older DDPM variant) |
| `src/data/dataset.py` (`ControllableVideoDataset`) | `src/data/data_loader.py` (AnimeShooter HF downloader only) |
| `src/models/wan_controllable.py`, `control_adapter.py`, `encoders.py`, `encode_controls.py` | entire `src/temp/` (old dataset/train drafts) |
| `tests/test-allcontrols.py` (full inference) | `src/tests/` (single-video / quick-train experiments) |

`src/data/process_dataset.py` converts raw AnimeShooter annotations (timestamps like `start_time`/`end_time`) into the per-shot metadata the dataset expects. `src/util/` holds one-off data utilities (`meta_info.py`, `split_data.py`, `process_control.py`).

## Conventions & gotchas

- **Dual-GPU is hardcoded, not configurable.** `wan_controllable.py` pins VAE to `cuda:1`, WAN DiT + adapter + zero_convs to `cuda:0`, and T5 to CPU (loaded/offloaded per call). `encode_video`/`decode_video` hardcode `cuda:1`→`cuda:0` transfers. Single-GPU requires editing these device literals, not just a flag.
- **Hardcoded Linux data root.** `data_dir`/path defaults are `/mnt/d1/controllable-generation` (and `extract_control.py` uses `target_size=(360, 640)` etc.). This repo's dev box is Windows — these paths won't exist locally and must be edited before running.
- **WAN must be present but is gitignored.** `Wan2.2/` (and `models/`, `checkpoints/`, `results/`, `data/`) are in `.gitignore`; `Wan2.2/` is referenced as a submodule in the README but there is no `.gitmodules`. The WAN package path is resolved and inserted into `sys.path` inside `wan_controllable.py` (`project_root/Wan2.2`); `import wan` works only once that module is imported or `Wan2.2/` is `pip install -e`'d.
- **Two import styles — run from repo root.** `src/models/*.py` add `src/` to `sys.path` and import `from data.… / from models.…`; `tests/test-allcontrols.py` adds repo root and imports `from src.models.…`. Run scripts from the repository root so both resolve.
- **Dataset split is deterministic by `shot_id` sort:** first 60% train / 20% val / 20% test (`dataset.py`). Captions are pre-encoded through T5 once and cached.
- The dataset's `__getitem__` swallows load errors and returns a zero-filled dummy sample with fixed shapes — silent data problems show up as flat loss, not exceptions.
