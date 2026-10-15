# Controllable Video Generation with WAN 2.2

This repository implements **Experiment B**, a ControlNet-style extension for
WAN 2.2 TI2V-5B. Depth, Canny edge, and semantic mask are trained as independent
control experts and can be composed at inference time through one frozen WAN
backbone.

The WAN DiT, VAE, and T5 encoder remain frozen. Each expert learns a lightweight
adapter and zero-initialized residual projections, allowing control influence to
grow without changing the pretrained model at initialization.

## Project status

| Control | Preparation | Training | Standalone inference | Combined inference |
|---|---:|---:|---:|---:|
| Depth | Complete | Complete | Validated | Validated |
| Canny edge | Complete | Complete | Validated | Validated |
| Semantic mask | Complete | Complete | Validated | Validated |

The validated three-control setting is depth `0.25`, Canny `0.25`, and mask
`0.25`, with a combined residual-ratio cap of `0.1`. A strength of `0.5` remains
the standalone default for each validated expert.

## Architecture

```text
Reference video
      |
      +--> MiDaS depth --------> depth expert ---+
      +--> Canny edges --------> Canny expert ---+--> normalized, capped residuals
      +--> SegFormer-B5 masks -> mask expert ----+              |
                                                                v
Reference image + prompt ------------------------------> frozen WAN 2.2 DiT
                                                                |
                                                                v
                                                        generated video
```

Each expert is trained separately and retains its own checkpoint. During
combined inference, `MultiControlHookController` attaches the requested experts
to the same WAN DiT blocks. Expert strengths are independent; residual
normalization and the combined cap prevent accumulated control pressure from
overwhelming the base model.

## Repository layout

```text
src/
  data/                 Shared dataset and frame-sampling contracts
  depth_models/         Depth preprocessing, adapter, trainer, and WAN wrapper
  sketch_models/        Canny preprocessing, adapter, training, and evaluation
  mask_models/          SegFormer mask pipeline, training, and evaluation
  control_integration/  Shared preparation, checkpoint loading, and fusion
tests/
  test_onecontrol_inference.py       Depth inference runner
  test_sketch_inference.py           Canny inference runner
  test_mask_inference.py             Mask inference runner
  prepare_multi_control_inputs.py    Matched multi-control preparation
  test_multi_control_inference.py    One-WAN combined inference runner
```

`Wan2.2/`, model weights, datasets, checkpoints, and generated outputs are kept
outside version control.

## Environment

Run commands from the repository root:

```bash
export PYTHONPATH="$PWD/src:$PWD/Wan2.2:$PWD"
```

The current WAN wrappers expect two CUDA devices:

- `cuda:0`: WAN generation workload
- `cuda:1`: VAE or control-expert workload, depending on the entry point

The repository does not currently provide a single dependency lock file. The
runtime requires PyTorch, OpenCV, NumPy, Pillow, Diffusers, Transformers,
Safetensors, tqdm, the local WAN 2.2 package, MiDaS for depth, and SegFormer-B5
for semantic masks.

## Dataset contract

Experiments use the validated 524-video AnimeShooter dataset and a frozen
`314/105/105` train, validation, and test split. Reuse the same split manifest
for every expert so results remain comparable.

Prepared control files are compressed NumPy archives with one canonical key:

- Depth: `depth_encoded`
- Canny: `sketch_encoded`
- Mask: `mask_encoded`

Training should use strict dataset loading and the preprocessing manifest
created with each control dataset.

## Prepare training controls

### Depth

Convert raw MiDaS depth arrays into the depth training contract:

```bash
python src/depth_models/depth_control.py \
  --control_dir <raw-control-directory> \
  --output_dir <depth-output-directory> \
  --num_frames 8
```

### Canny edge

```bash
python -m src.sketch_models.prepare_dataset \
  --videos-dir <video-directory> \
  --metadata <shots-metadata.json> \
  --output-dir <canny-output-directory> \
  --num-frames 8 \
  --height 128 \
  --width 128
```

### Semantic mask

```bash
python -m src.mask_models.prepare_dataset \
  --videos-dir <video-directory> \
  --metadata <shots-metadata.json> \
  --output-dir <mask-output-directory> \
  --num-frames 8 \
  --height 128 \
  --width 128 \
  --device cuda:0 \
  --cache-dir <segformer-cache-directory>
```

Use `--allow-download` only when downloading the pinned SegFormer files is
intended. Otherwise, preprocessing requires the model to exist in the supplied
cache.

## Train standalone experts

Depth configuration is currently defined in `src/depth_models/train.py`:

```bash
python src/depth_models/train.py
```

Canny and mask trainers use explicit paths and validate their manifests:

```bash
python -m src.sketch_models.train \
  --data-dir <dataset-root> \
  --checkpoint-dir <canny-checkpoint-directory> \
  --wan-dir <WAN-directory> \
  --split-manifest <split-manifest.json> \
  --preprocessing-manifest <sketch-preprocessing-manifest.json>

python -m src.mask_models.train \
  --data-dir <dataset-root> \
  --checkpoint-dir <mask-checkpoint-directory> \
  --wan-dir <WAN-directory> \
  --split-manifest <split-manifest.json> \
  --preprocessing-manifest <mask-preprocessing-manifest.json>
```

The default Canny and mask schedule is 40 epochs with a maximum of 1,600
optimizer steps.

## Run combined inference

First create one immutable bundle in which every expert uses the same decoded
reference frames:

```bash
python tests/prepare_multi_control_inputs.py \
  --ref-video <reference.mp4> \
  --ref-image <target-image.jpg> \
  --output-dir <prepared-controls-directory> \
  --frame-num 81 \
  --experts depth canny mask \
  --midas-repo <local-MiDaS-repository> \
  --midas-weights <dpt-large-weights> \
  --midas-device cuda:0 \
  --segformer-device cuda:0 \
  --segformer-cache-dir <segformer-cache-directory>
```

Then run the base and controlled generations through one WAN pipeline:

```bash
python tests/test_multi_control_inference.py \
  --ref-image <target-image.jpg> \
  --prompt "<generation prompt>" \
  --prepared-controls-dir <prepared-controls-directory> \
  --depth-checkpoint <depth-checkpoint.pt> \
  --canny-checkpoint <canny-checkpoint.pt> \
  --mask-checkpoint <mask-checkpoint.pt> \
  --wan-dir <WAN-directory> \
  --output-dir <output-directory> \
  --size '480*832' \
  --frame-num 81 \
  --steps 20 \
  --guidance 3.0 \
  --seed 42 \
  --fps 16 \
  --control-combinations depth+canny+mask \
  --depth-strength 0.25 \
  --canny-strength 0.25 \
  --mask-strength 0.25 \
  --combined-ratio-cap 0.1 \
  --controller-device cuda:1 \
  --diagnostics
```

`frame-num` must be positive and satisfy `4n+1`, such as 17 or 81. The runner
produces base, controlled, comparison, diagnostic-frame, and metadata artifacts.

## Verification

Run the contract tests before inference:

```bash
python -m unittest discover -s tests -p 'test_control_integration_*.py' -v
python -m unittest discover -s tests -p 'test_sketch_*.py' -v
python -m unittest discover -s tests -p 'test_mask_*.py' -v
```

## Development workflow

- `main` is the stable reviewed branch.
- `dev` contains the current integrated implementation.
- New changes are developed on feature branches created from `dev`.
- Feature branches merge back into `dev`; validated releases are promoted from
  `dev` to `main`.
