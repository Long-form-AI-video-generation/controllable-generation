# Controllable WAN — Depth & Multi-Modal Video Generation

A ControlNet-style adapter for **WAN 2.2 TI2V-5B** that enables controllable video generation using up to 6 control modalities (depth, sketch, motion, style, pose, mask). The adapter is trained on top of the frozen WAN backbone using zero-convolution injection ,meaning the pretrained model's behavior is fully preserved at initialization and controls are introduced gradually during training.



---

## Architecture Overview

```
Reference Video
    │
    ▼
EnhancedControlExtractor          ← MiDaS, Canny, Farneback, CLIP, YOLO/MediaPipe, SAM/DeepLab
    │
    ▼
ControlEncoderProcessor           ← 6 lightweight CNN encoders → 256-dim feature volumes
    │                                (depth, sketch, motion, style, pose, mask)
    ▼
ControlAdapter
    ├── per-modality projection   ← Linear → SiLU → LayerNorm  (one per modality)
    ├── modality_gates            ← 6 learned sigmoid scalars (which modalities matter)
    └── fusion layer              ← concat → Linear → SiLU → LayerNorm → (B, T×16×16, dit_dim)
    │
    ▼
ZeroLinear × 4                    ← zero-initialized projections, one per injection layer
    │
    ▼
WAN 2.2 DiT (frozen)
    ├── Block 0   ← pre-hook: x = x + zero_conv[0](ctrl)
    ├── ...
    ├── Block 8   ← pre-hook: x = x + zero_conv[1](ctrl)
    ├── ...
    ├── Block 16  ← pre-hook: x = x + zero_conv[2](ctrl)
    ├── ...
    └── Block 24  ← pre-hook: x = x + zero_conv[3](ctrl)
    │
    ▼
noise_pred → denoised video
```

**Trainable parameters only:** ControlAdapter + 4 ZeroLinear layers (~tens of millions). WAN, VAE, and T5 are fully frozen throughout training.

---

## Project Structure

```
.
├── Wan2.2/                         # WAN 2.2 submodule (TI2V-5B weights + model code)
│   └── Wan2.2-TI2V-5B/
│       ├── config.json
│       ├── diffusion_pytorch_model.safetensors.index.json
│       ├── Wan2.2_VAE.pth
│       └── models_t5_umt5-xxl-enc-bf16.pth
│
├── models/                         # Pretrained weights for control extractors
│   ├── midas_v3_dpt_large.pth
│   ├── open_clip_pytorch_model.bin
│   ├── table5_pidinet.pth          # optional, falls back to Canny
│   └── sam_vit_h_4b8939.pth       # optional, falls back to DeepLabV3
│
├── src/
│   ├── data/
│   │   ├── dataset.py              # ControllableVideoDataset
│   │   └── extract_control.py      # EnhancedControlExtractor + process_shot_with_all_controls
│   │
│   └── models/
│       ├── control_adapter.py      # ControlAdapter 
│       ├── wan_controllable.py     # ControllableWAN 
│       └── encode_controls.py      # ControlEncoderProcessor 
        └── train.py                # Training entry point (MultiVideoTrainer)
│
├── tests/test-allcontrols.py         # Inference + side-by-side comparison output
└── checkpoints/
    └── multi_video/
        ├── checkpoint_best.pt
        ├── checkpoint_step_*.pt
        └── training_log.jsonl
```

---

## Requirements

```
torch >= 2.1
diffusers
transformers
safetensors
opencv-python
mediapipe
ultralytics          # YOLOv8 pose (falls back to MediaPipe if unavailable)
segment-anything     # SAM (falls back to DeepLabV3 if unavailable)
tqdm
numpy
Pillow
```

WAN 2.2 dependencies (inside `Wan2.2/`):
```
pip install -e Wan2.2/
```

---

## Setup

**1. Clone and install**
```bash
git clone <repo-url>
cd controllable-generation
pip install -r requirements.txt
pip install -e Wan2.2/
```

**2. Download WAN 2.2 TI2V-5B weights**

Place the following files under `Wan2.2/Wan2.2-TI2V-5B/`:
- `config.json`
- `diffusion_pytorch_model.safetensors.index.json` + shards
- `Wan2.2_VAE.pth`
- `models_t5_umt5-xxl-enc-bf16.pth`

**3. Download control extractor weights**

Place under `models/`:
- `midas_v3_dpt_large.pth` — depth estimation ([MiDaS](https://github.com/isl-org/MiDaS))
- `open_clip_pytorch_model.bin` — style encoding ([OpenCLIP](https://github.com/mlfoundations/open_clip))
- `sam_vit_h_4b8939.pth` — segmentation masks ([SAM](https://github.com/facebookresearch/segment-anything)) *(optional)*

---

## Data Preparation

**Step 1 — Extract raw control signals from your video dataset**
```bash
python src/data/extract_control.py \
    --videos_dir   /data/videos \
    --shots_json   /data/shots_metadata.json \
    --output_dir   /data/control_signals
```

This runs MiDaS (depth), Canny (sketch/edges), Farneback (optical flow), CLIP (style), YOLO/MediaPipe (pose), and SAM/DeepLabV3 (masks) on every shot and saves `.npz` files per shot.

**Step 2 — Encode raw signals into 256-dim feature volumes**
```bash
python src/models/encode_controls.py \
    --control_dir  /data/control_signals \
    --output_dir   /data/encoded_controls \
    --num_frames   8 \
    --resolution   256 256
```

Each output `*_encoded.npz` contains 6 keys: `depth_encoded`, `sketch_encoded`, `motion_encoded`, `style_encoded`, `pose_encoded`, `mask_encoded` — all shaped `(1, 256, T, H, W)` in float16.

**Expected dataset layout after both steps:**
```
/data/
├── videos/
│   └── <video_id>.mp4
├── shots_metadata.json
├── control_signals/
│   └── <video_id>/shot_<id>_controls.npz
└── encoded_controls/
    └── <video_id>/shot_<id>_controls_encoded.npz
```

---

## Training

```bash
python src/models/train.py
```

Key config values (edit inside `main()` in `train_multi_video.py`):

| Parameter | Default | Notes |
|---|---|---|
| `num_frames` | 4 | Frames per training clip — increase once pipeline is stable |
| `resolution` | (128, 128) | Spatial resolution of training latents |
| `lr` | 1e-4 | Base adapter learning rate |
| `grad_accum_steps` | 8 | Effective batch = batch_size × grad_accum_steps |
| `loss_flow_weight` | 1.0 | Weight on standard flow-matching MSE loss |
| `loss_weighted_weight` | 0.1 | Weight on timestep-weighted flow loss |
| `checkpoint_dir` | `checkpoints/multi_video` | Where to save |
| `data_dir` | `/mnt/d1/controllable-generation` | Root of prepared dataset |

**Resuming:** On startup the script scans `checkpoint_dir` for the latest `.pt` file and prompts to resume.

**What to monitor in `training_log.jsonl`:**

| Metric | Healthy range | Action if wrong |
|---|---|---|
| `zero_conv_mean_weight_norm` | Starts at 0, slowly rises | If stuck at 0 after 500 steps → gradient not flowing |
| `zero_conv_max_weight_norm` | Should not exceed ~0.1 in first 100 steps | If it does → LR too high |
| `gate_depth` → `gate_style` | Should diverge from 0.5 over time | If all stay at 0.5 → gates not learning |
| `loss` | Should decrease steadily | Plateau early → check data pipeline |

**GPU layout:** Designed for dual-GPU setups.
- `cuda:0` — WAN DiT + ControlAdapter + ZeroConvs
- `cuda:1` — VAE encoder/decoder + T5 text encoder + control encoders

Single-GPU is possible but requires careful memory management (reduce `num_frames` and `resolution`).

---

## Inference

```bash
python tests/test-allcontrols.py \
    --checkpoint  checkpoints/multi_video/checkpoint_best.pt \
    --ref_video   /path/to/reference.mp4 \
    --ref_image   /path/to/first_frame.png \
    --prompt     "An anime character in a dramatic scene" \
    --output      result \
    --steps       50 \
    --size        480*832
```

This produces **three files**:
- `base_output.mp4` — generation with no controls (frozen WAN only)
- `controlled_output.mp4` — generation with all 6 controls injected
- `comparison.mp4` — side-by-side with labels burned in



## Checkpoints

Checkpoints saved by the trainer contain:

```python
{
    'model':        control_adapter.state_dict(),   # ControlAdapter weights
    'zero_convs':   zero_convs.state_dict(),        # 4 × ZeroLinear weights
    'optimizer':    ...,
    'lr_scheduler': ...,
    'scaler':       ...,
    'global_step':  int,
    'epoch':        int,
    'best_val_loss': float,
    'config':       dict,
}
