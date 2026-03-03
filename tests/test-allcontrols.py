
import sys
import gc
import argparse
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'Wan2.2'))

import wan
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS
from src.models.wan_controllable import ControllableWAN

ALL_CONTROL_KEYS = [
    'depth_encoded',
    'mask_encoded',
    'motion_encoded',
    'pose_encoded',
    'sketch_encoded',
    'style_encoded',
]

def extract_and_encode_all_controls(
    video_path: str,
    device: str,
    num_frames: int = 8,
    resolution: tuple = (256, 256),
) -> dict:
   
    from src.data.extract_control import (
        EnhancedControlExtractor,
        process_shot_with_all_controls,
    )
    from src.models.encode_controls import ControlEncoderProcessor

    print(f"\n  [ctrl] Extracting raw control signals...")
    extractor = EnhancedControlExtractor(
        device=device,
        models_dir=str(project_root / 'models'),
    )

    cap          = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    shot = {
        'video_id':            Path(video_path).stem,
        'shot_id':             'infer',
        'segment_start_frame': 0,
        'segment_end_frame':   total_frames,
    }

    with tempfile.TemporaryDirectory() as tmp_raw:
        ok = process_shot_with_all_controls(
            video_path=str(video_path),
            shot=shot,
            output_dir=tmp_raw,
            extractor=extractor,
            sample_rate=4,
            target_size=(360, 640),
            extract_full_controls=True,
        )
        if not ok:
            raise RuntimeError("Control signal extraction failed")

        print(f"  [ctrl] Encoding into 256-dim feature volumes...")
        with tempfile.TemporaryDirectory() as tmp_enc:
            processor = ControlEncoderProcessor(
                control_base_dir=tmp_raw,
                output_dir=tmp_enc,
                device=device,
                num_frames=num_frames,
                resolution=resolution,
            )
            processor.process_all()

            npz_files = list(Path(tmp_enc).rglob('*_encoded.npz'))
            if not npz_files:
                raise RuntimeError("ControlEncoderProcessor produced no output")

            encoded = np.load(npz_files[0])
            controls = {
                k: torch.from_numpy(encoded[k]).float().to(device)
                for k in encoded.keys()
            }

    ref_shape = next(iter(controls.values())).shape  
    for key in ALL_CONTROL_KEYS:
        if key not in controls:
            print(f"  [ctrl] WARNING: '{key}' missing — padding with zeros")
            controls[key] = torch.zeros(
                ref_shape[0], 256, ref_shape[2], 128, 128,
                device=device,
            )

    controls = {k: controls[k] for k in ALL_CONTROL_KEYS}

    print(f"  [ctrl] Final control shapes:")
    for k, v in controls.items():
        print(f"    {k:20s}: {tuple(v.shape)}")

    return controls



def activate_adapter(controllable_wan: ControllableWAN, control_features: dict):
    
    with torch.no_grad():
        ctrl = controllable_wan.control_adapter(control_features)
    controllable_wan._control_signal = ctrl

    gates = torch.sigmoid(controllable_wan.control_adapter.modality_gates)
    gate_str = "  ".join(
        f"{k.replace('_encoded','')[:5]}={gates[i].item():.3f}"
        for i, k in enumerate(sorted(ALL_CONTROL_KEYS))
    )
    print(f"  Adapter activated  |  ctrl norm={ctrl.norm():.4f}")
    print(f"  Gates: {gate_str}")


def deactivate_adapter(controllable_wan: ControllableWAN):
    
    controllable_wan._control_signal = None



def tensor_to_frames(video_tensor: torch.Tensor) -> np.ndarray:
  
    v = video_tensor.float()
    v = (v.clamp(-1, 1) + 1) / 2 * 255
    return v.permute(1, 2, 3, 0).cpu().numpy().astype(np.uint8)


def save_video(frames: np.ndarray, path: Path, fps: int = 16):
    T, H, W, _ = frames.shape
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H)
    )
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"    Saved: {path}  ({T} frames @ {fps} fps)")


def depth_to_rgb(depth_seq: np.ndarray, target_hw: tuple) -> np.ndarray:
   
    H, W = target_hw
    frames = []
    for d in depth_seq:
        d_u8 = (d * 255).astype(np.uint8)
        col  = cv2.applyColorMap(d_u8, cv2.COLORMAP_PLASMA)
        col  = cv2.resize(col, (W, H))
        frames.append(cv2.cvtColor(col, cv2.COLOR_BGR2RGB))
    return np.stack(frames)


def extract_depth_for_viz(
    video_path: str,
    device: str,
    num_frames: int,
    hw: tuple,
) -> np.ndarray:
   
    print("  [viz] Loading MiDaS for depth visualisation...")
    midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
    midas.to(device).eval()
    tf = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform

    cap   = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs  = np.linspace(0, total - 1, num_frames, dtype=int)

    depths = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            depths.append(depths[-1] if depths else np.zeros(hw, np.float32))
            continue
        inp = tf(frame).to(device)
        with torch.no_grad():
            pred = midas(inp)
            pred = F.interpolate(
                pred.unsqueeze(1), size=hw, mode='bicubic', align_corners=False,
            ).squeeze()
        d = pred.cpu().numpy().astype(np.float32)
        depths.append((d - d.min()) / (d.max() - d.min() + 1e-8))

    cap.release()
    del midas
    torch.cuda.empty_cache()
    gc.collect()

    return np.stack(depths)



def make_comparison_video(
    depth_frames: np.ndarray,    
    base_frames:  np.ndarray,    
    ctrl_frames:  np.ndarray,   
    path:         Path,
    fps:          int = 16,
    active_keys:  list = None,
):
    """
    [Depth viz | Base WAN | Controlled WAN] side-by-side.
    A subtitle strip below the comparison shows which controls were active.
    """
    T = min(len(depth_frames), len(base_frames), len(ctrl_frames))
    H, W = base_frames.shape[1], base_frames.shape[2]

    def _resize(seq):
        return np.stack([cv2.resize(f, (W, H)) for f in seq[:T]])

    depth_frames = _resize(depth_frames)
    base_frames  = _resize(base_frames)
    ctrl_frames  = _resize(ctrl_frames)

    LABEL_H   = 30   
    FOOTER_H  = 28    
    canvas_h  = LABEL_H + H + FOOTER_H
    canvas_w  = W * 3

    font   = cv2.FONT_HERSHEY_SIMPLEX
    fscale = 0.55
    fthick = 1

    col_labels = ["Depth (ref video)", "Base WAN", "Controlled WAN"]
    col_bgs    = [
        (50,  40,  40),   
        (20,  60,  20),   
        (20,  20,  80),   
    ]

    ctrl_summary = (
        "Active controls: " + ", ".join(k.replace('_encoded', '') for k in (active_keys or ALL_CONTROL_KEYS))
    )

    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (canvas_w, canvas_h)
    )

    for i in range(T):
        canvas = np.zeros((canvas_h, canvas_w, 3), np.uint8)
        panels = [depth_frames[i], base_frames[i], ctrl_frames[i]]

        for col_idx, (panel, lbl, bg) in enumerate(zip(panels, col_labels, col_bgs)):
            x0 = col_idx * W

            strip = np.full((LABEL_H, W, 3), bg, np.uint8)
            (tw, th), _ = cv2.getTextSize(lbl, font, fscale, fthick)
            cv2.putText(
                strip, lbl,
                (max(0, (W - tw) // 2), (LABEL_H + th) // 2 - 2),
                font, fscale, (220, 220, 220), fthick, cv2.LINE_AA,
            )
            canvas[:LABEL_H, x0:x0 + W] = strip

        
            canvas[LABEL_H:LABEL_H + H, x0:x0 + W] = cv2.cvtColor(
                panel, cv2.COLOR_RGB2BGR
            )

      
        footer = np.full((FOOTER_H, canvas_w, 3), (30, 30, 30), np.uint8)
        (fw, fh), _ = cv2.getTextSize(ctrl_summary, font, 0.48, 1)
        cv2.putText(
            footer, ctrl_summary,
            (max(0, (canvas_w - fw) // 2), (FOOTER_H + fh) // 2 - 2),
            font, 0.48, (180, 220, 180), 1, cv2.LINE_AA,
        )
        canvas[LABEL_H + H:, :] = footer

        writer.write(canvas)

    writer.release()
    print(f"    Saved: {path}  ({T} frames, 3-panel comparison)")



def parse_args():
    p = argparse.ArgumentParser(description="ControllableWAN — all-6-control comparison")

    p.add_argument('--ref_video',   required=True,
                   help='Reference video — all 6 controls extracted from this')
    p.add_argument('--ref_image',   required=True,
                   help='Reference image (first frame) for WAN TI2V')
    p.add_argument('--prompt',      required=True)
    p.add_argument('--checkpoint',  required=True,
                   help='checkpoint_*.pt containing adapter + zero_convs')
    p.add_argument('--wan_dir',     default='Wan2.2/Wan2.2-TI2V-5B')
    p.add_argument('--output_dir',  default='results')

    p.add_argument('--size',        default='480*832')
    p.add_argument('--frame_num',   type=int, default=81,
                   help='Frames to generate (must be 4n+1: 17,33,49,81...)')
    p.add_argument('--steps',       type=int,   default=40)
    p.add_argument('--guidance',    type=float, default=3.0)
    p.add_argument('--fps',         type=int,   default=16)
    p.add_argument('--seed',        type=int,   default=42)

    p.add_argument('--ctrl_num_frames', type=int, default=8,
                   help='Frames sampled for control encoding (independent of frame_num)')
    p.add_argument('--ctrl_resolution', type=int, nargs=2, default=[256, 256],
                   metavar=('H', 'W'),
                   help='Spatial resolution fed to ControlEncoderProcessor')
    p.add_argument('--depth_hw',    type=int, nargs=2, default=[128, 128],
                   metavar=('H', 'W'),
                   help='Resolution for depth visualisation only')

    p.add_argument('--active_controls', nargs='+', default=None,
                   choices=[k.replace('_encoded','') for k in ALL_CONTROL_KEYS],
                   metavar='CTRL',
                   help='Subset of controls to activate (default: all 6). '
                        'Choices: depth mask motion pose sketch style')

    p.add_argument('--offload', action='store_true', default=True,
                   help='Offload WAN to CPU between steps (saves VRAM)')

    return p.parse_args()



def main():
    args    = parse_args()
    device  = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.active_controls:
        active_keys = [f'{c}_encoded' for c in args.active_controls]
    else:
        active_keys = ALL_CONTROL_KEYS

    depth_hw = tuple(args.depth_hw)
    ctrl_res = tuple(args.ctrl_resolution)

    print("\n" + "="*70)
    print("ControllableWAN — All-Control Comparison Inference")
    print("="*70)
    print(f"  ref_video       : {args.ref_video}")
    print(f"  ref_image       : {args.ref_image}")
    print(f"  prompt          : {args.prompt}")
    print(f"  checkpoint      : {args.checkpoint}")
    print(f"  wan_dir         : {args.wan_dir}")
    print(f"  size            : {args.size}  |  frame_num : {args.frame_num}")
    print(f"  steps           : {args.steps}  |  guidance  : {args.guidance}")
    print(f"  seed            : {args.seed}")
    print(f"  ctrl_num_frames : {args.ctrl_num_frames}  |  ctrl_res : {ctrl_res}")
    print(f"  active_controls : {[k.replace('_encoded','') for k in active_keys]}")
    print(f"  output_dir      : {out_dir}")
    print("="*70 + "\n")

    cfg = WAN_CONFIGS['ti2v-5B']

    
    print("[1/5] Extracting & encoding all 6 control modalities...")
    all_controls = extract_and_encode_all_controls(
        video_path  = args.ref_video,
        device      = device,
        num_frames  = args.ctrl_num_frames,
        resolution  = ctrl_res,
    )

    
    active_controls = {k: all_controls[k] for k in active_keys}
  
    adapter_controls = {}
    for k in ALL_CONTROL_KEYS:
        if k in active_controls:
            adapter_controls[k] = all_controls[k]
        else:
            ref = next(iter(all_controls.values()))
            adapter_controls[k] = torch.zeros_like(ref)

    print("\n[2/5] Extracting depth for visualisation...")
    depth_seq = extract_depth_for_viz(
        video_path  = args.ref_video,
        device      = device,
        num_frames  = args.frame_num,
        hw          = depth_hw,
    )
    print(f"  depth_seq shape : {depth_seq.shape}")

  
    print("\n[3/5] Loading ControllableWAN + checkpoint...")
    ctrl_model = ControllableWAN(checkpoint_dir=args.wan_dir, device=device)
    ctrl_model.eval()

    ckpt = torch.load(args.checkpoint, map_location=device)
    ctrl_model.control_adapter.load_state_dict(ckpt['model'])
    if 'zero_convs' in ckpt:
        ctrl_model.zero_convs.load_state_dict(ckpt['zero_convs'])
        print("  Loaded adapter + zero_convs")
    else:
        print("  WARNING: no zero_convs key — controls will have zero effect")
    print(f"  step={ckpt.get('global_step','?')}  "
          f"best_val_loss={ckpt.get('best_val_loss','?')}")

    # ── [4/5] build WanTI2V pipeline, swap DiT ───────────────────────────────
    print("\n[4/5] Building WanTI2V pipeline (DiT swap)...")
    wan_pipeline = wan.WanTI2V(
        config=cfg,
        checkpoint_dir=args.wan_dir,
        device_id=0,
        rank=0,
        t5_cpu=True,
    )
    original_wan_model = wan_pipeline.model
    wan_pipeline.model = ctrl_model.wan
    print("  Pipeline DiT → ControllableWAN.wan  (hooks active)")

    ref_image = Image.open(args.ref_image).convert("RGB")

    gen_kwargs = dict(
        img           = ref_image,
        size          = SIZE_CONFIGS[args.size],
        max_area      = MAX_AREA_CONFIGS[args.size],
        frame_num     = args.frame_num,
        sampling_steps= args.steps,
        guide_scale   = args.guidance,
        seed          = args.seed,
        offload_model = args.offload,
    )

    def set_seed():
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    print("\n[5/5] Generating videos...")

    print("\n" + "-"*60)
    print("Run A — BASE  (vanilla WAN, no controls)")
    print("-"*60)
    deactivate_adapter(ctrl_model)
    set_seed()
    with torch.no_grad():
        video_base = wan_pipeline.generate(args.prompt, **gen_kwargs)
    frames_base = tensor_to_frames(video_base)
    save_video(frames_base, out_dir / "base.mp4", fps=args.fps)
    del video_base
    torch.cuda.empty_cache()

    print("\n" + "-"*60)
    print(f"Run B — CONTROLLED  "
          f"({len(active_keys)} controls: "
          f"{', '.join(k.replace('_encoded','') for k in active_keys)})")
    print("-"*60)
    activate_adapter(ctrl_model, adapter_controls)
    set_seed()
    with torch.no_grad():
        video_ctrl = wan_pipeline.generate(args.prompt, **gen_kwargs)
    deactivate_adapter(ctrl_model)
    frames_ctrl = tensor_to_frames(video_ctrl)
    save_video(frames_ctrl, out_dir / "controlled.mp4", fps=args.fps)
    del video_ctrl
    torch.cuda.empty_cache()

    wan_pipeline.model = original_wan_model

    print("\n  Building side-by-side comparison...")
    T_out = frames_base.shape[0]
    if len(depth_seq) != T_out:
        idxs      = np.linspace(0, len(depth_seq) - 1, T_out, dtype=int)
        depth_seq = depth_seq[idxs]

    depth_rgb = depth_to_rgb(
        depth_seq,
        target_hw=(frames_base.shape[1], frames_base.shape[2]),
    )
    make_comparison_video(
        depth_frames = depth_rgb,
        base_frames  = frames_base,
        ctrl_frames  = frames_ctrl,
        path         = out_dir / "comparison.mp4",
        fps          = args.fps,
        active_keys  = active_keys,
    )

    diff = np.abs(frames_ctrl.astype(float) - frames_base.astype(float)).mean()
    print(f"\n  Pixel-level control effect: {diff:.4f}")
    if diff < 0.5:
        print("  ⚠  Outputs nearly identical.")
        print("     Either zero_conv weights are still near-zero (early training),")
        print("     or all active controls are zero-padded.")
        print("     Check 'zero_conv_mean_weight_norm' in training_log.jsonl.")
    else:
        print("  ✓  Controls are visibly influencing the output.")

    print("\n" + "="*70)
    print("Done!")
    print(f"  base.mp4       : {out_dir / 'base.mp4'}")
    print(f"  controlled.mp4 : {out_dir / 'controlled.mp4'}")
    print(f"  comparison.mp4 : {out_dir / 'comparison.mp4'}  ← start here")
    print("="*70)
    print("\nWhat to look for in comparison.mp4:")
    print("  Left   — depth map from reference video (plasma colourmap)")
    print("  Centre — WAN with prompt only, no adapter")
    print("  Right  — WAN with all active controls injected, same seed")
    print()
    print("  ✓ Working  : spatial layout / motion in Right tracks the reference")
    print("  ✗ Not yet  : both video columns look identical")
    print()


if __name__ == '__main__':
    main()