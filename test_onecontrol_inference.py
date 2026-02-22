
import sys
import gc
import argparse
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



def load_midas(device):
    print("  Loading MiDaS DPT_Large...")
    midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
    midas.to(device).eval()
    tf = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
    return midas, tf


def extract_depth_frame(frame_bgr, midas, tf, device, hw=(128, 128)):
    inp = tf(frame_bgr).to(device)
    with torch.no_grad():
        pred = midas(inp)
        pred = F.interpolate(pred.unsqueeze(1), size=hw,
                             mode="bicubic", align_corners=False).squeeze()
    d = pred.cpu().numpy().astype(np.float32)
    return (d - d.min()) / (d.max() - d.min() + 1e-8)


def extract_depth_sequence(video_path, midas, tf, device, num_frames=8, hw=(128, 128)):
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
        depths.append(extract_depth_frame(frame, midas, tf, device, hw))
    cap.release()
    return np.stack(depths) 

def depth_to_control_tensor(depth_seq, device):
    """(T, H, W) → (1, 256, T, H, W) float32"""
    t = torch.from_numpy(depth_seq).float()
    t = t.unsqueeze(0).unsqueeze(0)           
    t = t.expand(-1, 256, -1, -1, -1).clone()
    return t.to(device)


def activate_adapter(controllable_wan, control_features):
    """Compute control signal and set it so hooks fire on next WAN forward."""
    with torch.no_grad():
        ctrl = controllable_wan.control_adapter(control_features)
    controllable_wan._control_signal = ctrl
    print(f"  Adapter activated  |  ctrl norm={ctrl.norm():.4f}  "
          f"gate={torch.sigmoid(controllable_wan.control_adapter.modality_gates).item():.4f}")


def deactivate_adapter(controllable_wan):
    """Clear control signal — WAN runs without adapter."""
    controllable_wan._control_signal = None

def tensor_to_frames(video_tensor):
    """
    WAN generate() returns (C, T, H, W) in [-1, 1].
    Returns (T, H, W, 3) uint8 RGB.
    """
    v = video_tensor.float()
    v = (v.clamp(-1, 1) + 1) / 2 * 255
    v = v.permute(1, 2, 3, 0).cpu().numpy().astype(np.uint8)
    return v


def save_video(frames, path, fps=16):
    T, H, W, _ = frames.shape
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H)
    )
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"    Saved: {path}  ({T} frames @ {fps} fps)")


def depth_to_rgb(depth_seq, target_hw):
    """(T, H, W) float32 → (T, H, W, 3) uint8 plasma, resized to target_hw."""
    H, W = target_hw
    frames = []
    for d in depth_seq:
        d_u8 = (d * 255).astype(np.uint8)
        col  = cv2.applyColorMap(d_u8, cv2.COLORMAP_PLASMA)
        col  = cv2.resize(col, (W, H))
        frames.append(cv2.cvtColor(col, cv2.COLOR_BGR2RGB))
    return np.stack(frames)


def make_comparison_video(depth_frames, base_frames, ctrl_frames, path, fps=16):
    """[Depth | Base WAN | Controlled WAN] side-by-side with labels."""
    T = min(len(depth_frames), len(base_frames), len(ctrl_frames))
    H, W = base_frames.shape[1], base_frames.shape[2]

    def resize_seq(seq):
        return [cv2.resize(f, (W, H)) for f in seq[:T]]

    d_list = resize_seq(depth_frames)
    b_list = resize_seq(base_frames)
    c_list = resize_seq(ctrl_frames)

    label_h  = 30
    canvas_h = H + label_h
    canvas_w = W * 3

    font   = cv2.FONT_HERSHEY_SIMPLEX
    fscale = 0.55
    fthick = 1
    labels = ["Depth (ref video)", "Base WAN (no control)", "Controlled WAN"]
    bgs    = [(40, 40, 40), (20, 60, 20), (20, 20, 80)]  # BGR

    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (canvas_w, canvas_h)
    )

    for i in range(T):
        row = np.zeros((canvas_h, canvas_w, 3), np.uint8)
        panels = [d_list[i], b_list[i], c_list[i]]
        for col_idx, (panel, lbl, bg) in enumerate(zip(panels, labels, bgs)):
            x0    = col_idx * W
            strip = np.full((label_h, W, 3), bg, np.uint8)
            (tw, th), _ = cv2.getTextSize(lbl, font, fscale, fthick)
            cv2.putText(strip, lbl,
                        (max(0, (W - tw) // 2), (label_h + th) // 2 - 2),
                        font, fscale, (220, 220, 220), fthick, cv2.LINE_AA)
            row[:label_h, x0:x0 + W] = strip
            row[label_h:, x0:x0 + W] = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
        writer.write(row)

    writer.release()
    print(f"    Saved: {path}  ({T} frames, side-by-side)")



def parse_args():
    p = argparse.ArgumentParser(description="ControllableWAN comparison inference")
    p.add_argument('--ref_video',   required=True,
                   help='Reference video — depth extracted from this')
    p.add_argument('--ref_image',   required=True,
                   help='Reference image — first frame for TI2V')
    p.add_argument('--prompt',      required=True)
    p.add_argument('--checkpoint',  required=True,
                   help='checkpoint_*.pt with adapter + zero_convs')
    p.add_argument('--wan_dir',     default='Wan2.2/Wan2.2-TI2V-5B')
    p.add_argument('--output_dir',  default='results')
    p.add_argument('--size',        default='480*832',
                   help='WAN size string e.g. 480*832 or 720*1280')
    p.add_argument('--frame_num',   type=int, default=17,
                   help='Frames to generate — must be 4n+1 (17, 33, 49, 81...)')
    p.add_argument('--steps',       type=int,   default=20)
    p.add_argument('--guidance',    type=float, default=5.0)
    p.add_argument('--fps',         type=int,   default=16)
    p.add_argument('--seed',        type=int,   default=42)
    p.add_argument('--depth_hw',    type=int,   nargs=2, default=[128, 128],
                   metavar=('H', 'W'),
                   help='Resolution for depth control tensor — match training')
    p.add_argument('--offload',     action='store_true', default=True,
                   help='Offload WAN to CPU between steps (saves VRAM)')
    return p.parse_args()



def main():
    args    = parse_args()
    device  = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    depth_hw = tuple(args.depth_hw)

    print("\n" + "="*70)
    print("ControllableWAN — Base vs Controlled Comparison")
    print("="*70)
    print(f"  ref_video  : {args.ref_video}")
    print(f"  ref_image  : {args.ref_image}")
    print(f"  prompt     : {args.prompt}")
    print(f"  checkpoint : {args.checkpoint}")
    print(f"  wan_dir    : {args.wan_dir}")
    print(f"  size       : {args.size}  |  frame_num : {args.frame_num}")
    print(f"  steps      : {args.steps}  |  guidance  : {args.guidance}")
    print(f"  seed       : {args.seed}  |  depth_hw  : {depth_hw}")
    print(f"  output_dir : {out_dir}")
    print("="*70 + "\n")

    cfg = WAN_CONFIGS['ti2v-5B']

   
    print("[1/4] Extracting depth from reference video...")
    midas, midas_tf = load_midas(device)
    depth_seq = extract_depth_sequence(
        args.ref_video, midas, midas_tf, device,
        num_frames=args.frame_num, hw=depth_hw,
    )
    print(f"  depth shape : {depth_seq.shape}  "
          f"min={depth_seq.min():.3f}  max={depth_seq.max():.3f}")
    del midas, midas_tf
    torch.cuda.empty_cache()
    gc.collect()

    control_features = {
        'depth_encoded': depth_to_control_tensor(depth_seq, device)
    }
    print(f"  control tensor : {control_features['depth_encoded'].shape}")

    print("\n[2/4] Loading ControllableWAN...")
    ctrl_model = ControllableWAN(checkpoint_dir=args.wan_dir, device=device)
    ctrl_model.eval()

    print(f"\n[3/4] Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    ctrl_model.control_adapter.load_state_dict(ckpt['model'])
    if 'zero_convs' in ckpt:
        ctrl_model.zero_convs.load_state_dict(ckpt['zero_convs'])
        print("  Loaded adapter + zero_convs")
    else:
        print("  WARNING: no zero_convs key in checkpoint")
    print(f"  step={ckpt.get('global_step','?')}  "
          f"best_val_loss={ckpt.get('best_val_loss','?')}")


    print("\n[4/4] Building WanTI2V pipeline...")
    wan_pipeline = wan.WanTI2V(
        config=cfg,
        checkpoint_dir=args.wan_dir,
        device_id=0,
        rank=0,
        t5_cpu=True,
       
    )

   
    original_wan_model  = wan_pipeline.model
    wan_pipeline.model  = ctrl_model.wan
    print("  Pipeline DiT swapped → ControllableWAN.wan (hooks active)")

    ref_image = Image.open(args.ref_image).convert("RGB")

    print("\n" + "-"*60)
    print("Run A — BASE  (vanilla WAN, _control_signal = None)")
    print("-"*60)

    deactivate_adapter(ctrl_model)  

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    with torch.no_grad():
        video_base = wan_pipeline.generate(
            args.prompt,
            img=ref_image,
            size=SIZE_CONFIGS[args.size],
            max_area=MAX_AREA_CONFIGS[args.size],
            frame_num=args.frame_num,
            sampling_steps=args.steps,
            guide_scale=args.guidance,
            seed=args.seed,
            offload_model=args.offload,
        )

    frames_base = tensor_to_frames(video_base)
    path_base   = out_dir / "base.mp4"
    save_video(frames_base, path_base, fps=args.fps)
    del video_base
    torch.cuda.empty_cache()

    
    print("\n" + "-"*60)
    print("Run B — CONTROLLED  (depth adapter active)")
    print("-"*60)

    activate_adapter(ctrl_model, control_features)

    torch.manual_seed(args.seed)   
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    with torch.no_grad():
        video_ctrl = wan_pipeline.generate(
            args.prompt,
            img=ref_image,
            size=SIZE_CONFIGS[args.size],
            max_area=MAX_AREA_CONFIGS[args.size],
            frame_num=args.frame_num,
            sampling_steps=args.steps,
            guide_scale=args.guidance,
            seed=args.seed,
            offload_model=args.offload,
        )

    deactivate_adapter(ctrl_model) 
    frames_ctrl = tensor_to_frames(video_ctrl)
    path_ctrl   = out_dir / "controlled.mp4"
    save_video(frames_ctrl, path_ctrl, fps=args.fps)
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
        target_hw=(frames_base.shape[1], frames_base.shape[2])
    )
    path_cmp = out_dir / "comparison.mp4"
    make_comparison_video(depth_rgb, frames_base, frames_ctrl, path_cmp, fps=args.fps)

    
    print("\n" + "="*70)
    print("Done")
    print(f"  base.mp4        : {path_base}")
    print(f"  controlled.mp4  : {path_ctrl}")
    print(f"  comparison.mp4  : {path_cmp}  ← start here")
    print("="*70)
    print("\nWhat to look for:")
    print("  Depth col    — depth signal fed to adapter (plasma colourmap)")
    print("  Base col     — WAN with prompt only, no control")
    print("  Controlled   — WAN with depth adapter active, same seed")
    print()
    print("  Working  : spatial layout in 'Controlled' tracks depth structure")
    print("  Not yet  : both video cols look identical → adapter has no effect\n")


if __name__ == '__main__':
    main()