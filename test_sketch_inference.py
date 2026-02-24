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


def extract_sketch_xdog(frame_bgr, sigma=0.5, k=4.5, p=19, epsilon=-0.1, phi=200):
    
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    g1   = cv2.GaussianBlur(gray, (0, 0), sigma)
    g2   = cv2.GaussianBlur(gray, (0, 0), sigma * k)
    dog  = g1 - p * g2
    sketch = np.where(dog >= epsilon, 1.0, 1.0 + np.tanh(phi * dog))
    sketch = np.clip(sketch, 0, 1)
    sketch = (sketch * 255).astype(np.uint8)
    return cv2.bitwise_not(sketch)


def _try_load_lineart_anime():
    try:
     
        import types, mediapipe as _mp
        if not hasattr(_mp, 'solutions'):
            _mp.solutions = types.ModuleType('mediapipe.solutions')
        from controlnet_aux import LineartAnimeDetector
        det = LineartAnimeDetector.from_pretrained("lllyasviel/Annotators")
        print("  Using ControlNet anime lineart detector")
        return det
    except Exception as e:
        print(f"  controlnet-aux unavailable ({e}), falling back to XDoG")
        return None


def extract_sketch_sequence(video_path, num_frames, hw=(128, 128), method='auto'):
   
    detector = None
    if method in ('auto', 'lineart_anime'):
        detector = _try_load_lineart_anime()

    cap   = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs  = np.linspace(0, total - 1, num_frames, dtype=int)

    sketches = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            sketches.append(sketches[-1] if sketches else np.zeros(hw, np.uint8))
            continue

        if detector is not None:
            h, w = hw
            pil    = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = detector(pil, detect_resolution=512, image_resolution=max(h, w))
            sk     = np.array(result.convert('L'))
        else:
            sk = extract_sketch_xdog(frame)

        sk = cv2.resize(sk, (hw[1], hw[0]), interpolation=cv2.INTER_LINEAR)
        sketches.append(sk)

    cap.release()
    return np.stack(sketches)  


def sketch_to_control_tensor(sketch_seq, device):
  
    T, H, W = sketch_seq.shape
    t = torch.from_numpy(sketch_seq.astype(np.float32)) / 255.0   
    t = t.reshape(T, 1, H, W)
    t = F.interpolate(t, size=(128, 128), mode='bilinear', align_corners=False)
    t = t.reshape(1, T, 128, 128)
    t = t.expand(256, -1, -1, -1).clone()   
   
    return t.unsqueeze(0).to(device)



def activate_adapter(controllable_wan, control_features):
    with torch.no_grad():
        ctrl = controllable_wan.control_adapter(control_features)
    controllable_wan._control_signal = ctrl
    gate = torch.sigmoid(controllable_wan.control_adapter.modality_gates).item()
    print(f"  Adapter activated  |  ctrl norm={ctrl.norm():.4f}  gate={gate:.4f}")


def deactivate_adapter(controllable_wan):
    controllable_wan._control_signal = None


def tensor_to_frames(video_tensor):
  
    v = video_tensor.float()
    v = (v.clamp(-1, 1) + 1) / 2 * 255
    return v.permute(1, 2, 3, 0).cpu().numpy().astype(np.uint8)


def save_video(frames, path, fps=16):
    T, H, W, _ = frames.shape
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H)
    )
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"  Saved: {path}  ({T} frames @ {fps} fps)")


def sketch_to_rgb(sketch_seq, target_hw):
   
    H, W = target_hw
    frames = []
    for sk in sketch_seq:
        sk_resized = cv2.resize(sk, (W, H))
        frames.append(cv2.cvtColor(sk_resized, cv2.COLOR_GRAY2RGB))
    return np.stack(frames)


def make_comparison_video(sketch_frames, base_frames, ctrl_frames, path, fps=16):
  
    T = min(len(sketch_frames), len(base_frames), len(ctrl_frames))
    H, W = base_frames.shape[1], base_frames.shape[2]

    def resize_seq(seq):
        return [cv2.resize(f, (W, H)) for f in seq[:T]]

    s_list = resize_seq(sketch_frames)
    b_list = resize_seq(base_frames)
    c_list = resize_seq(ctrl_frames)

    label_h  = 30
    canvas_h = H + label_h
    canvas_w = W * 3

    font   = cv2.FONT_HERSHEY_SIMPLEX
    fscale = 0.55
    fthick = 1
    labels = ["Sketch (ref video)", "Base WAN (no control)", "Controlled WAN"]
    bgs    = [(40, 40, 40), (20, 60, 20), (20, 20, 80)]

    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (canvas_w, canvas_h)
    )

    for i in range(T):
        row    = np.zeros((canvas_h, canvas_w, 3), np.uint8)
        panels = [s_list[i], b_list[i], c_list[i]]
        for col_idx, (panel, lbl, bg) in enumerate(zip(panels, labels, bgs)):
            x0    = col_idx * W
            strip = np.full((label_h, W, 3), bg, np.uint8)
            (tw, _), _ = cv2.getTextSize(lbl, font, fscale, fthick)
            cv2.putText(strip, lbl,
                        (max(0, (W - tw) // 2), label_h // 2 + 5),
                        font, fscale, (220, 220, 220), fthick, cv2.LINE_AA)
            row[:label_h, x0:x0 + W] = strip
            row[label_h:,  x0:x0 + W] = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
        writer.write(row)

    writer.release()
    print(f"  Saved: {path}  ({T} frames, side-by-side)")

def parse_args():
    p = argparse.ArgumentParser(description="ControllableWAN — sketch inference")
    p.add_argument('--ref_video',  required=True,
                   help='Reference video — sketch extracted from this')
    p.add_argument('--ref_image',  required=True,
                   help='Reference image — first frame for TI2V')
    p.add_argument('--prompt',     required=True)
    p.add_argument('--checkpoint', required=True,
                   help='checkpoint_*.pt with adapter + zero_convs')
    p.add_argument('--wan_dir',    default='Wan2.2/Wan2.2-TI2V-5B')
    p.add_argument('--output_dir', default='results')
    p.add_argument('--size',       default='480*832',
                   help='WAN size string e.g. 480*832 or 720*1280')
    p.add_argument('--frame_num',  type=int, default=81,
                   help='Frames to generate — must be 4n+1 (17, 33, 49, 81…)')
    p.add_argument('--steps',      type=int,   default=30)
    p.add_argument('--guidance',   type=float, default=3.0)
    p.add_argument('--fps',        type=int,   default=24)
    p.add_argument('--seed',       type=int,   default=42)
    p.add_argument('--sketch_method', default='auto',
                   choices=['auto', 'lineart_anime', 'xdog', 'canny'],
                   help='Sketch extraction method (auto tries lineart_anime first)')
    p.add_argument('--offload',    action='store_true', default=True,
                   help='Offload WAN to CPU between steps (saves VRAM)')
    return p.parse_args()


def main():
    args    = parse_args()
    device  = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("ControllableWAN — Sketch Control Inference")
    print("="*70)
    print(f"  ref_video      : {args.ref_video}")
    print(f"  ref_image      : {args.ref_image}")
    print(f"  prompt         : {args.prompt}")
    print(f"  checkpoint     : {args.checkpoint}")
    print(f"  wan_dir        : {args.wan_dir}")
    print(f"  size           : {args.size}  |  frame_num : {args.frame_num}")
    print(f"  steps          : {args.steps}  |  guidance  : {args.guidance}")
    print(f"  seed           : {args.seed}")
    print(f"  sketch_method  : {args.sketch_method}")
    print(f"  output_dir     : {out_dir}")
    print("="*70 + "\n")

    cfg = WAN_CONFIGS['ti2v-5B']

    print("[1/4] Extracting sketch from reference video...")
    sketch_seq = extract_sketch_sequence(
        args.ref_video,
        num_frames=args.frame_num,
        hw=(128, 128),             
        method=args.sketch_method,
    )
    print(f"  sketch shape : {sketch_seq.shape}  "
          f"min={sketch_seq.min()}  max={sketch_seq.max()}")

    nonzero_pct = (sketch_seq > 10).mean() * 100
    if nonzero_pct < 0.5:
        print("  ⚠  WARNING: sketch appears all-black — check extraction")
    elif nonzero_pct > 99:
        print("  ⚠  WARNING: sketch appears all-white — phi may be too large")
    else:
        print(f"  sketch line density : {nonzero_pct:.1f}% non-zero pixels  ✓")

    control_features = {
        'sketch': sketch_to_control_tensor(sketch_seq, device)
    }
    print(f"  control tensor : {control_features['sketch'].shape}  "
          f"dtype={control_features['sketch'].dtype}")

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

    original_wan_model = wan_pipeline.model
    wan_pipeline.model = ctrl_model.wan
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
    print("Run B — CONTROLLED  (sketch adapter active)")
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
    if len(sketch_seq) != T_out:
        idxs       = np.linspace(0, len(sketch_seq) - 1, T_out, dtype=int)
        sketch_seq = sketch_seq[idxs]

    sketch_rgb = sketch_to_rgb(
        sketch_seq,
        target_hw=(frames_base.shape[1], frames_base.shape[2])
    )
    path_cmp = out_dir / "comparison.mp4"
    make_comparison_video(sketch_rgb, frames_base, frames_ctrl, path_cmp, fps=args.fps)

    print("\n" + "="*70)
    print("Done")
    print(f"  base.mp4        : {path_base}")
    print(f"  controlled.mp4  : {path_ctrl}")
    print(f"  comparison.mp4  : {path_cmp}  ← start here")
    print("="*70)
    print("\nWhat to look for:")
    print("  Sketch col   — lineart signal fed to adapter (grayscale)")
    print("  Base col     — WAN with prompt only, no control")
    print("  Controlled   — WAN with sketch adapter active, same seed")
    print()
    print("  Working  : edges/lines in 'Controlled' track sketch structure")
    print("  Not yet  : both video cols look identical → adapter has no effect\n")


if __name__ == '__main__':
    main()