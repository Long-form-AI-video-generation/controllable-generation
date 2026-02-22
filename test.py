import torch
import torch.nn as nn
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root / 'Wan2.2'))
sys.path.insert(0, str(project_root))

import wan
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS
from wan.utils.utils import save_video
from PIL import Image
import numpy as np


def encode_controls_from_video(video_path: str, device: str = 'cuda:1'):
    import cv2
    sys.path.insert(0, str(project_root / 'src'))
    from data.extract_control import EnhancedControlExtractor, process_shot_with_all_controls
    from models.encode_controls import ControlEncoderProcessor
    import tempfile

    print(f"  Extracting controls from: {video_path}")
    extractor = EnhancedControlExtractor(
        device=device,
        models_dir=str(project_root / 'models')
    )

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    shot = {
        'video_id': Path(video_path).stem,
        'shot_id': 'test_shot',
        'segment_start_frame': 0,
        'segment_end_frame': total_frames
    }

    with tempfile.TemporaryDirectory() as tmp_raw:
        success = process_shot_with_all_controls(
            video_path=video_path,
            shot=shot,
            output_dir=tmp_raw,
            extractor=extractor,
            sample_rate=4,
            target_size=(360, 640),
            extract_full_controls=True
        )
        if not success:
            raise RuntimeError("Control extraction failed")

        with tempfile.TemporaryDirectory() as tmp_encoded:
            processor = ControlEncoderProcessor(
                control_base_dir=str(Path(tmp_raw)),
                output_dir=str(tmp_encoded),
                device=device,
                num_frames=8,
                resolution=(256, 256)
            )
            processor.process_all()

            encoded_npz = list(Path(tmp_encoded).rglob('*_encoded.npz'))[0]
            encoded = np.load(encoded_npz)
            controls = {
                k: torch.from_numpy(encoded[k]).float().to(device)
                for k in encoded.keys()
            }

    print(f"  Encoded controls: {list(controls.keys())}")
    return controls


def visualize_controls(controls: dict, output_dir: str = 'control_viz'):
    import cv2
    Path(output_dir).mkdir(exist_ok=True)
    for key, arr in controls.items():
        arr = arr[0].cpu().numpy()            
        T, H, W = arr.shape[1], arr.shape[2], arr.shape[3]
        r = arr[:85].mean(0)
        g = arr[85:170].mean(0)
        b = arr[170:].mean(0)
        rgb = np.stack([r, g, b], axis=-1)    
        rgb = rgb - rgb.min()
        rgb = rgb / (rgb.max() + 1e-8)
        rgb = (rgb * 255).astype(np.uint8)
        out = cv2.VideoWriter(
            f"{output_dir}/{key}.mp4",
            cv2.VideoWriter_fourcc(*'mp4v'), 8, (W*4, H*4)
        )
        for t in range(T):
            frame = cv2.cvtColor(rgb[t], cv2.COLOR_RGB2BGR)
            frame = cv2.resize(frame, (W*4, H*4), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(f"{output_dir}/{key}_frame{t:02d}.png", frame)
            out.write(frame)
        out.release()
        print(f"  ✓ {key}: {T} frames → {output_dir}/")


def test_trained_model(
    checkpoint_path: str,
    ref_image_path: str,
    caption: str,
    control_file: str,
    output_path: str = 'test_trained_output.mp4',
    preloaded_controls=None,
    size: str = '480*832',
):
    device = 'cuda:0'
    cfg = WAN_CONFIGS['ti2v-5B']

   
    INJECTION_LAYERS =[0, 8, 16,24]

   
    print("[1/5] Loading WAN TI2V pipeline...")
    wan_ti2v = wan.WanTI2V(
        config=cfg,
        checkpoint_dir='Wan2.2/Wan2.2-TI2V-5B',
        device_id=0,
        rank=0,
    )

    print("[2/5] Loading trained control adapter...")
    from src.models.control_adapter import ControlAdapter
    from src.models.wan_controllable import ZeroLinear

    control_adapter = ControlAdapter(
        control_dim=256,
        hidden_dim=1024,
        dit_dim=3072,
        num_controls=6,
        use_gradient_checkpointing=False,
    ).to('cuda:1')

    checkpoint = torch.load(checkpoint_path, map_location='cuda:1')
    control_adapter.load_state_dict(checkpoint['model'])
    control_adapter.eval()
    print(f"  Step {checkpoint.get('global_step', 'N/A')}  "
          f"| best val loss: {checkpoint.get('best_val_loss', 'N/A')}")

  
    zero_convs = nn.ModuleList([ZeroLinear(3072) for _ in INJECTION_LAYERS])
    if 'zero_convs' in checkpoint:
        zero_convs.load_state_dict(checkpoint['zero_convs'])
        print(f"  ✓ Loaded {len(INJECTION_LAYERS)} zero convs from checkpoint")
    else:
        print("  WARNING: no zero_convs in checkpoint — using zero-init (controls = 0)")
    zero_convs.eval()

  
    print("[3/5] Loading control signals...")
    if preloaded_controls is not None:
        controls = preloaded_controls
    else:
        encoded = np.load(control_file)
        controls = {
            k: torch.from_numpy(encoded[k]).float().to('cuda:1')
            for k in encoded.keys()
        }
    print(f" Controls: {list(controls.keys())}")

    with torch.no_grad():
        control_signal = control_adapter(controls)  
    print(f" Control signal: {control_signal.shape}")

    visualize_controls(controls, output_dir='control_viz')

    img = Image.open(ref_image_path).convert('RGB')

    generate_kwargs = dict(
    img=img,
    size=SIZE_CONFIGS[size],
    max_area=MAX_AREA_CONFIGS[size],
    frame_num=50,
    shift=cfg.sample_shift,
    sample_solver='unipc',
    sampling_steps=50,
    guide_scale=cfg.sample_guide_scale,
    seed=42,
    offload_model=True,
)

    def make_hook(ctrl_signal, zero_conv, scale=0.1):
        def hook(module, input):
            x = input[0]
            B, L, C = x.shape
            ctrl = ctrl_signal.to(x.device, dtype=x.dtype)

            if ctrl.shape[1] != L:
                ctrl = ctrl.permute(0, 2, 1)
                ctrl = torch.nn.functional.interpolate(
                    ctrl.float(), size=L, mode='linear', align_corners=False
                )
                ctrl = ctrl.permute(0, 2, 1)

            ctrl = zero_conv.to(x.device)(ctrl)
            x = x + scale * ctrl
            return (x,) + input[1:]
        return hook

  
    print("[4/5] Generating without controls (baseline)...")
    video_no_ctrl =wan_ti2v.generate(caption, **generate_kwargs)
    save_video(
        tensor=video_no_ctrl[None],
        save_file='no_control.mp4',
        fps=cfg.sample_fps,
        nrow=1, normalize=True, value_range=(-1, 1)
    )
    print(" Saved: no_control.mp4")

   
    print("[5/5] Generating with controls...")
    hooks = []
    for i, layer_idx in enumerate(INJECTION_LAYERS):
        if layer_idx < len(wan_ti2v.model.blocks):
            h = wan_ti2v.model.blocks[layer_idx].register_forward_pre_hook(
                make_hook(control_signal, zero_convs[i], scale=3.0)
            )
            hooks.append(h)
    print(f" Registered {len(hooks)} hooks")

    try:
        video_with_ctrl = wan_ti2v.generate(caption, **generate_kwargs)
    finally:
        for h in hooks:
            h.remove()
        print("  Hooks removed")

    save_video(
        tensor=video_with_ctrl[None],
        save_file=output_path,
        fps=cfg.sample_fps,
        nrow=1, normalize=True, value_range=(-1, 1)
    )
    print(f"  Saved: {output_path}")

  
    diff = (video_with_ctrl - video_no_ctrl).abs().mean().item()
    print(f"\n  Control effect magnitude: {diff:.6f}")
    if diff < 1e-4:
        print("  WARNING: outputs nearly identical — controls having no effect")
        print("           Check zero_conv_mean_weight_norm from training logs")
    else:
        print("  Controls are influencing the output")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/multi_video/checkpoint_best.pt')
    parser.add_argument('--ref_image', required=True)
    parser.add_argument('--caption', default='An anime character in a dramatic scene')
    parser.add_argument('--control_file', required=False)
    parser.add_argument('--output', default='test_trained_output.mp4')
    parser.add_argument('--size', default='480*832')
    parser.add_argument('--video_input', type=str, default=None)
    args = parser.parse_args()

    if args.video_input:
        controls = encode_controls_from_video(args.video_input, device='cuda:1')
        test_trained_model(
            args.checkpoint, args.ref_image, args.caption,
            control_file=None, output_path=args.output,
            preloaded_controls=controls, size=args.size,
        )
    else:
        test_trained_model(
            args.checkpoint, args.ref_image, args.caption,
            control_file=args.control_file, output_path=args.output,
            size=args.size,
        )