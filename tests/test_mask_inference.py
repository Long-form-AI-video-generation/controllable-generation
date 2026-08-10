"""Run matched-seed base and mask-controlled WAN inference."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
try:
    import cv2
    import torch
    from PIL import Image
except ImportError:
    cv2 = torch = Image = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Wan2.2"))

from src.data.frame_sampling import select_frame_indices
from src.mask_models.labels import colorize_id_map
from src.mask_models.preprocessing import (
    SegFormerMaskConfig, load_segformer, predict_id_maps,
)
from src.mask_models.inference_config import (
    strength_tag,
    validate_inference_settings,
)
if cv2 is not None and torch is not None:
    from src.mask_models.video_io import (
        describe_video_tensor, save_debug_frames, save_rgb_video,
        tensor_to_frames,
    )


def extract_reference_frames(video_path: str, frame_num: int) -> np.ndarray:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        capture.release()
        raise RuntimeError(f"Invalid frame count for {video_path}")
    indices = select_frame_indices(0, frame_count, frame_num)
    frames = []
    try:
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            okay, frame = capture.read()
            if not okay:
                raise RuntimeError(
                    f"Could not decode frame {index} from {video_path}"
                )
            frames.append(frame)
    finally:
        capture.release()
    return np.stack(frames)


def mask_to_rgb(mask: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    height, width = target_hw
    result = []
    for frame in mask:
        resized = cv2.resize(
            colorize_id_map(frame),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
        result.append(resized)
    return np.stack(result)


def save_comparison(
    mask_frames: np.ndarray,
    base_frames: np.ndarray,
    controlled_frames: np.ndarray,
    path: Path,
    fps: int,
) -> None:
    count = min(len(mask_frames), len(base_frames), len(controlled_frames))
    height, width = base_frames.shape[1:3]
    labels = ("semantic mask reference", "Base WAN", "Mask controlled")
    output = []
    for index in range(count):
        panels = []
        for label, source in zip(
            labels,
            (mask_frames, base_frames, controlled_frames),
        ):
            panel = cv2.resize(source[index], (width, height))
            panel = panel.copy()
            cv2.rectangle(panel, (0, 0), (width, 32), (0, 0, 0), -1)
            cv2.putText(
                panel,
                label,
                (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            panels.append(panel)
        output.append(np.concatenate(panels, axis=1))
    save_rgb_video(np.stack(output), path, fps=fps)


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _generate(pipeline, args, image):
    return pipeline.generate(
        args.prompt,
        img=image,
        size=SIZE_CONFIGS[args.size],
        max_area=MAX_AREA_CONFIGS[args.size],
        frame_num=args.frame_num,
        sampling_steps=args.steps,
        guide_scale=args.guidance,
        seed=args.seed,
        offload_model=args.offload,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref-video", required=True)
    parser.add_argument("--ref-image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--preprocessing-manifest", required=True)
    parser.add_argument("--wan-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--size", default="480*832")
    parser.add_argument("--frame-num", type=int, default=49)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--control-height", type=int, default=128)
    parser.add_argument("--control-width", type=int, default=128)
    parser.add_argument("--segformer-cache-dir")
    parser.add_argument("--allow-segformer-download", action="store_true")
    parser.add_argument(
        "--control-strengths", type=float, nargs="+", required=True
    )
    parser.add_argument("--offload", action="store_true", default=True)
    parser.add_argument("--no-offload", dest="offload", action="store_false")
    return parser


def main() -> None:
    if cv2 is None or torch is None:
        raise RuntimeError("mask inference requires OpenCV and PyTorch")
    import wan
    from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, WAN_CONFIGS
    from src.mask_models.wan_controllable import ControllableWAN
    globals().update(
        wan=wan,
        MAX_AREA_CONFIGS=MAX_AREA_CONFIGS,
        SIZE_CONFIGS=SIZE_CONFIGS,
        WAN_CONFIGS=WAN_CONFIGS,
    )
    args = build_parser().parse_args()
    args.control_strengths = validate_inference_settings(
        args.frame_num,
        args.control_strengths,
    )

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    reference_frames = extract_reference_frames(args.ref_video, args.frame_num)
    preprocessing = SegFormerMaskConfig(
        num_frames=args.frame_num,
        output_size=(args.control_height, args.control_width),
    )
    processor, extractor = load_segformer(
        preprocessing,
        cache_dir=args.segformer_cache_dir,
        local_files_only=not args.allow_segformer_download,
    )
    rgb_frames = reference_frames[..., ::-1].copy()
    mask = predict_id_maps(
        rgb_frames, processor, extractor, preprocessing, device="cuda"
    )
    extractor.to("cpu")
    del extractor, processor
    torch.cuda.empty_cache()
    control_features = {
        "mask_encoded": torch.from_numpy(mask)
        .unsqueeze(0)
        .unsqueeze(0)
        .long()
        .to("cuda")
    }

    pipeline = wan.WanTI2V(
        config=WAN_CONFIGS["ti2v-5B"],
        checkpoint_dir=args.wan_dir,
        device_id=0,
        rank=0,
        t5_cpu=True,
    )
    pipeline.model.to("cpu")
    del pipeline.model
    pipeline.model = None
    torch.cuda.empty_cache()
    gc.collect()

    model = ControllableWAN(checkpoint_dir=args.wan_dir, device="cuda")
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cuda",
        weights_only=False,
    )
    expected = model.checkpoint_metadata()
    if checkpoint.get("control_metadata") != expected:
        raise ValueError(
            "Checkpoint/model mismatch: "
            f"expected {expected}, got {checkpoint.get('control_metadata')}"
        )
    saved_preprocessing = checkpoint.get("config", {}).get("preprocessing", {})
    runtime_preprocessing = preprocessing.to_metadata()
    identity_keys = (
        "model_id", "revision", "weights_format", "num_classes",
        "label_order_sha256", "preprocessing_version",
    )
    mismatches = {
        key: (saved_preprocessing.get(key), runtime_preprocessing.get(key))
        for key in identity_keys
        if saved_preprocessing.get(key) != runtime_preprocessing.get(key)
    }
    if mismatches:
        raise ValueError(f"Checkpoint/preprocessing mismatch: {mismatches}")
    manifest = json.loads(Path(args.preprocessing_manifest).read_text(encoding="utf-8"))
    provenance = {
        **manifest.get("preprocessing", {}),
        "weights_sha256": manifest.get("weights_sha256"),
        "processor_sha256": manifest.get("processor", {}).get("sha256"),
    }
    provenance_keys = identity_keys + ("weights_sha256", "processor_sha256")
    provenance_mismatches = {
        key: (saved_preprocessing.get(key), provenance.get(key))
        for key in provenance_keys
        if saved_preprocessing.get(key) != provenance.get(key)
    }
    if provenance_mismatches:
        raise ValueError(f"Checkpoint/manifest mismatch: {provenance_mismatches}")
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "run_metadata.json").write_text(
        json.dumps({
            "arguments": vars(args),
            "preprocessing": preprocessing.to_metadata(),
            "control_metadata": expected,
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "checkpoint_step": checkpoint.get("global_step"),
            "checkpoint_best_val_loss": checkpoint.get("best_val_loss"),
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    model.control_adapter.load_state_dict(checkpoint["model"])
    model.zero_convs.load_state_dict(checkpoint["zero_convs"])
    model.eval()
    pipeline.model = model.wan
    if not args.offload:
        pipeline.model.to("cuda")

    image = Image.open(args.ref_image).convert("RGB")
    model._control_signal = None
    _seed_everything(args.seed)
    with torch.no_grad():
        base_video = _generate(pipeline, args, image)
    describe_video_tensor("base", base_video)
    base_frames = tensor_to_frames(base_video)
    save_rgb_video(base_frames, output_dir / "base.mp4", fps=args.fps)
    save_debug_frames(base_frames, output_dir, "base")
    del base_video

    with torch.no_grad():
        model._control_signal = model.control_adapter(control_features)
    visualization = mask_to_rgb(mask, base_frames.shape[1:3])

    for strength in args.control_strengths:
        model._control_strength = float(strength)
        _seed_everything(args.seed)
        with torch.no_grad():
            controlled_video = _generate(pipeline, args, image)
        controlled_frames = tensor_to_frames(controlled_video)
        tag = strength_tag(strength)
        controlled_path = output_dir / f"controlled_strength_{tag}.mp4"
        comparison_path = output_dir / f"comparison_strength_{tag}.mp4"
        save_rgb_video(controlled_frames, controlled_path, fps=args.fps)
        save_debug_frames(controlled_frames, output_dir, f"controlled_{tag}")
        save_comparison(
            visualization,
            base_frames,
            controlled_frames,
            comparison_path,
            args.fps,
        )
        del controlled_video
        torch.cuda.empty_cache()
        gc.collect()

    model._control_signal = None
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()

