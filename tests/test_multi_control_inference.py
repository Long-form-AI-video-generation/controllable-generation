"""Run base and additive multi-control inference through one official WAN model.

This entry point deliberately attaches the integration controller to the
official pipeline DiT.  It never creates a second WAN backbone or swaps the
pipeline model, so all requested controls are fused at the same injection
sites during one normal WAN generation call.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Wan2.2"))

import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, WAN_CONFIGS

from src.control_integration.checkpoint_loading import load_requested_experts
from src.control_integration.control_artifacts import load_prepared_control_bundle
from src.control_integration.hook_controller import MultiControlHookController
from src.control_integration.inference_config import build_inference_config
from src.control_integration.preprocessing import controls_to_adapter_tensors
from src.mask_models.labels import colorize_id_map
from src.sketch_models.video_io import (
    describe_video_tensor,
    save_debug_frames,
    save_rgb_video,
    tensor_to_frames,
)


def _combination_tag(names: tuple[str, ...]) -> str:
    return "_".join(names)


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _generate(pipeline: object, args: argparse.Namespace, image: Image.Image):
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


def _control_panel(
    controls: Mapping[str, np.ndarray],
    index: int,
    target_hw: tuple[int, int],
) -> np.ndarray:
    """Show the requested controls side-by-side for a comparison video."""

    height, width = target_hw
    panels: list[np.ndarray] = []
    for name, array in controls.items():
        frame = array[0, 0, index]
        if name == "mask":
            rgb = colorize_id_map(frame)
        else:
            value = frame.astype(np.float32)
            if name == "depth":
                low, high = float(value.min()), float(value.max())
                value = (value - low) / max(high - low, 1e-6)
            rgb = np.repeat((value * 255).clip(0, 255).astype(np.uint8)[..., None], 3, axis=-1)
        panels.append(cv2.resize(rgb, (width, height), interpolation=cv2.INTER_NEAREST))
    return np.concatenate(panels, axis=1)


def _save_comparison(
    controls: Mapping[str, np.ndarray],
    base_frames: np.ndarray,
    controlled_frames: np.ndarray,
    path: Path,
    fps: int,
) -> None:
    count = min(len(base_frames), len(controlled_frames))
    height, width = base_frames.shape[1:3]
    labels = ("Prepared controls", "Base WAN", "Multi-control")
    output: list[np.ndarray] = []
    for index in range(count):
        control = _control_panel(controls, index, (height, width))
        control = cv2.resize(control, (width, height), interpolation=cv2.INTER_NEAREST)
        panels = (control, base_frames[index], controlled_frames[index])
        labelled: list[np.ndarray] = []
        for label, frame in zip(labels, panels):
            frame = frame.copy()
            cv2.rectangle(frame, (0, 0), (width, 30), (0, 0, 0), -1)
            cv2.putText(frame, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
            labelled.append(frame)
        output.append(np.concatenate(labelled, axis=1))
    save_rgb_video(np.stack(output), path, fps=fps)


def _write_metadata(path: Path, data: Mapping[str, object]) -> None:
    """Atomically publish run state so incomplete output cannot look final."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent) as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref-image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--prepared-controls-dir", required=True)
    parser.add_argument("--depth-checkpoint")
    parser.add_argument("--canny-checkpoint")
    parser.add_argument("--mask-checkpoint")
    parser.add_argument("--wan-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--size", default="480*832")
    parser.add_argument("--frame-num", type=int, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument(
        "--control-combinations", nargs="+", default=["depth+canny+mask"],
        help="One or more combinations such as depth+canny or depth+canny+mask.",
    )
    parser.add_argument("--depth-strength", type=float, default=0.5)
    parser.add_argument("--canny-strength", type=float, default=0.5)
    parser.add_argument("--mask-strength", type=float, default=0.5)
    parser.add_argument("--combined-ratio-cap", type=float, default=0.1)
    parser.add_argument(
        "--controller-device",
        default="cuda:1",
        help="Device for adapters and zero-convs; WAN remains on its pipeline device.",
    )
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--offload", action="store_true", default=True)
    parser.add_argument("--no-offload", dest="offload", action="store_false")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    strengths = {
        "depth": args.depth_strength,
        "canny": args.canny_strength,
        "mask": args.mask_strength,
    }
    config = build_inference_config(
        frame_num=args.frame_num,
        steps=args.steps,
        fps=args.fps,
        size=args.size,
        control_combinations=args.control_combinations,
        strengths=strengths,
        combined_ratio_cap=args.combined_ratio_cap,
    )
    active_experts = tuple(config.strengths)
    bundle = load_prepared_control_bundle(
        args.prepared_controls_dir,
        expected_experts=active_experts,
        expected_frame_num=config.frame_num,
    )
    checkpoint_paths = {
        name: getattr(args, f"{name}_checkpoint")
        for name in active_experts
    }
    missing = [name for name, path in checkpoint_paths.items() if not path]
    if missing:
        raise ValueError(f"missing checkpoint paths for requested experts: {missing}")
    experts = load_requested_experts(checkpoint_paths)

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if not Path(args.ref_image).is_file():
        raise FileNotFoundError(args.ref_image)
    output_dir.mkdir(parents=True)
    metadata_path = output_dir / "run_metadata.json"
    metadata: dict[str, object] = {
        "status": "running",
        "arguments": vars(args),
        "prepared_controls": {
            "directory": str(bundle.root),
            "sha256": bundle.artifact_sha256,
            "metadata": bundle.metadata,
        },
        "experts": {
            name: {
                "checkpoint": str(expert.checkpoint_path),
                "checkpoint_sha256": expert.checkpoint_sha256,
                "global_step": expert.global_step,
                "best_val_loss": expert.best_val_loss,
            }
            for name, expert in experts.items()
        },
    }
    _write_metadata(metadata_path, metadata)

    try:
        pipeline = wan.WanTI2V(
            config=WAN_CONFIGS["ti2v-5B"], checkpoint_dir=args.wan_dir,
            device_id=0, rank=0, t5_cpu=True,
        )
        if not args.offload:
            pipeline.model.to("cuda")
        image = Image.open(args.ref_image).convert("RGB")

        # Make the matched no-control baseline before any expert reaches a GPU.
        # This is both the intended comparison and an independent WAN health check.
        _seed_everything(args.seed)
        with torch.inference_mode():
            base_video = _generate(pipeline, args, image)
        describe_video_tensor("base", base_video)
        base_frames = tensor_to_frames(base_video)
        save_rgb_video(base_frames, output_dir / "base.mp4", fps=config.fps)
        save_debug_frames(base_frames, output_dir, "base")
        del base_video
        torch.cuda.empty_cache()

        with MultiControlHookController(pipeline.model, experts).to(args.controller_device) as controller:
            controller.eval()
            controls = controls_to_adapter_tensors(
                bundle.controls,
                device=args.controller_device,
            )
            adapter_signals = controller.compute_adapter_signals(controls)
            # CUDA kernels are asynchronous.  Synchronize the expert device
            # here so any adapter failure is reported at its actual source,
            # rather than later in WAN's next GPU-0 operation.
            torch.cuda.synchronize(args.controller_device)
            del controls
            torch.cuda.empty_cache()

            for combination in config.combinations:
                tag = _combination_tag(combination)
                controller.activate_controls(
                    enabled_experts=combination,
                    adapter_signals=adapter_signals,
                    strengths=config.strengths,
                    control_artifact_sha256=bundle.artifact_sha256,
                    combined_ratio_cap=config.combined_ratio_cap,
                    diagnostics_enabled=args.diagnostics,
                )
                _seed_everything(args.seed)
                with torch.inference_mode():
                    controlled_video = _generate(pipeline, args, image)
                describe_video_tensor(f"controlled_{tag}", controlled_video)
                controlled_frames = tensor_to_frames(controlled_video)
                save_rgb_video(controlled_frames, output_dir / f"controlled_{tag}.mp4", fps=config.fps)
                save_debug_frames(controlled_frames, output_dir, f"controlled_{tag}")
                _save_comparison(bundle.controls, base_frames, controlled_frames, output_dir / f"comparison_{tag}.mp4", config.fps)
                if args.diagnostics:
                    metadata.setdefault("diagnostics", {})[tag] = {
                        "generation_id": controller.diagnostics().generation_id,
                        "wan_grid": controller.diagnostics().wan_grid,
                        "layers": list(controller.diagnostics().by_layer),
                    }
                controller.deactivate_controls()
                del controlled_video
                torch.cuda.empty_cache()
                gc.collect()
    except BaseException as error:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(error).__name__}: {error}"
        _write_metadata(metadata_path, metadata)
        raise
    metadata["status"] = "complete"
    _write_metadata(metadata_path, metadata)
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
