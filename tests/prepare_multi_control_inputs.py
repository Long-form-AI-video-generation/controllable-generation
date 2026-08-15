"""Create one immutable depth/Canny/mask input bundle without loading WAN."""

from __future__ import annotations

import argparse
import gc
import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.control_integration.control_artifacts import write_prepared_control_bundle
from src.control_integration.contracts import canonicalize_expert_names
from src.control_integration.preprocessing import (
    MidasConfig,
    build_matched_controls,
    decode_reference_frames,
    load_midas_local,
    preprocessing_identities,
)
from src.mask_models.labels import colorize_id_map
from src.mask_models.preprocessing import SegFormerMaskConfig, load_segformer
from src.sketch_models.preprocessing import CannyConfig


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_experts(values: list[str]) -> tuple[str, ...]:
    parts: list[str] = []
    for value in values:
        parts.extend(value.split("+"))
    return canonicalize_expert_names(parts)


def _control_panel(name: str, array: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    height, width = target_hw
    frame = array[0, 0, 0]
    if name == "depth":
        panel = cv2.applyColorMap(
            np.clip(frame * 255.0, 0, 255).astype(np.uint8),
            cv2.COLORMAP_PLASMA,
        )
        panel = cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)
    elif name == "canny":
        panel = np.repeat((frame * 255).astype(np.uint8)[..., None], 3, axis=-1)
    elif name == "mask":
        panel = colorize_id_map(frame)
    else:
        raise AssertionError(name)
    return cv2.resize(panel, (width, height), interpolation=cv2.INTER_NEAREST)


def _write_contact_sheet(
    output_dir: Path,
    sequence,
    controls: dict[str, np.ndarray],
) -> None:
    """Save a compact frame-zero coherence panel for operator review."""

    source = cv2.cvtColor(sequence.frames_bgr[0], cv2.COLOR_BGR2RGB)
    height, width = source.shape[:2]
    panels = [("source frame 0", source)]
    panels.extend((name, _control_panel(name, controls[name], (height, width))) for name in controls)
    labelled = []
    for label, panel in panels:
        result = panel.copy()
        cv2.rectangle(result, (0, 0), (width, 28), (0, 0, 0), -1)
        cv2.putText(result, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        labelled.append(result)
    from PIL import Image

    sheet = np.concatenate(labelled, axis=1)
    Image.fromarray(sheet).save(output_dir / "controls_contact_sheet.jpg")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref-video", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frame-num", type=int, required=True)
    parser.add_argument("--experts", nargs="+", default=["depth", "canny", "mask"])
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--control-height", type=int, default=128)
    parser.add_argument("--control-width", type=int, default=128)
    parser.add_argument("--canny-low", type=int, default=100)
    parser.add_argument("--canny-high", type=int, default=200)
    parser.add_argument("--midas-repo")
    parser.add_argument("--midas-weights")
    parser.add_argument("--midas-device", default="cuda:0")
    parser.add_argument("--segformer-device", default="cuda:0")
    parser.add_argument("--segformer-cache-dir")
    parser.add_argument("--ref-image")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    experts = _parse_experts(args.experts)
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(output)
    if args.frame_num <= 0 or args.frame_num % 4 != 1:
        raise ValueError("frame-num must be positive and satisfy 4n+1")
    if args.ref_image is not None and not Path(args.ref_image).is_file():
        raise FileNotFoundError(args.ref_image)

    sequence = decode_reference_frames(
        args.ref_video,
        frame_num=args.frame_num,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    target_size = (args.control_height, args.control_width)
    canny_config = CannyConfig(
        low_threshold=args.canny_low,
        high_threshold=args.canny_high,
        num_frames=args.frame_num,
        output_size=target_size,
        color_order="BGR",
    ) if "canny" in experts else None
    mask_config = SegFormerMaskConfig(
        num_frames=args.frame_num,
        output_size=target_size,
    ) if "mask" in experts else None
    midas_config = None
    controls: dict[str, np.ndarray] = {}
    if "canny" in experts:
        controls.update(
            build_matched_controls(
                sequence,
                experts=["canny"],
                canny_config=canny_config,
            )
        )
    if "depth" in experts:
        if not args.midas_repo or not args.midas_weights:
            raise ValueError("depth requires --midas-repo and --midas-weights")
        midas_config = MidasConfig(
            repo_dir=Path(args.midas_repo),
            weights_path=Path(args.midas_weights),
            output_size=target_size,
        )
        midas, transform = load_midas_local(midas_config, device=args.midas_device)
        try:
            controls.update(
                build_matched_controls(
                    sequence,
                    experts=["depth"],
                    midas_config=midas_config,
                    midas=midas,
                    midas_transform=transform,
                    midas_device=args.midas_device,
                )
            )
        finally:
            del midas, transform
            gc.collect()
            import torch
            torch.cuda.empty_cache()
    if "mask" in experts:
        mask_processor, mask_model = load_segformer(
            mask_config,
            cache_dir=args.segformer_cache_dir,
            local_files_only=True,
        )
        try:
            controls.update(
                build_matched_controls(
                    sequence,
                    experts=["mask"],
                    mask_config=mask_config,
                    mask_processor=mask_processor,
                    mask_model=mask_model,
                    mask_device=args.segformer_device,
                )
            )
        finally:
            del mask_processor, mask_model
            gc.collect()
            import torch
            torch.cuda.empty_cache()
    identities = preprocessing_identities(
        experts=experts,
        canny_config=canny_config,
        mask_config=mask_config,
        midas_config=midas_config,
        mask_cache_dir=args.segformer_cache_dir,
    )

    metadata: dict[str, object] = {
        "source_video": str(Path(args.ref_video).resolve()),
        "source_video_sha256": sequence.source_video_sha256,
        "source_frame_indices": list(sequence.source_indices),
        "padded_positions": list(sequence.padded_positions),
        "actual_frame_count": sequence.actual_frame_count,
        "interval": list(sequence.interval),
        "frame_num": args.frame_num,
        "shared_frame_color_order": "BGR",
        "preprocessing": identities,
    }
    if args.ref_image is not None:
        image_path = Path(args.ref_image)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"could not decode reference image {image_path}")
        metadata["reference_image"] = {
            "path": str(image_path.resolve()),
            "sha256": _file_sha256(image_path),
            "height": int(image.shape[0]),
            "width": int(image.shape[1]),
        }

    bundle = write_prepared_control_bundle(output, controls=controls, metadata=metadata)
    _write_contact_sheet(bundle.root, sequence, controls)
    print(f"Prepared controls: {bundle.root}")
    print(f"Experts: {', '.join(bundle.controls)}")
    print(f"Artifact SHA-256: {bundle.artifact_sha256}")


if __name__ == "__main__":
    main()
