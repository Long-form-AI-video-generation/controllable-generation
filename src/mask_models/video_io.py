"""Small RGB video I/O helpers for mask inference diagnostics."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


def describe_video_tensor(name: str, video_tensor: torch.Tensor) -> None:
    value = video_tensor.detach().float().nan_to_num()
    print(
        f"{name}: shape={tuple(video_tensor.shape)} "
        f"dtype={video_tensor.dtype} device={video_tensor.device} "
        f"min={value.min().item():.4f} max={value.max().item():.4f} "
        f"mean={value.mean().item():.4f}"
    )


def video_to_bcthw(video_tensor: torch.Tensor) -> torch.Tensor:
    if video_tensor.dim() == 5:
        if video_tensor.shape[1] in (1, 3):
            return video_tensor
        if video_tensor.shape[2] in (1, 3):
            return video_tensor.permute(0, 2, 1, 3, 4).contiguous()
    if video_tensor.dim() == 4:
        if video_tensor.shape[0] in (1, 3):
            return video_tensor.unsqueeze(0)
        if video_tensor.shape[1] in (1, 3):
            return video_tensor.permute(1, 0, 2, 3).unsqueeze(0).contiguous()
    raise ValueError(
        f"Unexpected video tensor shape: {tuple(video_tensor.shape)}"
    )


def tensor_to_frames(video_tensor: torch.Tensor) -> np.ndarray:
    value = video_to_bcthw(video_tensor)[0].float()
    value = (value.clamp(-1, 1) + 1.0) * 127.5
    return (
        value.permute(1, 2, 3, 0)
        .cpu()
        .numpy()
        .astype(np.uint8)
    )


def save_rgb_video(frames: np.ndarray, path: Path, fps: int = 16) -> None:
    frames = np.asarray(frames)
    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected RGB [T,H,W,3], got {frames.shape}")
    pad_h = frames.shape[1] % 2
    pad_w = frames.shape[2] % 2
    if pad_h or pad_w:
        frames = np.pad(
            frames,
            ((0, 0), (0, pad_h), (0, pad_w), (0, 0)),
            mode="edge",
        )
    frames = np.ascontiguousarray(frames)

    try:
        import imageio.v2 as imageio

        imageio.mimsave(
            str(path),
            list(frames),
            fps=fps,
            codec="libx264",
            quality=8,
            macro_block_size=1,
            ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        )
    except Exception as error:
        fallback = path.with_suffix(".avi")
        print(f"H.264 encoding failed ({error}); writing {fallback}")
        height, width = frames.shape[1:3]
        writer = cv2.VideoWriter(
            str(fallback),
            cv2.VideoWriter_fourcc(*"MJPG"),
            fps,
            (width, height),
        )
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()


def save_debug_frames(
    frames: np.ndarray,
    output_dir: Path,
    prefix: str,
) -> None:
    for index in sorted({0, len(frames) // 2, len(frames) - 1}):
        Image.fromarray(frames[index]).save(
            output_dir / f"{prefix}_frame_{index:03d}.png"
        )

