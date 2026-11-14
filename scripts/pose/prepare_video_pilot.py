#!/usr/bin/env python3
"""Render a diagnostic 3-channel pose sequence from a saved video smoke run.

This is a data-path pilot, not a training dataset: it includes every detected
person and treats in-frame coordinates as estimates, not verified visibility.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from src.data.frame_sampling import select_frame_indices
from src.pose_models.extraction.dwpose import CANONICAL_JOINT_NAMES


GROUPS = ((0, 1, 2, 3, 4), (5, 6, 11, 12), (7, 8, 9, 10, 13, 14, 15, 16))


def render_pose(result, height, width, sigma):
    """Return grouped Gaussian keypoint maps in source-frame coordinates."""
    if (not isinstance(height, int) or not isinstance(width, int)
            or height <= 0 or width <= 0 or not np.isfinite(sigma) or sigma <= 0):
        raise ValueError('height, width and sigma must be positive')
    source_h, source_w = result['height'], result['width']
    if source_h <= 1 or source_w <= 1:
        raise ValueError('source dimensions must exceed one pixel')
    yy, xx = np.ogrid[:height, :width]
    maps = np.zeros((3, height, width), dtype=np.float32)
    for person in result['candidates']:
        if person['joint_names'] != CANONICAL_JOINT_NAMES:
            raise ValueError('unexpected joint ordering')
        if len(person['xy_pixels']) != 17:
            raise ValueError('expected 17 joint positions')
        for channel, indices in enumerate(GROUPS):
            for joint in indices:
                point = person['xy_pixels'][joint]
                if point is None:
                    continue
                x, y = map(float, point)
                if not np.isfinite([x, y]).all() or not (0 <= x < source_w and 0 <= y < source_h):
                    raise ValueError('joint outside source frame')
                # Match pixel-centre scaling used by OpenCV's direct resize.
                x = (x + 0.5) * width / source_w - 0.5
                y = (y + 0.5) * height / source_h - 0.5
                bump = np.exp(-((xx-x)**2 + (yy-y)**2) / (2*sigma**2))
                maps[channel] = np.maximum(maps[channel], bump)
    return maps


def build_sequence(rows, start, end, count, height, width, sigma):
    if start < 0 or end <= start or count <= 0:
        raise ValueError('invalid frame interval or count')
    indices = [int(i) for i in select_frame_indices(start, end, count)]
    by_index = {}
    for row in rows:
        index = row['source_frame_index']
        if index in by_index:
            raise ValueError(f'duplicate source frame {index}')
        by_index[index] = row['result']
    missing = sorted(set(indices) - set(by_index))
    if missing:
        raise ValueError(f'missing source frames: {missing}')
    chosen = [by_index[i] for i in indices]
    sizes = {(r['width'], r['height']) for r in chosen}
    hashes = {r['model_info']['config_sha256'] for r in chosen}
    if len(sizes) != 1 or len(hashes) != 1:
        raise ValueError('selected frames have inconsistent dimensions or extractor configuration')
    maps = np.stack([render_pose(r, height, width, sigma) for r in chosen], axis=1)
    return maps, indices, chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=48, help='exclusive source frame index')
    parser.add_argument('--num-frames', type=int, default=8)
    parser.add_argument('--height', type=int, default=128)
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--sigma', type=float, default=2.0)
    args = parser.parse_args()
    raw = args.results.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    maps, indices, selected = build_sequence(rows, args.start, args.end, args.num_frames, args.height, args.width, args.sigma)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output_dir / 'pose_pilot.npz', pose_encoded=maps)
    (args.output_dir / 'selected_predictions.json').write_text(json.dumps([
        {'source_frame_index': i, 'result': result} for i, result in zip(indices, selected)
    ], indent=2, allow_nan=False) + '\n')
    manifest = {
        'purpose': 'diagnostic alignment and loading pilot; not training eligible',
        'training_eligible': False,
        'input_results': str(args.results.resolve()),
        'input_results_sha256': hashlib.sha256(raw).hexdigest(),
        'extractor_config_sha256': selected[0]['model_info']['config_sha256'],
        'source_size_wh': [selected[0]['width'], selected[0]['height']],
        'source_interval_half_open': [args.start, args.end],
        'frame_indices': indices,
        'shape_cthw': list(maps.shape),
        'channels': ['head', 'torso', 'limbs'],
        'sigma_pixels': args.sigma,
        'candidate_counts': [len(r['candidates']) for r in selected],
        'policy': 'all detected candidates; all in-frame points; no score filter or identity tracking',
        'limitations': 'No calibrated visibility or confidence; hidden joints may be wrong. No pose training contract validated.',
    }
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Frame indices:', indices)
    print('Pose shape [C,T,H,W]:', maps.shape)
    print('Candidate counts:', manifest['candidate_counts'])
    print('Saved:', args.output_dir.resolve())
    print('DIAGNOSTIC ONLY: not eligible for training.')


if __name__ == '__main__':
    main()
