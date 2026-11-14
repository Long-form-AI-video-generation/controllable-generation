#!/usr/bin/env python3
"""Verify an audited shot bundle, full dataset loading and untrained model compatibility.

Does not update weights or run WAN. Randomly initialized legacy components only
check the dimensional interface; their outputs are not useful trained controls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from scripts.pose.prepare_video_pilot import build_sequence
from src.data.frame_sampling import resolve_frame_interval


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def check_artifacts(folder, videos_dir, annotations_path, split_path):
    """Check source identity and rebuild the tensor before any model is loaded."""
    folder = Path(folder)
    manifest = json.loads((folder / 'manifest.json').read_text())
    if manifest['training_eligible'] is not False:
        raise ValueError('this command accepts diagnostic bundles only')
    vid, shot = manifest['video_id'], manifest['shot_id']
    for value in (vid, shot):
        if not isinstance(value, str) or not value or value in ('.', '..') or '/' in value or '\\' in value:
            raise ValueError('invalid video or shot identifier')
    paths = {
        'source_video_sha256': Path(videos_dir) / f'{vid}.mp4',
        'annotations_sha256': Path(annotations_path),
        'split_manifest_sha256': Path(split_path),
        'raw_predictions_sha256': folder / 'raw_predictions.json',
    }
    for key, path in paths.items():
        if file_hash(path) != manifest[key]:
            raise ValueError(f'hash mismatch: {key}')
    split = json.loads(Path(split_path).read_text())
    sets = {key: {str(v) for v in split[key]} for key in ('train', 'val', 'test')}
    if any(sets[a] & sets[b] for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test'))):
        raise ValueError('overlapping frozen splits')
    if manifest['partition'] not in ('train', 'val') or vid not in sets[manifest['partition']]:
        raise ValueError('source is not in the declared development partition')
    annotations = json.loads(Path(annotations_path).read_text())
    matches = [row for row in annotations if str(row['video_id']) == vid and str(row['shot_id']) == shot]
    if len(matches) != 1:
        raise ValueError('annotation must match exactly one shot')
    ann = matches[0]
    interval = resolve_frame_interval(int(ann['segment_start_frame']), int(ann['segment_end_frame']),
                                      manifest['source_frame_count'])
    if list(interval) != manifest['source_interval_half_open']:
        raise ValueError('annotation and saved frame interval differ')
    rows = json.loads((folder / 'raw_predictions.json').read_text())
    shape = manifest['shape_cthw']
    if len(shape) != 4 or shape[0] != 3 or any(type(d) is not int or d <= 0 for d in shape):
        raise ValueError('invalid control shape')
    rebuilt, indices, selected = build_sequence(rows, *interval, shape[1], shape[2], shape[3],
                                                 manifest['sigma_pixels'])
    if indices != manifest['source_frame_indices']:
        raise ValueError('stored frame indices do not match shared sampler')
    if selected[0]['model_info']['config_sha256'] != manifest['extractor_config_sha256']:
        raise ValueError('extractor configuration differs')
    path = folder / 'controls' / vid / f'shot_{shot}_controls_encoded.npz'
    if 'control_file_sha256' in manifest and file_hash(path) != manifest['control_file_sha256']:
        raise ValueError('hash mismatch: control_file_sha256')
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {'pose_encoded'}:
            raise ValueError('unexpected control archive keys')
        stored = archive['pose_encoded']
        if stored.shape != tuple(shape) or not np.isfinite(stored).all():
            raise ValueError('invalid control tensor')
        if not np.allclose(stored, rebuilt, atol=1e-6, rtol=1e-6):
            raise ValueError('control tensor does not match retained raw predictions')
    return manifest, path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'videos-dir', 'annotations', 'split-manifest', 'report'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    manifest, control_path = check_artifacts(args.bundle, args.videos_dir, args.annotations, args.split_manifest)
    print('PASS: source hashes, split, indices and reconstructed controls.', flush=True)

    import cv2
    import torch
    from src.data.dataset import ControllableVideoDataset
    from src.depth_models.encoders import PoseEncoder
    from src.depth_models.control_adapter import ControlAdapter

    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')
    capture = cv2.VideoCapture(str(args.videos_dir / (manifest['video_id'] + '.mp4')))
    try:
        if not capture.isOpened() or int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) != manifest['source_frame_count']:
            raise ValueError('source frame count differs from manifest')
    finally:
        capture.release()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    _, frames, height, width = manifest['shape_cthw']
    dataset = ControllableVideoDataset(encoded_controls_dir=str(args.bundle / 'controls'),
        videos_dir=str(args.videos_dir), annotations_path=str(args.annotations),
        split_manifest_path=str(args.split_manifest), split=manifest['partition'],
        num_frames=frames, resolution=(width, height), control_key='pose_encoded', strict=True)
    if len(dataset) != 1:
        raise ValueError(f'expected one diagnostic shot, found {len(dataset)}')
    sample = dataset[0]
    if str(sample['video_id']) != manifest['video_id'] or str(sample['shot_id']) != manifest['shot_id']:
        raise ValueError('loaded a different shot')
    if not isinstance(sample['caption'], str) or not sample['caption'].strip():
        raise ValueError('expected a nonempty caption')
    pose = sample['controls']['pose_encoded'].unsqueeze(0).float().to(args.device)
    video = sample['video']
    if tuple(pose.shape) != (1, 3, frames, height, width) or tuple(video.shape) != (3, frames, height, width):
        raise ValueError('unexpected dataset shapes')
    for tensor in (pose, video):
        if not torch.isfinite(tensor).all() or tensor.min() < 0 or tensor.max() > 1:
            raise ValueError('dataset values must be finite and in [0,1]')
    print('PASS: complete dataset loader, video/control shapes and ranges.', flush=True)
    encoder = PoseEncoder(out_channels=256).to(args.device).eval()
    adapter = ControlAdapter(control_dim=256, hidden_dim=512, dit_dim=2048, num_controls=1).to(args.device).eval()
    started = time.perf_counter()
    with torch.inference_mode():
        features = encoder(pose)
        signal = adapter({'pose_encoded': features})
    if tuple(features.shape) != (1, 256, frames, height // 2, width // 2):
        raise ValueError(f'unexpected encoder shape: {tuple(features.shape)}')
    if tuple(signal.shape) != (1, frames * 16 * 16, 2048):
        raise ValueError(f'unexpected adapter shape: {tuple(signal.shape)}')
    if not torch.isfinite(features).all() or not torch.isfinite(signal).all():
        raise ValueError('model output is not finite')
    report = {
        'purpose': 'untrained legacy pose encoder and adapter dimensional compatibility',
        'training_eligible': False, 'wan_forward_tested': False, 'weights_updated': False,
        'video_id': manifest['video_id'], 'shot_id': manifest['shot_id'],
        'frame_indices': manifest['source_frame_indices'], 'seed': 0, 'device': args.device,
        'torch_version': torch.__version__, 'numpy_version': np.__version__, 'opencv_version': cv2.__version__,
        'pose_shape': list(pose.shape), 'encoder_shape': list(features.shape), 'adapter_shape': list(signal.shape),
        'finite_outputs': True, 'model_forward_seconds': time.perf_counter() - started,
        'bundle_manifest_sha256': file_hash(args.bundle / 'manifest.json'),
        'control_file_sha256': file_hash(control_path),
        'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'limitations': 'Random weights; no quality, gradient, WAN injection or training eligibility claim.',
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print('PASS: untrained pose encoder -> adapter; finite outputs.')
    print('Encoder shape:', tuple(features.shape))
    print('Adapter shape:', tuple(signal.shape))
    print('Saved report:', args.report.resolve())


if __name__ == '__main__':
    main()
