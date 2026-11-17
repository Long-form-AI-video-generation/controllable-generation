#!/usr/bin/env python3
"""Bounded, resumable diagnostic pose preprocessing; never marks training eligibility."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
import uuid

from scripts.pose.bundle_io import code_provenance, exclusive_lock, file_hash, publish_bundle, write_json
from scripts.pose.check_preprocessing_pilot import check_artifacts
from scripts.pose.prepare_shot_pilot import load_inputs, prepare_shot, select_shot
from src.pose_models.extraction.dwpose_onnx import DWPoseONNXConfig, create_dwpose_onnx_extractor

PATH_KEYS = ('videos_dir', 'annotations', 'split_manifest', 'output_dir', 'detector_model', 'pose_model')
PIN_KEYS = ('detector_sha256', 'pose_sha256', 'onnxruntime_version', 'opencv_version')


def load_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    required = set(PATH_KEYS + PIN_KEYS) | {'shots', 'max_shots'}
    allowed = required | {'num_frames', 'size', 'sigma'}
    if not isinstance(config, dict) or required - config.keys() or config.keys() - allowed:
        raise ValueError('config has missing or unknown keys')
    for key in PATH_KEYS:
        if not isinstance(config[key], str) or not config[key]:
            raise ValueError(f'{key} must be a nonempty path')
        config[key] = str((path.parent / Path(config[key]).expanduser()).resolve())
    config.setdefault('num_frames', 8)
    config.setdefault('size', 128)
    config.setdefault('sigma', 2.0)
    for key in ('num_frames', 'size', 'max_shots'):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if type(config['sigma']) not in (int, float) or not math.isfinite(config['sigma']) or config['sigma'] <= 0:
        raise ValueError('sigma must be finite and positive')
    config['sigma'] = float(config['sigma'])
    shots = config['shots']
    if not isinstance(shots, list) or not 1 <= len(shots) <= config['max_shots']:
        raise ValueError('shots must be a nonempty list within max_shots')
    seen = set()
    for shot in shots:
        if not isinstance(shot, dict) or set(shot) != {'video_id', 'shot_id'}:
            raise ValueError('each shot needs exactly video_id and shot_id')
        if not all(isinstance(value, str) for value in shot.values()):
            raise ValueError('shot identifiers must be strings')
        pair = (shot['video_id'], shot['shot_id'])
        if pair in seen:
            raise ValueError('duplicate requested shot')
        seen.add(pair)
    # Validate pinned versions/hash formats even in dry-run mode.
    model_config(config)
    return config


def model_config(config):
    return DWPoseONNXConfig(detector_path=config['detector_model'], pose_path=config['pose_model'],
        detector_sha256=config['detector_sha256'], pose_sha256=config['pose_sha256'],
        expected_onnxruntime_version=config['onnxruntime_version'],
        expected_opencv_version=config['opencv_version'], providers=('CPUExecutionProvider',))


def batch_plan(config, provenance):
    hashes = {key: file_hash(config[key]) for key in ('annotations', 'split_manifest', 'detector_model', 'pose_model')}
    annotations, split = load_inputs(config['annotations'], config['split_manifest'])
    plan = []
    video_hashes = {}
    for shot in config['shots']:
        annotation, partition = select_shot(annotations, split, **shot)
        video = Path(config['videos_dir']) / (shot['video_id'] + '.mp4')
        if shot['video_id'] not in video_hashes:
            video_hashes[shot['video_id']] = file_hash(video)
        caption = annotation.get('narrative_caption') or annotation.get('descriptive_caption')
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError(f'{shot}: requires a meaningful annotation caption, not loader fallback')
        plan.append({**shot, 'partition': partition, 'caption': caption})
    for key, expected in hashes.items():
        if file_hash(config[key]) != expected:
            raise RuntimeError(f'{key} changed while planning batch')
    for key in ('detector', 'pose'):
        if hashes[key + '_model'] != config[key + '_sha256']:
            raise ValueError(f'{key} model hash mismatch')
    identity = {'schema': 'pose-diagnostic-batch-v2', 'config': config, 'input_hashes': hashes,
                'source_video_sha256': video_hashes,
                'code_provenance': provenance, 'training_eligible': False}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return identity, fingerprint, plan


def bundle_path(root, shot):
    return root / 'bundles' / shot['video_id'] / shot['shot_id']


def verify_completed(path, config, fingerprint, shot):
    manifest, control = check_artifacts(path, config['videos_dir'], config['annotations'], config['split_manifest'])
    if manifest.get('batch_identity') != fingerprint:
        raise ValueError('completed bundle belongs to different settings or code')
    if (manifest['video_id'], manifest['shot_id']) != (shot['video_id'], shot['shot_id']):
        raise ValueError('completed bundle belongs to a different shot')
    if manifest['shape_cthw'] != [3, config['num_frames'], config['size'], config['size']]:
        raise ValueError('completed bundle shape differs from batch config')
    if manifest['sigma_pixels'] != config['sigma']:
        raise ValueError('completed bundle sigma differs from batch config')
    return manifest, control


def run_batch(config, *, resume=False, allow_dirty_code=False, dry_run=False,
              extractor_factory=create_dwpose_onnx_extractor, cv2_module=None):
    provenance = code_provenance(require_clean=not allow_dirty_code)
    identity, fingerprint, plan = batch_plan(config, provenance)
    if dry_run:
        return {'dry_run': True, 'identity': fingerprint, 'shots': plan, 'training_eligible': False}
    root = Path(config['output_dir'])
    # The lock sits outside root, so even initialization is serialized.
    with exclusive_lock(root.parent / f'.{root.name}.batch.lock'):
        if root.exists():
            if not resume:
                raise FileExistsError(f'{root}: use --resume to verify existing results')
            saved = json.loads((root / 'batch.json').read_text())
            if saved != identity:
                raise ValueError('batch identity changed; use a fresh output directory')
        else:
            with publish_bundle(root, completion_file='batch.json') as initial:
                write_json(initial / 'batch.json', identity)
        (root / 'attempts').mkdir(exist_ok=True)
        run_id = uuid.uuid4().hex
        summary = {'run_id': run_id, 'batch_identity': fingerprint, 'training_eligible': False,
                   'started_utc': datetime.now(timezone.utc).isoformat(), 'shots': [],
                   'completed': 0, 'skipped': 0, 'failed': 0, 'interrupted': False}
        extractor = None
        started_batch = time.perf_counter()
        try:
            for shot in plan:
                started = time.perf_counter()
                target = bundle_path(root, shot)
                result = {**shot}
                try:
                    # Do not allow inputs or code to drift midway through the run.
                    if code_provenance(require_clean=not allow_dirty_code) != provenance:
                        raise RuntimeError('source code changed during batch')
                    for key, expected in identity['input_hashes'].items():
                        if file_hash(config[key]) != expected:
                            raise RuntimeError(f'{key} changed during batch')
                    expected_sources = {
                        'source_video_sha256': identity['source_video_sha256'][shot['video_id']],
                        'annotations_sha256': identity['input_hashes']['annotations'],
                        'split_manifest_sha256': identity['input_hashes']['split_manifest'],
                    }
                    video = Path(config['videos_dir']) / (shot['video_id'] + '.mp4')
                    if file_hash(video) != expected_sources['source_video_sha256']:
                        raise RuntimeError(f'{shot["video_id"]}: source video changed during batch')
                    if target.exists() or target.is_symlink():
                        manifest, _ = verify_completed(target, config, fingerprint, shot)
                        result['status'] = 'skipped'
                    else:
                        if extractor is None:
                            extractor = extractor_factory(model_config(config))
                        manifest = prepare_shot(video=video,
                            video_id=shot['video_id'], shot_id=shot['shot_id'],
                            annotations_path=config['annotations'], split_path=config['split_manifest'],
                            output_dir=target, extractor=extractor, num_frames=config['num_frames'],
                            size=config['size'], sigma=config['sigma'], provenance=provenance,
                            batch_identity=fingerprint, cv2_module=cv2_module,
                            expected_source_hashes=expected_sources)
                        result['status'] = 'completed'
                    result['candidate_counts'] = manifest['candidate_counts']
                    result['source_frame_indices'] = manifest['source_frame_indices']
                    result['output_bytes'] = sum(p.stat().st_size for p in target.rglob('*') if p.is_file())
                    summary[result['status']] += 1
                except Exception as error:
                    result.update(status='failed', error_type=type(error).__name__, error=str(error))
                    summary['failed'] += 1
                except BaseException as error:
                    result.update(status='interrupted', error_type=type(error).__name__, error=str(error))
                    summary['interrupted'] = True
                    raise
                finally:
                    result['seconds'] = time.perf_counter() - started
                    summary['shots'].append(result)
                    # Immutable per-shot receipts survive a later kill of this process.
                    write_json(root / 'attempts' / f'{run_id}-{len(summary["shots"]):04d}.json', result)
        finally:
            summary['seconds'] = time.perf_counter() - started_batch
            summary['finished_utc'] = datetime.now(timezone.utc).isoformat()
            summary['unprocessed'] = len(plan) - len(summary['shots'])
            write_json(root / 'attempts' / f'{run_id}-summary.json', summary)
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--allow-dirty-code', action='store_true', help='development only; records source changes')
    args = parser.parse_args()
    result = run_batch(load_config(args.config), resume=args.resume,
                       dry_run=args.dry_run, allow_dirty_code=args.allow_dirty_code)
    print(json.dumps(result, indent=2, allow_nan=False))
    if result.get('failed', 0):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
