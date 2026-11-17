#!/usr/bin/env python3
"""Create one annotated-shot pose-map fixture for a full loader diagnostic.

The output contains all detected people and uncalibrated in-frame keypoints.
It is not a validated pose record or approved training control.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.data.frame_sampling import resolve_frame_interval, select_frame_indices
from src.pose_models.extraction.dwpose_onnx import DWPoseONNXConfig, create_dwpose_onnx_extractor
from scripts.pose.prepare_video_pilot import build_sequence
from scripts.pose.bundle_io import code_provenance, file_hash, publish_bundle, write_json


def load_inputs(annotations_path, split_path):
    annotations = json.loads(Path(annotations_path).read_text())
    split = json.loads(Path(split_path).read_text())
    sets = {}
    for name in ('train', 'val', 'test'):
        if not isinstance(split.get(name), list):
            raise ValueError('split must contain train/val/test video-ID lists')
        sets[name] = {str(value) for value in split[name]}
    if any(sets[a] & sets[b] for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test'))):
        raise ValueError('overlapping frozen splits')
    return annotations, sets


def select_shot(annotations, split, video_id, shot_id):
    for value in (video_id, shot_id):
        if not isinstance(value, str) or not value or value in ('.', '..') or '/' in value or '\\' in value:
            raise ValueError('video and shot IDs must be plain string identifiers')
    matches = [item for item in annotations if str(item.get('video_id')) == video_id
               and str(item.get('shot_id')) == shot_id]
    if len(matches) != 1:
        raise ValueError(f'expected exactly one matching annotation, found {len(matches)}')
    partitions = [name for name in ('train', 'val', 'test') if video_id in split[name]]
    if len(partitions) != 1 or partitions[0] not in ('train', 'val'):
        raise ValueError(f'expected one development partition, found {partitions}')
    return matches[0], partitions[0]


def prepare_shot(*, video, video_id, shot_id, annotations_path, split_path,
                 output_dir, extractor, num_frames=8, size=128, sigma=2.0,
                 provenance=None, batch_identity=None, cv2_module=None,
                 expected_source_hashes=None):
    """Prepare one diagnostic bundle; shared extractors support bounded batches."""
    if type(num_frames) is not int or type(size) is not int or num_frames <= 0 or size <= 0:
        raise ValueError('frame count and size must be positive integers')
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError('sigma must be finite and positive')
    if cv2_module is None:
        import cv2 as cv2_module
    cv2 = cv2_module
    video, output_dir = Path(video), Path(output_dir)
    annotations_path, split_path = Path(annotations_path), Path(split_path)
    observed_code = code_provenance(require_clean=False)
    provenance = provenance if provenance is not None else code_provenance()
    if provenance != observed_code:
        raise ValueError('source code changed since run initialization')
    if output_dir.exists():
        raise FileExistsError(output_dir)
    source_hashes = {'source_video_sha256': file_hash(video),
                     'annotations_sha256': file_hash(annotations_path),
                     'split_manifest_sha256': file_hash(split_path)}
    if expected_source_hashes is not None and source_hashes != expected_source_hashes:
        raise ValueError('shot source hashes differ from immutable batch inputs')
    annotations, split = load_inputs(annotations_path, split_path)
    annotation, partition = select_shot(annotations, split, video_id, shot_id)
    if video.name != video_id + '.mp4':
        raise ValueError('video filename must match the annotated video ID')
    # Include decoding/inference failures in the retained attempt directory too.
    with publish_bundle(output_dir) as output:
        capture = cv2.VideoCapture(str(video))
        try:
            if not capture.isOpened():
                raise RuntimeError(f'could not open {video}')
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            start, end = resolve_frame_interval(int(annotation['segment_start_frame']),
                                                int(annotation['segment_end_frame']), frame_count)
            indices = [int(i) for i in select_frame_indices(start, end, num_frames)]
            observations = {}
            for index in dict.fromkeys(indices):
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, bgr = capture.read()
                if not ok or bgr is None:
                    raise RuntimeError(f'could not decode frame {index}')
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                observations[index] = extractor.extract_frame(rgb)
        finally:
            capture.release()
        rows = [{'source_frame_index': i, 'result': r} for i, r in observations.items()]
        maps, selected_indices, selected = build_sequence(rows, start, end, num_frames, size, size, sigma)
        control = output / 'controls' / video_id / f'shot_{shot_id}_controls_encoded.npz'
        control.parent.mkdir(parents=True)
        np.savez_compressed(control, pose_encoded=maps)
        raw_path = output / 'raw_predictions.json'
        write_json(raw_path, rows)
        manifest = {
            'purpose': 'annotated-shot loader diagnostic; not eligible for training',
            'training_eligible': False, 'video_id': video_id, 'shot_id': shot_id,
            'partition': partition, 'source_frame_count': frame_count,
            'source_interval_half_open': [start, end], 'source_frame_indices': selected_indices,
            **source_hashes, 'raw_predictions_sha256': file_hash(raw_path),
            'extractor_config_sha256': selected[0]['model_info']['config_sha256'],
            'model_info': selected[0]['model_info'],
            'code_commit': provenance['commit'], 'code_provenance': provenance,
            'shape_cthw': list(maps.shape), 'sigma_pixels': sigma,
            'control_file_sha256': file_hash(control),
            'resize_policy': 'direct_resize_pixel_centres_v1',
            'channels': ['head', 'torso', 'limbs'],
            'candidate_counts': [len(r['candidates']) for r in selected],
            'policy': 'all detections; all in-frame positions; no calibrated visibility, score filtering or tracking',
            'limitations': 'diagnostic only; potential false positives, hidden-joint guesses and mixed identities',
        }
        if batch_identity is not None:
            manifest['batch_identity'] = batch_identity
        write_json(output / 'manifest.json', manifest)
        # Check persisted bytes and source stability before publishing completion.
        from scripts.pose.check_preprocessing_pilot import check_artifacts
        check_artifacts(output, video.parent, annotations_path, split_path)
        if code_provenance(require_clean=False) != provenance:
            raise RuntimeError('source code changed during shot preparation')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('video', 'annotations', 'split-manifest', 'detector-model', 'pose-model', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('video-id', 'shot-id', 'detector-sha256', 'pose-sha256', 'onnxruntime-version', 'opencv-version'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--num-frames', type=int, default=8)
    parser.add_argument('--size', type=int, default=128)
    parser.add_argument('--sigma', type=float, default=2.0)
    parser.add_argument('--allow-dirty-code', action='store_true',
                        help='development only: record modified source identity')
    args = parser.parse_args()
    provenance = code_provenance(require_clean=not args.allow_dirty_code)
    config = DWPoseONNXConfig(detector_path=args.detector_model, pose_path=args.pose_model,
        detector_sha256=args.detector_sha256, pose_sha256=args.pose_sha256,
        expected_onnxruntime_version=args.onnxruntime_version,
        expected_opencv_version=args.opencv_version, providers=('CPUExecutionProvider',))
    extractor = create_dwpose_onnx_extractor(config)
    manifest = prepare_shot(video=args.video, video_id=args.video_id, shot_id=args.shot_id,
        annotations_path=args.annotations, split_path=args.split_manifest,
        output_dir=args.output_dir, extractor=extractor, num_frames=args.num_frames,
        size=args.size, sigma=args.sigma, provenance=provenance)
    print('Source frame indices:', manifest['source_frame_indices'])
    print('Output:', args.output_dir.resolve())
    print('DIAGNOSTIC ONLY: not eligible for training.')


if __name__ == '__main__':
    main()
