#!/usr/bin/env python3
"""Audit a bounded development batch's detector filtering without changing controls."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from scripts.pose.bundle_io import code_provenance, file_hash, publish_bundle, write_json
from scripts.pose.check_preprocessing_pilot import check_artifacts
from scripts.pose.prepare_batch import load_config, model_config
from src.pose_models.extraction.dwpose_onnx import (
    _decode_yolox_output, _nms_indices, create_dwpose_onnx_extractor,
)


def audit_detector(detector, image, top_k=20):
    """Capture one ordinary detect() call, then explain its exact filter stages.

    Sequential diagnostic use only: the detector's session is temporarily wrapped.
    All decoded anchors are returned separately for lossless NPZ storage. Top-K
    JSON/overlays are a truncated view, never a claim that other proposals do not exist.
    """
    if type(top_k) is not int or not 1 <= top_k <= 100:
        raise ValueError('top_k must be an integer in [1,100]')
    original_session = detector.session
    captured = []

    class RecordingSession:
        def run(self, names, feed):
            outputs = original_session.run(names, feed)
            captured.append(outputs)
            return outputs

    detector.session = RecordingSession()
    try:
        actual = detector.detect(image)
    finally:
        detector.session = original_session
    if len(captured) != 1 or not captured[0]:
        raise ValueError('expected exactly one detector ONNX call with an output')
    raw = np.asarray(captured[0][0])
    predictions = raw[0] if raw.ndim == 3 else raw
    boxes, all_scores = _decode_yolox_output(raw, detector.input_size_hw)
    scores = all_scores[:, 0]
    height, width = image.shape[:2]
    ratio = min(detector.input_size_hw[0] / height, detector.input_size_hw[1] / width)
    source_boxes = boxes.astype(np.float64) / ratio
    clipped = source_boxes.copy()
    clipped[:, [0, 2]] = np.clip(clipped[:, [0, 2]], 0, width)
    clipped[:, [1, 3]] = np.clip(clipped[:, [1, 3]], 0, height)
    valid = (clipped[:, 2] > clipped[:, 0]) & (clipped[:, 3] > clipped[:, 1])
    initial = np.flatnonzero(scores > detector.score_threshold)
    kept = initial[_nms_indices(boxes[initial], scores[initial], detector.nms_iou_threshold)]
    # detect() casts each final score to Python float before comparison.
    final = kept[scores[kept].astype(np.float64) > detector.final_person_score_threshold]
    accepted = final[valid[final]]
    expected = [{'box_xyxy': clipped[i].tolist(), 'score_raw': float(scores[i])} for i in accepted]
    if expected != actual:
        raise ValueError('audited filtering differs from ordinary detector output')
    # Codes are stable and saved for every anchor, not just the displayed Top-K.
    stages = np.zeros(len(scores), dtype=np.uint8)
    stages[initial] = 1
    stages[kept] = 2
    stages[final] = 3
    stages[accepted] = 4
    names = ['below_initial_threshold', 'suppressed_by_nms', 'below_final_threshold',
             'empty_box_after_clipping', 'accepted']
    order = np.argsort(scores, kind='stable')[::-1][:top_k]
    top = [{'anchor_index': int(i), 'person_score_raw': float(scores[i]),
            'objectness_raw': float(predictions[i, 4]), 'person_class_raw': float(predictions[i, 5]),
            'box_source_xyxy': source_boxes[i].tolist(), 'box_clipped_xyxy': clipped[i].tolist(),
            'stage': names[int(stages[i])]} for i in order]
    report = {
        'initial_threshold': detector.score_threshold,
        'final_threshold': detector.final_person_score_threshold,
        'nms_iou_threshold': detector.nms_iou_threshold,
        'raw_anchor_count': len(scores), 'above_initial_count': len(initial),
        'after_nms_count': len(kept), 'above_final_count': len(final),
        'accepted_count': len(accepted), 'max_person_score_raw': float(scores.max()),
        'score_definition': 'raw objectness times raw person-class score; not calibrated visibility',
        'stage_codes': names, 'top_k': top_k, 'top_view_truncated': len(order) < len(scores),
        'top_candidates': top, 'baseline_detections': actual,
    }
    arrays = {'boxes_detector_xyxy': boxes, 'boxes_source_xyxy': source_boxes,
              'person_scores_raw': scores, 'objectness_raw': predictions[:, 4],
              'person_class_raw': predictions[:, 5], 'stage_codes': stages}
    return report, arrays


def same_detections(actual, stored):
    def values(items, box_key, score_key):
        return sorted([list(item[box_key]) + [float(item[score_key])] for item in items])
    a = values(actual, 'box_xyxy', 'score_raw')
    b = values(stored, 'box_xyxy', 'detector_score_raw')
    return len(a) == len(b) and bool(np.allclose(a, b, atol=1e-5, rtol=1e-5))


def diagnostic_plan(config, video_ids, max_frames):
    if type(max_frames) is not int or max_frames <= 0:
        raise ValueError('max_frames must be a positive integer')
    if not video_ids or len(set(video_ids)) != len(video_ids):
        raise ValueError('request unique video IDs')
    selected = [shot for shot in config['shots'] if shot['video_id'] in video_ids]
    if {shot['video_id'] for shot in selected} != set(video_ids):
        raise ValueError('every video ID must belong to the existing batch config')
    baseline = json.loads((Path(config['output_dir']) / 'batch.json').read_text())
    if baseline['config'] != config:
        raise ValueError('config differs from immutable baseline batch')
    planned = []
    for shot in selected:
        bundle = Path(config['output_dir']) / 'bundles' / shot['video_id'] / shot['shot_id']
        manifest, _ = check_artifacts(bundle, config['videos_dir'], config['annotations'], config['split_manifest'])
        planned.append({'bundle': bundle, 'manifest': manifest,
                        'manifest_sha256': file_hash(bundle / 'manifest.json')})
    if sum(len(p['manifest']['source_frame_indices']) for p in planned) > max_frames:
        raise ValueError('requested frames exceed explicit diagnostic bound')
    return planned


def draw_candidates(frame, report, cv2):
    overlay = frame.copy()
    colors = {'accepted': (0, 220, 0), 'below_final_threshold': (0, 190, 255),
              'below_initial_threshold': (0, 0, 255), 'suppressed_by_nms': (220, 0, 220),
              'empty_box_after_clipping': (160, 160, 160)}
    for rank, item in enumerate(report['top_candidates'][:5], 1):
        x0, y0, x1, y1 = (round(v) for v in item['box_clipped_xyxy'])
        if x1 <= x0 or y1 <= y0:
            continue
        color = colors[item['stage']]
        cv2.rectangle(overlay, (x0, y0), (x1, y1), color, 2)
        text = f"#{rank} {item['person_score_raw']:.4f} {item['stage']}"
        cv2.putText(overlay, text, (x0, max(16, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return overlay


def run_diagnostic(config_path, video_ids, output_dir, *, max_frames=16, top_k=20,
                   dry_run=False, allow_dirty_code=False):
    config_path = Path(config_path)
    config_hash = file_hash(config_path)
    config = load_config(config_path)
    planned = diagnostic_plan(config, video_ids, max_frames)
    if type(top_k) is not int or not 1 <= top_k <= 100:
        raise ValueError('top_k must be an integer in [1,100]')
    provenance = code_provenance(require_clean=not allow_dirty_code)
    if dry_run:
        return {'dry_run': True, 'frames': [
            {'video_id': p['manifest']['video_id'], 'partition': p['manifest']['partition'],
             'indices': p['manifest']['source_frame_indices']} for p in planned]}
    import cv2
    extractor = create_dwpose_onnx_extractor(model_config(config))
    records = []
    started = time.perf_counter()
    with publish_bundle(output_dir) as output:
        for entry in planned:
            manifest, bundle = entry['manifest'], entry['bundle']
            if extractor.model_info['config_sha256'] != manifest['extractor_config_sha256']:
                raise ValueError('extractor config differs from baseline')
            vid, sid = manifest['video_id'], manifest['shot_id']
            raw = json.loads((bundle / 'raw_predictions.json').read_text())
            by_index = {row['source_frame_index']: row['result'] for row in raw}
            tiles = []
            capture = cv2.VideoCapture(str(Path(config['videos_dir']) / f'{vid}.mp4'))
            try:
                for index in manifest['source_frame_indices']:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        raise RuntimeError(f'could not decode {vid} frame {index}')
                    report, arrays = audit_detector(extractor.detector, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), top_k)
                    stored = by_index[index]['candidates']
                    agrees = same_detections(report['baseline_detections'], stored)
                    name = f'{vid}-{sid}-frame-{index}'
                    np.savez_compressed(output / f'{name}.npz', **arrays)
                    report.update(video_id=vid, shot_id=sid, source_frame_index=index,
                                  stored_candidate_count=len(stored), baseline_rerun_matches_stored=agrees,
                                  arrays_file=f'{name}.npz', arrays_sha256=file_hash(output / f'{name}.npz'))
                    write_json(output / f'{name}.json', report)
                    records.append({key: report[key] for key in (
                        'video_id', 'shot_id', 'source_frame_index', 'max_person_score_raw',
                        'above_initial_count', 'after_nms_count', 'accepted_count',
                        'stored_candidate_count', 'baseline_rerun_matches_stored')})
                    # Show original next to Top-5 raw proposals, including rejections.
                    overlay = draw_candidates(frame, report, cv2)
                    height, width = frame.shape[:2]
                    panel_h = max(1, round(480 * height / width))
                    images = np.hstack([cv2.resize(frame, (480, panel_h)), cv2.resize(overlay, (480, panel_h))])
                    header = np.zeros((32, 960, 3), dtype=np.uint8)
                    label = f'{vid} frame={index} accepted={report["accepted_count"]} max score={report["max_person_score_raw"]:.4f}'
                    cv2.putText(header, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    tiles.append(np.vstack([header, images]))
            finally:
                capture.release()
            sheet = np.vstack(tiles)
            if not cv2.imwrite(str(output / f'{vid}-{sid}-detector-review.png'), sheet):
                raise RuntimeError('failed to write diagnostic contact sheet')
        # Recheck every baseline at the end, including earlier shots.
        for entry in planned:
            bundle = entry['bundle']
            check_artifacts(bundle, config['videos_dir'], config['annotations'], config['split_manifest'])
            if file_hash(bundle / 'manifest.json') != entry['manifest_sha256']:
                raise ValueError('baseline manifest changed during diagnostic')
        for key in ('detector', 'pose'):
            if file_hash(config[f'{key}_model']) != config[f'{key}_sha256']:
                raise ValueError(f'{key} model changed during diagnostic')
        if file_hash(config_path) != config_hash or code_provenance(require_clean=False) != provenance:
            raise ValueError('config or source code changed during diagnostic')
        summary = {'purpose': 'detector filtering diagnosis only; baseline controls unchanged',
                   'training_eligible': False, 'code_provenance': provenance,
                   'baseline_code_commits': sorted({p['manifest']['code_commit'] for p in planned}),
                   'config_sha256': config_hash, 'model_info': extractor.model_info,
                   'baseline_bundles': [
                       {'video_id': p['manifest']['video_id'], 'shot_id': p['manifest']['shot_id'],
                        'manifest_sha256': p['manifest_sha256'],
                        'source_video_sha256': p['manifest']['source_video_sha256'],
                        'batch_identity': p['manifest'].get('batch_identity')} for p in planned],
                   'seconds': time.perf_counter() - started, 'frames': records}
        write_json(output / 'manifest.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-config', type=Path, required=True)
    parser.add_argument('--video-id', action='append', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-frames', type=int, default=16)
    parser.add_argument('--top-k', type=int, default=20)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--allow-dirty-code', action='store_true')
    args = parser.parse_args()
    summary = run_diagnostic(args.batch_config, args.video_id, args.output_dir,
        max_frames=args.max_frames, top_k=args.top_k, dry_run=args.dry_run,
        allow_dirty_code=args.allow_dirty_code)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
