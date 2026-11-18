#!/usr/bin/env python3
"""Review development-only no-person controls, then compare pose detector cutoffs."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from scripts.pose.bundle_io import code_provenance, file_hash, publish_bundle, write_json
from scripts.pose.prepare_batch import load_config, model_config
from scripts.pose.prepare_shot_pilot import load_inputs
from scripts.pose.check_preprocessing_pilot import check_artifacts
from scripts.pose.diagnose_detector import same_detections
from scripts.pose.prepare_video_pilot import render_pose
from scripts.pose.smoke_test_dwpose import _draw_overlay
from src.data.frame_sampling import resolve_frame_interval
from src.pose_models.extraction.dwpose_onnx import create_dwpose_onnx_extractor

SCRIPT = Path(__file__).resolve()
PATTERN = re.compile(r'\b(no people|no person|no characters|uninhabited|empty room|empty street|empty scene|landscape|establishing shot|aerial view|panoramic)\b', re.I)


def source_frame(path, index, cv2):
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f'Cannot open {path}')
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, bgr = capture.read()
        if not ok or bgr is None:
            raise RuntimeError(f'Cannot decode {path}, frame {index}')
        return bgr
    finally:
        capture.release()


def frame_hash(bgr):
    # Hash decoded RGB pixels and dimensions, not PNG compression bytes.
    rgb = np.ascontiguousarray(bgr[..., ::-1])
    return hashlib.sha256(str(rgb.shape).encode() + b'\0' + rgb.tobytes()).hexdigest()


def panel(image, cv2):
    h, w = image.shape[:2]
    scale = min(384/w, 216/h)
    small = cv2.resize(image, (max(1, round(w*scale)), max(1, round(h*scale))))
    out = np.zeros((216, 384, 3), dtype=np.uint8)
    out[:small.shape[0], :small.shape[1]] = small
    return out


def save_sheet(tiles, path, cv2, columns=2):
    if not tiles:
        raise ValueError('No frames for sheet')
    while len(tiles) % columns:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i:i+columns]) for i in range(0, len(tiles), columns)])
    if not cv2.imwrite(str(path), sheet):
        raise RuntimeError(f'Cannot save {path}')


def labelled(images, text, cv2):
    row = np.hstack([panel(image, cv2) for image in images])
    header = np.zeros((32, row.shape[1], 3), dtype=np.uint8)
    cv2.putText(header, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1)
    return np.vstack([header, row])


def candidates(rows, split, limit):
    """Caption shortlist only. Never interprets a keyword as a verified label."""
    if type(limit) is not int or not 1 <= limit <= 12:
        raise ValueError('limit must be in [1,12]')
    result, seen = [], set()
    for row in sorted(rows, key=lambda r: (str(r['video_id']), str(r['shot_id']))):
        vid = str(row['video_id'])
        if vid in {'01','37'} or vid in seen:
            continue
        partition = next((p for p in ('train','val') if vid in split[p]), None)
        caption = row.get('narrative_caption') or row.get('descriptive_caption') or ''
        if not partition or not isinstance(caption, str) or not PATTERN.search(caption):
            continue
        result.append({**row, 'partition': partition})
        seen.add(vid)
        if len(result) == limit:
            break
    return result


def verify_config(path):
    config = load_config(path)
    baseline = json.loads((Path(config['output_dir'])/'batch.json').read_text())
    if config != baseline['config']:
        raise ValueError('Config differs from baseline')
    for key, expected in baseline['input_hashes'].items():
        if file_hash(config[key]) != expected:
            raise ValueError(f'Baseline input changed: {key}')
    return config


def find_controls(config_path, output, limit=6):
    import cv2
    config_digest = file_hash(config_path)
    script_digest = file_hash(SCRIPT)
    code = code_provenance()
    config = verify_config(config_path)
    rows, split = load_inputs(config['annotations'], config['split_manifest'])
    chosen = candidates(rows, split, limit)
    if not chosen:
        raise ValueError('No caption shortlist found; supply visually chosen train/val frames instead')
    tiles, controls, video_hashes = [], [], {}
    with publish_bundle(output) as temporary:
        for number, row in enumerate(chosen):
            vid, sid = str(row['video_id']), str(row['shot_id'])
            video = Path(config['videos_dir'])/f'{vid}.mp4'
            video_hashes[str(video)] = file_hash(video)
            capture = cv2.VideoCapture(str(video))
            try:
                if not capture.isOpened():
                    raise RuntimeError(f'Cannot open {video}')
                start, end = resolve_frame_interval(int(row['segment_start_frame']), int(row['segment_end_frame']),
                                                    int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
            finally:
                capture.release()
            index = (start + end - 1)//2
            bgr = source_frame(video, index, cv2)
            filename = f'candidate-{number:02d}-{vid}-frame-{index}.png'
            if not cv2.imwrite(str(temporary/filename), bgr):
                raise RuntimeError('Cannot save candidate frame')
            controls.append({'video_id': vid, 'shot_id': sid, 'frame_index': index,
                'partition': row['partition'], 'source_video_sha256': video_hashes[str(video)],
                'frame_rgb_sha256': frame_hash(bgr), 'review_status': 'unreviewed',
                'image_file': filename, 'caption': row.get('narrative_caption') or row.get('descriptive_caption')})
            tiles.append(labelled([bgr], f'#{number:02d} {vid} / {row["partition"]} frame {index}: UNREVIEWED', cv2))
        save_sheet(tiles, temporary/'candidate-sheet.png', cv2, columns=3)
        for video, digest in video_hashes.items():
            if file_hash(video) != digest:
                raise ValueError('Candidate source video changed')
        if code_provenance() != code or file_hash(SCRIPT) != script_digest:
            raise ValueError('Code changed')
        verify_config(config_path)
        if file_hash(config_path) != config_digest:
            raise ValueError('Config changed')
        manifest = {'purpose': 'no-person candidates; HUMAN VISUAL REVIEW REQUIRED',
                    'training_eligible': False, 'config_sha256': config_digest,
                    'script_sha256': script_digest, 'code_provenance': code, 'controls': controls}
        write_json(temporary/'controls-to-review.json', manifest)
        write_json(temporary/'manifest.json', manifest)
    return {'candidate_count': len(controls), 'review_sheet': str(Path(output)/'candidate-sheet.png'),
            'all_controls_unreviewed': True}


def reviewed_controls(document, config, rows, split):
    selected = [c for c in document['controls'] if c.get('review_status') == 'verified_no_person']
    if not 2 <= len(selected) <= 6:
        raise ValueError('Need 2–6 visually verified no-person frames; keywords are not verification')
    seen = set()
    for c in selected:
        vid, sid, index = c['video_id'], c['shot_id'], c['frame_index']
        if type(index) is not int or index < 0 or (vid,index) in seen:
            raise ValueError('Invalid or duplicate control frame')
        seen.add((vid,index))
        matches = [r for r in rows if str(r['video_id']) == vid and str(r['shot_id']) == sid]
        if len(matches) != 1 or not (vid in split['train'] or vid in split['val']) or vid in split['test']:
            raise ValueError('Control must have one original train/val annotation')
        if not int(matches[0]['segment_start_frame']) <= index < int(matches[0]['segment_end_frame']):
            raise ValueError('Control frame outside annotated interval')
        if file_hash(Path(config['videos_dir'])/f'{vid}.mp4') != c['source_video_sha256']:
            raise ValueError('Control source bytes changed')
    return selected


def comparison_counts(records):
    positive = [r for r in records if r['kind'] == 'positive']
    negative = [r for r in records if r['kind'] == 'verified_no_person']
    return {
        'positive_frames': len(positive), 'verified_no_person_frames': len(negative),
        'newly_detected_positive_frames_NOT_accuracy': sum(r['baseline_count']==0 and r['trial_count']>0 for r in positive),
        'negative_frames_with_detections_at_0_3': sum(r['baseline_count']>0 for r in negative),
        'negative_frames_with_detections_at_0_2': sum(r['trial_count']>0 for r in negative),
    }


def compare(config_path, controls_path, output, *, factory=create_dwpose_onnx_extractor, cv2_module=None):
    if cv2_module is None:
        import cv2 as cv2_module
    cv2 = cv2_module
    script_digest, config_digest, controls_digest = file_hash(SCRIPT), file_hash(config_path), file_hash(controls_path)
    code = code_provenance()
    config = verify_config(config_path)
    document = json.loads(Path(controls_path).read_text())
    if document['config_sha256'] != config_digest:
        raise ValueError('Reviewed controls belong to another baseline config')
    rows, split = load_inputs(config['annotations'], config['split_manifest'])
    negatives = reviewed_controls(document, config, rows, split)
    items, bundles = [], []
    for shot in config['shots']:
        if shot['video_id'] not in {'01','37'}:
            continue
        bundle = Path(config['output_dir'])/'bundles'/shot['video_id']/shot['shot_id']
        manifest, _ = check_artifacts(bundle, config['videos_dir'], config['annotations'], config['split_manifest'])
        raw = json.loads((bundle/'raw_predictions.json').read_text())
        by_index = {r['source_frame_index']: r['result'] for r in raw}
        bundles.append((bundle, file_hash(bundle/'manifest.json')))
        for index in manifest['source_frame_indices']:
            items.append({'kind': 'positive', 'video_id': shot['video_id'], 'shot_id': shot['shot_id'],
                          'frame_index': index, 'source_video_sha256': manifest['source_video_sha256'],
                          'stored': by_index[index]})
    if {i['video_id'] for i in items} != {'01','37'} or len(items)>16:
        raise ValueError('Expected original 01/37 positive diagnostic samples, at most 16 frames')
    items.extend({**c, 'kind':'verified_no_person'} for c in negatives)
    baseline_config = model_config(config)
    trial_config = replace(baseline_config, final_person_score_threshold=0.2)
    # Separate instances avoid mutating either baseline or production defaults.
    baseline, trial = factory(baseline_config), factory(trial_config)
    records, sheets = [], {}
    with publish_bundle(output) as temporary:
        for item in items:
            vid, index = item['video_id'], item['frame_index']
            video = Path(config['videos_dir'])/f'{vid}.mp4'
            if file_hash(video) != item['source_video_sha256']:
                raise ValueError('Source video changed before comparison')
            frame = source_frame(video, index, cv2)
            if item['kind']=='verified_no_person' and frame_hash(frame) != item['frame_rgb_sha256']:
                raise ValueError('Reviewed control pixels differ')
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = [baseline.extract_frame(rgb), trial.extract_frame(rgb)]
            if 'stored' in item and not same_detections(
                [{'box_xyxy':c['box_xyxy'],'score_raw':c['detector_score_raw']} for c in results[0]['candidates']],
                item['stored']['candidates']):
                raise ValueError('Baseline detections no longer reproduce saved output')
            record = {k:v for k,v in item.items() if k not in {'stored','caption','image_file'}}
            record.update(baseline_count=len(results[0]['candidates']), trial_count=len(results[1]['candidates']))
            name = f'{item["kind"]}-{vid}-frame-{index}'
            write_json(temporary/f'{name}.json', {'frame':record,'baseline_0_3':results[0],'trial_0_2':results[1]})
            for label, result in zip(('baseline-0_3','trial-0_2'),results):
                maps = render_pose(result, config['size'], config['size'], config['sigma'])
                np.savez_compressed(temporary/f'{name}-{label}.npz', pose_encoded=maps[:, None])
            images = [frame] + [cv2.cvtColor(_draw_overlay(rgb,r,cv2),cv2.COLOR_RGB2BGR) for r in results]
            key = vid if item['kind']=='positive' else 'negative-controls'
            text=f'{vid} frame={index} | source | cutoff0.3: {record["baseline_count"]} | cutoff0.2: {record["trial_count"]}'
            sheets.setdefault(key,[]).append(labelled(images,text,cv2))
            records.append(record)
            print(text, flush=True)
        for key,tiles in sheets.items():
            save_sheet(tiles, temporary/f'{key}-comparison.png', cv2)
        for item in items:
            if file_hash(Path(config['videos_dir'])/f'{item["video_id"]}.mp4') != item['source_video_sha256']:
                raise ValueError('Source video changed during comparison')
        for bundle,digest in bundles:
            check_artifacts(bundle,config['videos_dir'],config['annotations'],config['split_manifest'])
            if file_hash(bundle/'manifest.json') != digest:
                raise ValueError('Baseline bundle changed')
        verify_config(config_path)
        if code_provenance()!=code or file_hash(SCRIPT)!=script_digest or file_hash(config_path)!=config_digest or file_hash(controls_path)!=controls_digest:
            raise ValueError('Code, config or reviewed controls changed')
        summary = {'purpose':'private development comparison, not a selected production threshold',
            'training_eligible':False,'initial_threshold':0.1,'final_thresholds':[0.3,0.2],
            'code_provenance':code,'script_sha256':script_digest,'config_sha256':config_digest,
            'reviewed_controls_sha256':controls_digest, 'baseline_model_info':baseline.model_info,
            'trial_model_info':trial.model_info,'counts':comparison_counts(records),'frames':records}
        write_json(temporary/'manifest.json',summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('find-controls','compare'))
    parser.add_argument('--batch-config',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--controls',type=Path)
    parser.add_argument('--limit',type=int,default=6)
    args=parser.parse_args()
    if args.mode=='find-controls':
        report=find_controls(args.batch_config,args.output_dir,args.limit)
    else:
        if args.controls is None:
            parser.error('--controls is required for compare')
        report=compare(args.batch_config,args.controls,args.output_dir)
    print(json.dumps(report,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
