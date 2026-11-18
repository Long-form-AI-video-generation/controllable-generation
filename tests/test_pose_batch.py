"""Failure/retry and resume contracts with synthetic frames, not real ONNX models."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.pose import bundle_io
from scripts.pose.prepare_batch import load_config, run_batch
from src.pose_models.extraction.dwpose import CANONICAL_JOINT_NAMES

PROVENANCE = {'commit': 'test-commit', 'dirty': False, 'changes_sha256': 'test-changes'}


class FakeCV:
    CAP_PROP_FRAME_COUNT = 1
    CAP_PROP_POS_FRAMES = 2
    COLOR_BGR2RGB = 3

    def __init__(self):
        self.captures = []
        self.fail_decode = False

    def VideoCapture(self, path):
        owner = self

        class Capture:
            def __init__(self):
                self.index = 0
                self.released = False

            def isOpened(self):
                return True

            def get(self, prop):
                return 5

            def set(self, prop, index):
                self.index = index

            def read(self):
                if owner.fail_decode:
                    return False, None
                value = int(Path(path).stem) * 10 + self.index
                return True, np.full((8, 10, 3), value, dtype=np.uint8)

            def release(self):
                self.released = True

        capture = Capture()
        self.captures.append(capture)
        return capture

    def cvtColor(self, frame, convention):
        return frame[..., ::-1]


class FakeExtractor:
    def __init__(self):
        self.fail_on = set()
        self.calls = 0

    def extract_frame(self, rgb):
        self.calls += 1
        value = int(rgb[0, 0, 0])
        if value in self.fail_on:
            raise RuntimeError('synthetic inference failure')
        positions = [None] * 17
        positions[0] = [2.0, 3.0]
        return {'width': 10, 'height': 8, 'model_info': {'config_sha256': 'fake-config'},
                'candidates': [] if value == 22 else [
                    {'joint_names': CANONICAL_JOINT_NAMES, 'xy_pixels': positions}]}


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cv = FakeCV()
        self.extractor = FakeExtractor()
        self.factory_calls = 0
        self.config = {key: str(self.root / key) for key in
                       ('videos_dir', 'annotations', 'split_manifest', 'output_dir', 'detector_model', 'pose_model')}
        Path(self.config['videos_dir']).mkdir()
        annotations = []
        for vid in ('02', '03'):
            (Path(self.config['videos_dir']) / f'{vid}.mp4').write_bytes(vid.encode())
            annotations.append({'video_id': vid, 'shot_id': vid + '_shot_0000',
                                'segment_start_frame': 0, 'segment_end_frame': 5,
                                'narrative_caption': 'Synthetic caption for loader pairing.'})
        Path(self.config['annotations']).write_text(json.dumps(annotations))
        Path(self.config['split_manifest']).write_text(json.dumps({'train': ['02'], 'val': ['03'], 'test': []}))
        for key in ('detector', 'pose'):
            path = Path(self.config[key + '_model'])
            path.write_bytes(key.encode())
            self.config[key + '_sha256'] = bundle_io.file_hash(path)
        self.config.update(onnxruntime_version='fake-1', opencv_version='fake-2',
                           num_frames=3, size=16, sigma=1.0, max_shots=2,
                           shots=[{'video_id': a['video_id'], 'shot_id': a['shot_id']} for a in annotations])
        self.config_path = self.root / 'config.json'
        self.config_path.write_text(json.dumps(self.config))
        self.config = load_config(self.config_path)
        self.provenance = patch('scripts.pose.prepare_batch.code_provenance', return_value=PROVENANCE)
        self.provenance.start()
        self.addCleanup(self.provenance.stop)
        shot_provenance = patch('scripts.pose.prepare_shot_pilot.code_provenance', return_value=PROVENANCE)
        shot_provenance.start()
        self.addCleanup(shot_provenance.stop)

    def factory(self, config):
        self.factory_calls += 1
        return self.extractor

    def run_batch(self, **kwargs):
        return run_batch(self.config, extractor_factory=self.factory, cv2_module=self.cv, **kwargs)

    def target(self, vid='02'):
        return Path(self.config['output_dir']) / 'bundles' / vid / (vid + '_shot_0000')

    def test_completed_batch_and_verified_resume_use_one_extractor(self):
        result = self.run_batch()
        self.assertEqual((result['completed'], result['failed'], result['unprocessed']), (2, 0, 0))
        self.assertEqual(self.factory_calls, 1)
        with np.load(self.target() / 'controls/02/shot_02_shot_0000_controls_encoded.npz') as archive:
            self.assertEqual(archive['pose_encoded'].shape, (3, 3, 16, 16))
            self.assertEqual(float(archive['pose_encoded'][:, 1].max()), 0.0)
        resumed = self.run_batch(resume=True)
        self.assertEqual(resumed['skipped'], 2)
        self.assertEqual(self.factory_calls, 1)
        self.assertEqual(self.extractor.calls, 6)
        self.assertTrue(all(c.released for c in self.cv.captures))

    def test_inference_failure_is_not_an_empty_success_and_can_retry(self):
        self.extractor.fail_on.add(30)
        result = self.run_batch()
        self.assertEqual((result['completed'], result['failed']), (1, 1))
        self.assertFalse(self.target('03').exists())
        attempts = list(self.target('03').parent.glob('.03_shot_0000.incomplete-*/failure.json'))
        self.assertEqual(len(attempts), 1)
        self.assertIn('synthetic inference failure', attempts[0].read_text())
        self.extractor.fail_on.clear()
        retry = self.run_batch(resume=True)
        self.assertEqual((retry['completed'], retry['skipped'], retry['failed']), (1, 1, 0))
        self.assertTrue(attempts[0].exists())  # failure evidence survives successful retry

    def test_interrupted_bundle_write_never_publishes_final_name(self):
        original = np.savez_compressed

        def broken_write(*args, **kwargs):
            original(*args, **kwargs)
            raise OSError('synthetic disk write interruption')

        with patch('scripts.pose.prepare_shot_pilot.np.savez_compressed', side_effect=broken_write):
            result = self.run_batch()
        self.assertEqual(result['failed'], 2)
        self.assertFalse(self.target().exists())
        self.assertTrue(list(self.target().parent.glob('.*.incomplete-*/failure.json')))
        retry = self.run_batch(resume=True)
        self.assertEqual(retry['completed'], 2)

    def test_keyboard_interrupt_records_summary_and_retries(self):
        with patch.object(self.extractor, 'extract_frame', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.run_batch()
        summaries = list((Path(self.config['output_dir']) / 'attempts').glob('*-summary.json'))
        saved = json.loads(summaries[0].read_text())
        self.assertTrue(saved['interrupted'])
        self.assertEqual(saved['unprocessed'], 1)
        self.assertEqual(saved['shots'][0]['status'], 'interrupted')
        self.assertFalse(self.target().exists())
        self.assertEqual(self.run_batch(resume=True)['completed'], 2)

    def test_changed_settings_rejected_before_more_inference(self):
        self.run_batch()
        calls = self.extractor.calls
        self.config['sigma'] = 2.0
        with self.assertRaisesRegex(ValueError, 'identity changed'):
            self.run_batch(resume=True)
        self.assertEqual(calls, self.extractor.calls)

    def test_corrupt_completed_control_is_reported_and_never_overwritten(self):
        self.run_batch()
        path = self.target() / 'controls/02/shot_02_shot_0000_controls_encoded.npz'
        path.write_bytes(b'corrupted')
        result = self.run_batch(resume=True)
        self.assertEqual((result['failed'], result['skipped']), (1, 1))
        self.assertEqual(path.read_bytes(), b'corrupted')
        self.assertEqual(self.factory_calls, 1)

    def test_changed_completed_source_video_rejects_resume_identity(self):
        self.run_batch()
        (Path(self.config['videos_dir']) / '02.mp4').write_bytes(b'changed source')
        with self.assertRaisesRegex(ValueError, 'identity changed'):
            self.run_batch(resume=True)
        self.assertEqual(self.factory_calls, 1)

    def test_failed_video_replacement_rejects_resume_identity(self):
        self.extractor.fail_on.add(30)
        self.assertEqual(self.run_batch()['failed'], 1)
        (Path(self.config['videos_dir']) / '03.mp4').write_bytes(b'replacement of failed source')
        self.extractor.fail_on.clear()
        with self.assertRaisesRegex(ValueError, 'identity changed'):
            self.run_batch(resume=True)
        self.assertFalse(self.target('03').exists())
        self.assertEqual(self.factory_calls, 1)

    def test_unprocessed_video_replacement_rejects_resume_identity(self):
        with patch.object(self.extractor, 'extract_frame', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.run_batch()
        (Path(self.config['videos_dir']) / '03.mp4').write_bytes(b'replacement of unprocessed source')
        with self.assertRaisesRegex(ValueError, 'identity changed'):
            self.run_batch(resume=True)
        self.assertFalse(self.target('03').exists())

    def test_not_yet_started_video_drift_is_recorded_as_failure(self):
        original = self.extractor.extract_frame
        video = Path(self.config['videos_dir']) / '03.mp4'

        def mutate_next_video(rgb):
            result = original(rgb)
            video.write_bytes(b'changed before second shot starts')
            return result

        with patch.object(self.extractor, 'extract_frame', side_effect=mutate_next_video):
            result = self.run_batch()
        self.assertEqual((result['completed'], result['failed']), (1, 1))
        self.assertIn('source video changed', result['shots'][1]['error'])
        self.assertFalse(self.target('03').exists())

    def test_video_change_during_extractor_initialization_is_rejected(self):
        video = Path(self.config['videos_dir']) / '02.mp4'

        def factory(config):
            video.write_bytes(b'changed after precheck but before prepare_shot')
            return self.extractor

        result = run_batch(self.config, extractor_factory=factory, cv2_module=self.cv)
        self.assertEqual(result['shots'][0]['status'], 'failed')
        self.assertIn('source hashes differ', result['shots'][0]['error'])
        self.assertFalse(self.target().exists())

    def test_video_change_during_inference_prevents_publication(self):
        original = self.extractor.extract_frame
        video = Path(self.config['videos_dir']) / '02.mp4'

        def mutate(rgb):
            result = original(rgb)
            video.write_bytes(b'changed while extracting')
            return result

        with patch.object(self.extractor, 'extract_frame', side_effect=mutate):
            result = self.run_batch()
        self.assertEqual(result['shots'][0]['status'], 'failed')
        self.assertIn('source_video_sha256', result['shots'][0]['error'])
        self.assertFalse(self.target().exists())

    def test_dry_run_has_no_output_or_model_initialization(self):
        plan = self.run_batch(dry_run=True)
        self.assertEqual([s['partition'] for s in plan['shots']], ['train', 'val'])
        self.assertFalse(Path(self.config['output_dir']).exists())
        self.assertEqual(self.factory_calls, 0)

    def test_test_partition_and_global_split_overlap_rejected_before_execution(self):
        path = Path(self.config['split_manifest'])
        for split in ({'train': ['02'], 'val': [], 'test': ['03']},
                      {'train': ['02'], 'val': ['03'], 'test': ['other'],}):
            if split['test'] == ['other']:
                split['train'].append('other')
            path.write_text(json.dumps(split))
            with self.assertRaises(ValueError):
                self.run_batch()
        self.assertFalse(Path(self.config['output_dir']).exists())

    def test_missing_meaningful_caption_rejected(self):
        path = Path(self.config['annotations'])
        rows = json.loads(path.read_text())
        rows[0]['narrative_caption'] = ''
        path.write_text(json.dumps(rows))
        with self.assertRaisesRegex(ValueError, 'caption'):
            self.run_batch()

    def test_decode_failure_releases_capture_and_leaves_no_completed_bundle(self):
        self.cv.fail_decode = True
        result = self.run_batch()
        self.assertEqual(result['failed'], 2)
        self.assertTrue(all(c.released for c in self.cv.captures))
        self.assertFalse(self.target().exists())

    def test_source_metadata_change_during_inference_prevents_publication(self):
        original = self.extractor.extract_frame
        path = Path(self.config['annotations'])

        def mutate(rgb):
            result = original(rgb)
            path.write_text(path.read_text() + ' ')
            return result

        with patch.object(self.extractor, 'extract_frame', side_effect=mutate):
            result = self.run_batch()
        self.assertEqual(result['failed'], 2)
        self.assertFalse(self.target().exists())
        self.assertIn('annotations_sha256', result['shots'][0]['error'])

    def test_changed_code_during_inference_prevents_publication(self):
        changed = {**PROVENANCE, 'dirty': True}
        with patch('scripts.pose.prepare_shot_pilot.code_provenance', side_effect=[PROVENANCE, changed]):
            # Limit this synthetic run to one shot, so exactly two code checks occur.
            self.config['shots'] = self.config['shots'][:1]
            result = self.run_batch()
        self.assertEqual(result['failed'], 1)
        self.assertFalse(self.target().exists())
        self.assertIn('source code changed', result['shots'][0]['error'])

    def test_invalid_configs_rejected(self):
        for update in ({'max_shots': 1}, {'size': True}, {'sigma': float('nan')},
                       {'shots': self.config['shots'] * 2}, {'num_frames': 0}, {'typo': 1}):
            with self.subTest(update=update):
                self.config_path.write_text(json.dumps({**self.config, **update}))
                with self.assertRaises(ValueError):
                    load_config(self.config_path)

    def test_changed_metadata_and_code_reject_resume_identity(self):
        self.run_batch()
        path = Path(self.config['annotations'])
        original = path.read_text()
        path.write_text(original + ' ')
        with self.assertRaisesRegex(ValueError, 'identity changed'):
            self.run_batch(resume=True)
        path.write_text(original)
        with patch('scripts.pose.prepare_batch.code_provenance', return_value={**PROVENANCE, 'commit': 'new'}):
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                self.run_batch(resume=True)

    def test_numeric_sigma_spellings_have_same_resume_identity(self):
        self.config_path.write_text(json.dumps({**self.config, 'sigma': 1}))
        self.config = load_config(self.config_path)
        self.run_batch()
        self.config_path.write_text(json.dumps({**self.config, 'sigma': 1.0}))
        self.config = load_config(self.config_path)
        self.assertEqual(self.run_batch(resume=True)['skipped'], 2)

    def test_no_overwrite_without_resume(self):
        self.run_batch()
        with self.assertRaises(FileExistsError):
            self.run_batch()


class ProvenanceAndLockTests(unittest.TestCase):
    def test_provenance_uses_source_repo_and_detects_tracked_and_untracked_code(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(['git', 'init', '-q', str(root)], check=True)
            (root / 'src').mkdir()
            source = root / 'src/example.py'
            source.write_text('original\n')
            subprocess.run(['git', '-C', str(root), 'add', 'src/example.py'], check=True)
            subprocess.run(['git', '-C', str(root), '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                            'commit', '-qm', 'test'], check=True)
            expected = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
            with patch.object(bundle_io, 'SOURCE_ROOT', root), patch.dict(os.environ, {'GIT_DIR': '/does-not-exist'}):
                clean = bundle_io.code_provenance()
                self.assertEqual(clean['commit'], expected)
                (root / 'local-notes.md').write_text('untracked notes do not make source dirty')
                self.assertEqual(bundle_io.code_provenance(), clean)
                source.write_text('edited\n')
                with self.assertRaisesRegex(ValueError, 'dirty'):
                    bundle_io.code_provenance()
                dirty = bundle_io.code_provenance(require_clean=False)
                self.assertTrue(dirty['dirty'])
                source.write_text('original\n')
                (root / 'src/new.py').write_text('new implementation\n')
                self.assertNotEqual(bundle_io.code_provenance(require_clean=False)['changes_sha256'], clean['changes_sha256'])

    def test_second_writer_cannot_enter_and_lock_is_released(self):
        with tempfile.TemporaryDirectory() as directory:
            lock = Path(directory) / 'writer.lock'
            with bundle_io.exclusive_lock(lock):
                with self.assertRaisesRegex(RuntimeError, 'another writer'):
                    with bundle_io.exclusive_lock(lock):
                        self.fail('second writer entered')
            with bundle_io.exclusive_lock(lock):
                pass


if __name__ == '__main__':
    unittest.main()
