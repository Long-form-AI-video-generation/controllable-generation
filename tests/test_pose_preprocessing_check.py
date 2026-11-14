"""Artifact-level integration checks, independent of heavyweight model runtimes."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.pose.check_preprocessing_pilot import check_artifacts, file_hash
from scripts.pose.prepare_video_pilot import build_sequence
from src.pose_models.extraction.dwpose import CANONICAL_JOINT_NAMES


class PilotAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bundle = self.root / 'bundle'
        self.bundle.mkdir()
        (self.root / '02.mp4').write_bytes(b'synthetic source identity only')
        self.annotations = self.root / 'annotations.json'
        self.annotations.write_text(json.dumps([{'video_id': '02', 'shot_id': '02_shot_0000',
                                                'segment_start_frame': 0, 'segment_end_frame': 5}]))
        self.split = self.root / 'split.json'
        self.split.write_text(json.dumps({'train': ['02'], 'val': [], 'test': []}))
        self.rows = []
        for i in range(5):
            points = [None] * 17
            points[0] = [10 + i, 12]
            self.rows.append({'source_frame_index': i, 'result': {
                'width': 100, 'height': 50, 'model_info': {'config_sha256': 'test'},
                'candidates': [{'joint_names': CANONICAL_JOINT_NAMES, 'xy_pixels': points}]}})
        self.raw = self.bundle / 'raw_predictions.json'
        self.raw.write_text(json.dumps(self.rows))
        maps, indices, _ = build_sequence(self.rows, 0, 5, 3, 16, 16, 1.0)
        self.control = self.bundle / 'controls/02/shot_02_shot_0000_controls_encoded.npz'
        self.control.parent.mkdir(parents=True)
        np.savez_compressed(self.control, pose_encoded=maps)
        self.manifest = {
            'training_eligible': False, 'video_id': '02', 'shot_id': '02_shot_0000',
            'partition': 'train', 'source_frame_count': 5, 'source_interval_half_open': [0, 5],
            'source_frame_indices': indices, 'sigma_pixels': 1.0,
            'shape_cthw': list(maps.shape), 'extractor_config_sha256': 'test',
            'source_video_sha256': file_hash(self.root / '02.mp4'),
            'annotations_sha256': file_hash(self.annotations), 'split_manifest_sha256': file_hash(self.split),
            'raw_predictions_sha256': file_hash(self.raw),
        }
        self.write_manifest()

    def write_manifest(self):
        (self.bundle / 'manifest.json').write_text(json.dumps(self.manifest))

    def check(self):
        return check_artifacts(self.bundle, self.root, self.annotations, self.split)

    def test_rebuilds_legacy_bundle_without_control_file_hash(self):
        manifest, path = self.check()
        self.assertEqual(manifest['source_frame_indices'], [0, 2, 4])
        self.assertEqual(path, self.control)

    def test_rejects_modified_video(self):
        (self.root / '02.mp4').write_bytes(b'modified video')
        with self.assertRaisesRegex(ValueError, 'source_video_sha256'):
            self.check()

    def test_rejects_control_corruption_even_without_file_hash(self):
        np.savez_compressed(self.control, pose_encoded=np.zeros((3, 3, 16, 16), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, 'retained raw predictions'):
            self.check()

    def test_rejects_index_or_annotation_drift(self):
        self.manifest['source_frame_indices'] = [0, 1, 4]
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'shared sampler'):
            self.check()
        self.manifest['source_frame_indices'] = [0, 2, 4]
        self.manifest['source_interval_half_open'] = [0, 4]
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'frame interval'):
            self.check()

    def test_rejects_test_partition_or_eligibility_change(self):
        self.manifest['partition'] = 'test'
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'development partition'):
            self.check()
        self.manifest['partition'] = 'train'
        self.manifest['training_eligible'] = True
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'diagnostic bundles'):
            self.check()

    def test_checks_new_control_hash_when_present(self):
        self.manifest['control_file_sha256'] = '0' * 64
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'control_file_sha256'):
            self.check()


if __name__ == '__main__':
    unittest.main()
