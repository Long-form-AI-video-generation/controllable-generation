"""Detector audit correctness without real models or heavyweight runtimes."""
from pathlib import Path
import unittest
import json
import types
from unittest.mock import patch

import numpy as np

from scripts.pose.diagnose_detector import audit_detector, diagnostic_plan, same_detections, run_diagnostic
from src.pose_models.extraction.dwpose_onnx import DWPoseONNXDetector, _yolox_grid
import test_dwpose_onnx as onnx_fixtures
import test_pose_batch as batch_fixtures


def output_with_boxes(specs):
    output = np.zeros((1, 21, 6), dtype=np.float32)
    grids, strides = _yolox_grid((32, 32))
    for i, (box, score) in enumerate(specs):
        x0, y0, x1, y1 = box
        stride = strides[0, i, 0]
        output[0, i, :2] = np.array([(x0 + x1) / 2, (y0 + y1) / 2]) / stride - grids[0, i]
        output[0, i, 2:4] = np.log(np.array([x1-x0, y1-y0]) / stride)
        output[0, i, 4:] = [1.0, score]
    return output


class DetectorAuditTests(unittest.TestCase):
    def detector(self, specs):
        session = onnx_fixtures.FakeSession(output_with_boxes(specs))
        detector = DWPoseONNXDetector(session, input_size_hw=(32, 32), cv2_module=onnx_fixtures.FakeCV2)
        return detector, session

    def test_distinguishes_score_nms_and_empty_box_rejections(self):
        detector, session = self.detector([
            ([0, 0, 16, 16], 0.8), ([0, 0, 16, 16], 0.7),
            ([18, 18, 26, 26], 0.2), ([2, 22, 10, 30], 0.05),
            ([-40, -40, -32, -32], 0.9),
        ])
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        expected = detector.detect(image)
        with patch.object(session, 'run', wraps=session.run) as run:
            report, arrays = audit_detector(detector, image)
        self.assertEqual(run.call_count, 1)
        self.assertIs(detector.session, session)
        self.assertEqual(report['baseline_detections'], expected)
        self.assertEqual(report['accepted_count'], 1)
        self.assertEqual(arrays['stage_codes'][:5].tolist(), [4, 1, 2, 0, 3])
        self.assertEqual(report['above_initial_count'], 4)
        self.assertEqual(report['after_nms_count'], 3)
        self.assertEqual(report['above_final_count'], 2)
        self.assertEqual(report['raw_anchor_count'], 21)
        np.testing.assert_allclose(arrays['person_scores_raw'][:5], [.8, .7, .2, .05, .9])

    def test_restores_session_on_inference_failure(self):
        detector, session = self.detector([])
        with patch.object(session, 'run', side_effect=RuntimeError('synthetic ONNX failure')):
            with self.assertRaisesRegex(RuntimeError, 'ONNX inference failed'):
                audit_detector(detector, np.zeros((32, 32, 3), dtype=np.uint8))
        self.assertIs(detector.session, session)

    def test_exact_final_threshold_matches_existing_python_float_semantics(self):
        detector, _ = self.detector([([0, 0, 8, 8], .3)])
        report, _ = audit_detector(detector, np.zeros((32, 32, 3), dtype=np.uint8))
        self.assertEqual(report['accepted_count'], 1)
        self.assertEqual(report['top_candidates'][0]['stage'], 'accepted')

    def test_initial_boundary_is_rejected_as_in_existing_numpy_filter(self):
        detector, _ = self.detector([([0, 0, 8, 8], .1)])
        report, _ = audit_detector(detector, np.zeros((32, 32, 3), dtype=np.uint8))
        self.assertEqual(report['above_initial_count'], 0)
        self.assertEqual(report['accepted_count'], 0)

    def test_top_view_truncation_does_not_truncate_raw_arrays(self):
        detector, _ = self.detector([([0, 0, 8, 8], .2)])
        report, arrays = audit_detector(detector, np.zeros((32, 32, 3), dtype=np.uint8), top_k=1)
        self.assertTrue(report['top_view_truncated'])
        self.assertEqual(len(report['top_candidates']), 1)
        self.assertEqual(len(arrays['person_scores_raw']), 21)
        self.assertEqual(report['accepted_count'], 0)

    def test_source_scaling_uses_original_frame_and_clips(self):
        detector, _ = self.detector([([-4, -4, 12, 12], .8)])
        report, arrays = audit_detector(detector, np.zeros((16, 16, 3), dtype=np.uint8))
        np.testing.assert_allclose(arrays['boxes_source_xyxy'][0], [-2, -2, 6, 6])
        self.assertEqual(report['baseline_detections'][0]['box_xyxy'], [0, 0, 6, 6])

    def test_invalid_top_k_rejected_without_touching_session(self):
        detector, session = self.detector([])
        for value in (0, 101, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                audit_detector(detector, np.zeros((32, 32, 3), dtype=np.uint8), value)
            self.assertIs(detector.session, session)

    def test_stored_detection_comparison_is_order_independent(self):
        actual = [{'box_xyxy': [0, 0, 8, 8], 'score_raw': .8},
                  {'box_xyxy': [9, 9, 16, 16], 'score_raw': .6}]
        stored = [{'box_xyxy': d['box_xyxy'], 'detector_score_raw': d['score_raw']} for d in reversed(actual)]
        self.assertTrue(same_detections(actual, stored))
        self.assertTrue(same_detections([], []))
        self.assertFalse(same_detections(actual, []))
        stored[0]['box_xyxy'] = [10, 10, 16, 16]
        self.assertFalse(same_detections(actual, stored))


class DiagnosticPlanTests(unittest.TestCase):
    def setUp(self):
        self.fixture = batch_fixtures.BatchTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.run_batch()

    def test_selects_only_requested_existing_development_videos(self):
        plan = diagnostic_plan(self.fixture.config, ['03'], max_frames=3)
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0]['manifest']['partition'], 'val')
        self.assertEqual(plan[0]['manifest']['video_id'], '03')
        self.assertEqual(len(plan[0]['manifest_sha256']), 64)

    def test_cannot_exceed_bound_or_request_outside_batch(self):
        for ids, limit in ((['02', '03'], 5), (['not-in-batch'], 8), (['02', '02'], 8)):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                diagnostic_plan(self.fixture.config, ids, limit)

    def test_changed_source_video_blocks_diagnostic(self):
        (Path(self.fixture.config['videos_dir']) / '02.mp4').write_bytes(b'replaced')
        with self.assertRaisesRegex(ValueError, 'source_video_sha256'):
            diagnostic_plan(self.fixture.config, ['02'], 3)

    def test_changed_config_blocks_diagnostic(self):
        config = {**self.fixture.config, 'sigma': 9.0}
        with self.assertRaisesRegex(ValueError, 'config differs'):
            diagnostic_plan(config, ['02'], 3)


class DiagnosticRunTests(unittest.TestCase):
    setUp = DiagnosticPlanTests.setUp
    def test_complete_audit_preserves_baseline_and_records_all_anchors(self):
        baseline = Path(self.fixture.config['output_dir'])
        # Add detector fields absent in the minimal map-rendering fixture.
        raw_path = self.fixture.target() / 'raw_predictions.json'
        rows = json.loads(raw_path.read_text())
        for row in rows:
            for candidate in row['result']['candidates']:
                candidate.update(box_xyxy=[0, 0, 2.5, 2.5], detector_score_raw=float(np.float32(.8)))
        raw_path.write_text(json.dumps(rows))
        manifest_path = self.fixture.target() / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['raw_predictions_sha256'] = batch_fixtures.bundle_io.file_hash(raw_path)
        manifest_path.write_text(json.dumps(manifest))
        before = {str(p.relative_to(baseline)): batch_fixtures.bundle_io.file_hash(p)
                  for p in baseline.rglob('*') if p.is_file()}

        class CV(batch_fixtures.FakeCV):
            FONT_HERSHEY_SIMPLEX = 0

            @staticmethod
            def rectangle(*args):
                pass

            @staticmethod
            def putText(*args):
                pass

            @staticmethod
            def resize(image, size):
                return np.zeros((size[1], size[0], 3), dtype=np.uint8)

            @staticmethod
            def imwrite(path, image):
                Path(path).write_bytes(b'synthetic PNG stand-in; no real OpenCV test')
                return True

        cv = CV()
        detector = DWPoseONNXDetector(
            onnx_fixtures.FakeSession(output_with_boxes([([0, 0, 8, 8], .8)])),
            input_size_hw=(32, 32), cv2_module=onnx_fixtures.FakeCV2)
        extractor = types.SimpleNamespace(detector=detector, model_info={'config_sha256': 'fake-config'})
        output = self.fixture.root / 'audit'
        with patch.dict('sys.modules', {'cv2': cv}), \
             patch('scripts.pose.diagnose_detector.create_dwpose_onnx_extractor', return_value=extractor), \
             patch('scripts.pose.diagnose_detector.code_provenance', return_value=batch_fixtures.PROVENANCE):
            report = run_diagnostic(self.fixture.config_path, ['02'], output, max_frames=3)
        self.assertEqual(len(report['frames']), 3)
        self.assertFalse(report['training_eligible'])
        self.assertEqual([r['baseline_rerun_matches_stored'] for r in report['frames']], [True, False, True])
        self.assertTrue((output / 'manifest.json').is_file())
        archives = list(output.glob('*.npz'))
        self.assertEqual(len(archives), 3)
        with np.load(archives[0]) as data:
            self.assertEqual(data['person_scores_raw'].shape, (21,))
        after = {str(p.relative_to(baseline)): batch_fixtures.bundle_io.file_hash(p)
                 for p in baseline.rglob('*') if p.is_file()}
        self.assertEqual(before, after)
        self.assertTrue(all(c.released for c in cv.captures))

    def test_dry_run_does_not_initialize_models_or_create_output(self):
        output = self.fixture.root / 'dry-run'
        with patch('scripts.pose.diagnose_detector.create_dwpose_onnx_extractor') as factory, \
             patch('scripts.pose.diagnose_detector.code_provenance', return_value=batch_fixtures.PROVENANCE):
            report = run_diagnostic(self.fixture.config_path, ['02'], output, max_frames=3, dry_run=True)
        factory.assert_not_called()
        self.assertTrue(report['dry_run'])
        self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
