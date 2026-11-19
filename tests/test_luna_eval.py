import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from scripts.evaluate_luna16 import (ANNO_FIELDS, KINDS, WORLD_FIELDS, check_coverage,
                                    prepare, read_csv, write_csv)
from utils.luna_eval import cropped_xyz_to_world, sample_froc, match_distance_squared


def metadata(uid='A'):
    return dict(seriesuid=np.array(uid), crop_start_zyx=np.array([3, 2, 1]),
                resampled_origin_xyz=np.array([10., 20., 30.]),
                resampled_spacing_xyz=np.array([1., 2., 3.]),
                resampled_direction=np.eye(3).ravel())


class CoordinateTests(unittest.TestCase):
    def test_strict_radius_and_historical_extra_tolerance(self):
        # Radius=3 mm; distance sqrt(10) is outside but matched by legacy +5.
        self.assertTrue(match_distance_squared(8.99, 9., 'strict'))
        self.assertFalse(match_distance_squared(9., 9., 'strict'))
        self.assertFalse(match_distance_squared(10., 9., 'strict'))
        self.assertTrue(match_distance_squared(10., 9., 'legacy'))
        self.assertFalse(match_distance_squared(14., 9., 'legacy'))
        with self.assertRaises(ValueError):
            match_distance_squared(1., 9., 'unknown')

    def test_user_annotation_round_trip(self):
        meta = dict(crop_start_zyx=[44, 85, 26], resampled_origin_xyz=[-200.7, -217.2, -385.25],
                    resampled_spacing_xyz=[1, 1, 1], resampled_direction=np.eye(3).ravel())
        center = [[(210.671265008 + 219.35832507199999) / 2,
                   (47.132160208 + 55.81922027200001) / 2,
                   (186.911694068 + 195.59875413199998) / 2]]
        np.testing.assert_allclose(cropped_xyz_to_world(center, meta),
                                   [[40.31479504, -80.72430976, -149.9947759]], atol=1e-9)

    def test_direction_and_anisotropic_spacing(self):
        meta = metadata()
        meta['resampled_direction'] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]).ravel()
        np.testing.assert_allclose(cropped_xyz_to_world([[1, 2, 3]], meta), [[2, 22, 48]])
        self.assertEqual(cropped_xyz_to_world(np.empty((0, 3)), meta).shape, (0, 3))

    def test_fixed_points_interpolate_and_extend_endpoint(self):
        np.testing.assert_allclose(sample_froc([0, 1, 2], [0, .5, 1]),
                                   [.0625, .125, .25, .5, 1, 1, 1])
        np.testing.assert_allclose(sample_froc([0, 0], [0, .5]), [.5] * 7)
        np.testing.assert_allclose(sample_froc([0], [0]), [0] * 7)

    def test_invalid_geometry_rejected(self):
        meta = metadata()
        meta['resampled_spacing_xyz'][0] = 0
        with self.assertRaises(ValueError):
            cropped_xyz_to_world([[1, 2, 3]], meta)


class EvaluationPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pred = self.root / 'pred'
        self.pred.mkdir()
        self.ids = ['A', 'B', 'C']
        np.save(self.root / 'test.npy', np.array(self.ids))
        preprocessed = []
        official = []
        for uid in self.ids:
            np.savez(self.root / f'{uid}_meta.npz', **metadata(uid))
            if uid != 'C':
                row = dict(pid=uid, xmin=-1, xmax=3, ymin=0, ymax=4, zmin=1, zmax=5,
                           coordX=12, coordY=28, coordZ=48, diameter_mm=4)
                preprocessed.append(row)
                official.append(dict(seriesuid=uid, **{k: row[k] for k in ANNO_FIELDS[1:]}))
        write_csv(self.root / 'test_annos.csv', list(preprocessed[0]), preprocessed)
        write_csv(self.root / 'annotations.csv', ANNO_FIELDS, official)
        write_csv(self.root / 'excluded.csv', ANNO_FIELDS,
                  [dict(zip(ANNO_FIELDS, ['A', 1000, 1000, 1000, 4]))])
        for kind in KINDS:
            write_csv(self.pred / f'submission_{kind}.csv', WORLD_FIELDS,
                      [dict(zip(WORLD_FIELDS, ['A', 1, 2, 3, .9])),
                       dict(zip(WORLD_FIELDS, ['A', 100, 100, 100, .1]))])
        self.args = SimpleNamespace(test_set=self.root / 'test.npy', meta_dir=self.root,
                                    preprocessed_annotations=None, annotations=self.root / 'annotations.csv',
                                    excluded=self.root / 'excluded.csv', pred_dir=self.pred,
                                    out_dir=self.root / 'world', assume_complete=True, seed=35202)

    def run_prepare(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return prepare(self.args)

    def test_conversion_keeps_zero_detection_scans_and_originals(self):
        original = (self.pred / 'submission_rcnn.csv').read_bytes()
        report = self.run_prepare()
        self.assertEqual(report['scans'], 3)
        self.assertEqual(report['nodules'], 2)
        self.assertEqual(report['scans_with_predictions']['rcnn'], 1)
        self.assertEqual((self.args.out_dir / 'seriesuids.csv').read_text(), 'A\nB\nC\n')
        converted = read_csv(self.args.out_dir / 'submission_rcnn.csv', WORLD_FIELDS)
        np.testing.assert_allclose([float(converted[0][k]) for k in WORLD_FIELDS[1:4]], [12, 28, 48])
        self.assertEqual((self.pred / 'submission_rcnn.csv').read_bytes(), original)

    def test_coordinate_mismatch_fails(self):
        meta = metadata('A')
        meta['resampled_origin_xyz'][0] += 1
        np.savez(self.root / 'A_meta.npz', **meta)
        with self.assertRaisesRegex(ValueError, 'coordinate check failed'):
            self.run_prepare()

    def test_official_annotation_mismatch_fails(self):
        write_csv(self.args.annotations, ANNO_FIELDS, [])
        with self.assertRaisesRegex(ValueError, 'official truth'):
            self.run_prepare()

    def test_empty_exclusion_file_rejected(self):
        write_csv(self.args.excluded, ANNO_FIELDS, [])
        with self.assertRaisesRegex(ValueError, 'placeholder'):
            self.run_prepare()

    def test_missing_or_incomplete_manifest_rejected(self):
        self.args.assume_complete = False
        with self.assertRaisesRegex(ValueError, 'No inference_manifest'):
            self.run_prepare()
        manifest = dict(status='complete', coordinate_system='cropped_voxel_xyz', seriesuids=self.ids,
                        prediction_counts={k: {'A': 2, 'B': 0, 'C': 0} for k in KINDS})
        path = self.pred / 'inference_manifest.json'
        path.write_text(json.dumps(manifest))
        self.assertEqual(self.run_prepare()['coverage'], 'manifest_verified')
        manifest['seriesuids'] = ['A']
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            self.run_prepare()

    def test_no_detections_still_keeps_all_scans(self):
        for kind in KINDS:
            write_csv(self.pred / f'submission_{kind}.csv', WORLD_FIELDS, [])
        report = self.run_prepare()
        self.assertEqual(report['prediction_rows'], {k: 0 for k in KINDS})
        self.assertEqual(report['scans'], 3)
        self.assertEqual(read_csv(self.args.out_dir / 'submission_rcnn.csv', WORLD_FIELDS), [])

    def test_matching_rule_reaches_evaluator(self):
        try:
            import evaluationScript.noduleCADEvaluationLUNA16 as evaluator
        except ModuleNotFoundError as exc:
            self.skipTest(str(exc))
        self.run_prepare()
        # GT A is at (12, 28, 48), radius 2 mm. This prediction is 2.5 mm
        # away: legacy accepts it (6.25 < 4 + 5), strict rejects it.
        write_csv(self.args.out_dir / 'submission_rcnn.csv', WORLD_FIELDS,
                  [dict(zip(WORLD_FIELDS, ['A', 14.5, 28, 48, .9]))])
        original = evaluator.bPerformBootstrapping
        evaluator.bPerformBootstrapping = False
        self.addCleanup(setattr, evaluator, 'bPerformBootstrapping', original)
        for rule, expected in [('strict', 0.), ('legacy', .5)]:
            out = self.root / rule
            out.mkdir()
            with contextlib.redirect_stdout(io.StringIO()):
                _, _, raw, cpm = evaluator.noduleCADEvaluation(
                    str(self.args.out_dir / 'annotations.csv'),
                    str(self.args.out_dir / 'annotations_excluded.csv'),
                    str(self.args.out_dir / 'seriesuids.csv'),
                    str(self.args.out_dir / 'submission_rcnn.csv'), str(out),
                    matching_rule=rule)
            np.testing.assert_allclose(raw, [expected] * 7)
            self.assertEqual(cpm, expected)
            self.assertIn('Matching rule: ' + rule, (out / 'CADAnalysis.txt').read_text())

    def test_complete_froc_pipeline(self):
        try:
            import evaluationScript.noduleCADEvaluationLUNA16 as evaluator
        except ModuleNotFoundError as exc:
            self.skipTest(str(exc))
        self.run_prepare()
        out = self.root / 'metrics'
        out.mkdir()
        original_samples = evaluator.bNumberOfBootstrapSamples
        evaluator.bNumberOfBootstrapSamples = 20
        self.addCleanup(setattr, evaluator, 'bNumberOfBootstrapSamples', original_samples)
        np.random.seed(35202)
        with contextlib.redirect_stdout(io.StringIO()):
            boot, cpm_boot, raw, cpm_raw = evaluator.noduleCADEvaluation(
                str(self.args.out_dir / 'annotations.csv'), str(self.args.out_dir / 'annotations_excluded.csv'),
                str(self.args.out_dir / 'seriesuids.csv'), str(self.args.out_dir / 'submission_rcnn.csv'), str(out))
        np.testing.assert_allclose(raw, [.5] * 7)
        self.assertEqual(cpm_raw, .5)
        self.assertEqual(len(boot), 7)
        self.assertTrue(np.isfinite(cpm_boot))
        for gt, scores, excluded, expected in [([], [], [], 0),
                                              ([1], [.9], [False], 1),
                                              ([1, 0], [-1, .5], [True, False], 0)]:
            fps, sens, _ = evaluator.computeFROC(gt, scores, 3, excluded)
            np.testing.assert_allclose(sample_froc(fps, sens), [expected] * 7)
        write_csv(self.args.out_dir / 'submission_rcnn.csv', WORLD_FIELDS, [])
        with contextlib.redirect_stdout(io.StringIO()):
            boot, cpm_boot, raw, cpm_raw = evaluator.noduleCADEvaluation(
                str(self.args.out_dir / 'annotations.csv'), str(self.args.out_dir / 'annotations_excluded.csv'),
                str(self.args.out_dir / 'seriesuids.csv'), str(self.args.out_dir / 'submission_rcnn.csv'), str(out))
        np.testing.assert_array_equal(raw, [0] * 7)
        np.testing.assert_array_equal(boot, [0] * 7)
        self.assertEqual(cpm_raw, 0)
        self.assertEqual(cpm_boot, 0)


if __name__ == '__main__':
    unittest.main()
