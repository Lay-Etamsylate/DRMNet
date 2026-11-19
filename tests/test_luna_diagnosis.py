import unittest

from scripts.diagnose_luna16 import box_multiset, describe, matching_report


class DiagnosisTests(unittest.TestCase):
    def test_zero_prediction_scan_is_counted(self):
        predictions = [dict(seriesuid='A', coordX=1, coordY=0, coordZ=0)]
        annotations = [dict(seriesuid=uid, coordX=0, coordY=0, coordZ=0, diameter_mm=4)
                       for uid in ('A', 'B')]
        result = matching_report(predictions, annotations, ['A', 'B', 'C'])
        self.assertEqual(result['scans'], 3)
        self.assertEqual(result['reference_nodules'], 2)
        self.assertEqual(result['nodules_in_scans_without_predictions'], 1)
        self.assertEqual(result['geometric_hits_strict'], 1)

    def test_extra_legacy_margin_is_reported_separately(self):
        predictions = [dict(seriesuid='A', coordX=2, coordY=0, coordZ=0)]
        annotations = [dict(seriesuid='A', coordX=0, coordY=0, coordZ=0, diameter_mm=2)]
        result = matching_report(predictions, annotations, ['A'])
        self.assertEqual(result['geometric_hits_strict'], 0)
        self.assertEqual(result['geometric_hits_legacy_margin'], 1)

    def test_box_comparison_ignores_order_not_multiplicity(self):
        a = dict(seriesuid='A', coordX=1, coordY=2, coordZ=3, diameter_mm=4)
        b = dict(a, coordX=10)
        self.assertEqual(box_multiset([a, b]), box_multiset([b, a]))
        self.assertNotEqual(box_multiset([a, b]), box_multiset([a, a, b]))

    def test_constant_and_empty_scores(self):
        self.assertEqual(describe([.9] * 3)['unique'], 1)
        self.assertEqual(describe([])['min'], None)


if __name__ == '__main__':
    unittest.main()
