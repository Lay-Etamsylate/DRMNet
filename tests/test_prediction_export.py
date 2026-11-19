import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from utils.prediction_export import COLUMNS, KINDS, export_saved_predictions


class PredictionExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def save(self, uid, boxes):
        for kind in KINDS:
            np.save(self.root / f'{uid}_{kind}s.npy', boxes)

    def read(self, kind):
        with (self.root / 'FROC' / f'submission_{kind}.csv').open(newline='') as stream:
            return list(csv.reader(stream))

    def test_mixed_empty_and_nonempty_scans_preserve_coordinates_and_coverage(self):
        self.save('zero', np.empty((0, 7), dtype=np.float32))
        self.save('detected', np.array([[.75, 10, 20, 30, 8, 9, 10]], dtype=np.float32))
        m = export_saved_predictions(self.root, ['zero', 'detected'])
        for kind in KINDS:
            self.assertEqual(self.read(kind), [COLUMNS, ['detected', '30.0', '20.0', '10.0', '8.0', '0.75']])
            self.assertEqual(m['prediction_counts'][kind], {'zero': 0, 'detected': 1})
        self.assertEqual(m['seriesuids'], ['zero', 'detected'])
        self.assertEqual(json.loads((self.root / 'FROC/inference_manifest.json').read_text()), m)

    def test_all_empty_writes_headers_and_zero_counts(self):
        self.save('a', np.empty((0, 7)))
        self.save('b', np.array([]))
        m = export_saved_predictions(self.root, ['a', 'b'])
        for kind in KINDS:
            self.assertEqual(self.read(kind), [COLUMNS])
            self.assertEqual(m['prediction_counts'][kind], {'a': 0, 'b': 0})

    def test_missing_file_is_not_silently_treated_as_empty(self):
        self.save('a', np.empty((0, 7)))
        export_saved_predictions(self.root, ['a'])
        (self.root / 'a_rcnns.npy').unlink()
        with self.assertRaisesRegex(FileNotFoundError, 'absence is not zero'):
            export_saved_predictions(self.root, ['a'])
        self.assertFalse((self.root / 'FROC/inference_manifest.json').exists())

    def test_rejects_invalid_data_without_publishing_complete_manifest(self):
        for boxes in (np.zeros(7), np.zeros((1, 8)), np.full((1, 7), np.nan)):
            self.save('a', boxes)
            with self.assertRaises(ValueError):
                export_saved_predictions(self.root, ['a'])
            self.assertFalse((self.root / 'FROC/inference_manifest.json').exists())

    def test_rejects_duplicate_or_empty_scan_list(self):
        for ids in ([], ['a', 'a']):
            with self.assertRaises(ValueError):
                export_saved_predictions(self.root, ids)


if __name__ == '__main__':
    unittest.main()
