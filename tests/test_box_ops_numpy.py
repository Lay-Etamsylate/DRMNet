import unittest
import numpy as np

from utils.box_ops_numpy import nms_indices_numpy, overlap_numpy


class BoxOpsTests(unittest.TestCase):
    def test_overlap_matches_scalar_geometry_with_chunking(self):
        rng = np.random.default_rng(12)
        a = np.concatenate([rng.normal(size=(11, 3)), rng.uniform(.1, 4, (11, 3))], 1)
        b = np.concatenate([rng.normal(size=(7, 3)), rng.uniform(.1, 4, (7, 3))], 1)
        expected = np.zeros((11, 7))
        for i, x in enumerate(a):
            for j, y in enumerate(b):
                extent = [max(0, min(x[k] + x[k+3]/2, y[k] + y[k+3]/2)
                              - max(x[k] - x[k+3]/2, y[k] - y[k+3]/2)) for k in range(3)]
                inter = np.prod(extent)
                expected[i, j] = inter / (np.prod(x[3:]) + np.prod(y[3:]) - inter)
        np.testing.assert_allclose(overlap_numpy(a, b, chunk_size=3), expected, rtol=1e-5, atol=1e-7)

    def test_nms_threshold_equality_and_order(self):
        dets = np.array([[.7, 0, 0, 0, 2, 2, 2], [.9, 0, 0, 0, 2, 2, 2],
                         [.8, 10, 0, 0, 2, 2, 2]], np.float32)
        # Legacy C suppresses equal IoU too: identical boxes at threshold 1.
        np.testing.assert_array_equal(nms_indices_numpy(dets, 1), [1, 2])
        np.testing.assert_array_equal(nms_indices_numpy(dets, .1, order=[0, 1, 2]), [0, 2])

    def test_touching_and_empty_boxes(self):
        a = np.array([[0, 0, 0, 2, 2, 2]], np.float32)
        b = np.array([[2, 0, 0, 2, 2, 2]], np.float32)
        np.testing.assert_array_equal(overlap_numpy(a, a), [[1]])
        np.testing.assert_array_equal(overlap_numpy(a, b), [[0]])
        self.assertEqual(overlap_numpy(np.empty((0, 6)), b).shape, (0, 1))
        self.assertEqual(overlap_numpy(a, np.empty((0, 6))).shape, (1, 0))
        self.assertEqual(nms_indices_numpy(np.empty((0, 7)), .1).shape, (0,))

    def test_rcnn_class_column_and_row_indices(self):
        # Seven eight-column rows used to reshape silently into eight rows.
        dets = np.array([
            [.7, 0, 0, 0, 2, 2, 2, 1],
            [.9, 0, 0, 0, 2, 2, 2, 1],
            [.8, 10, 0, 0, 2, 2, 2, 1],
            [.6, 10, 0, 0, 2, 2, 2, 1],
            [.5, 20, 0, 0, 2, 2, 2, 1],
            [.4, 20, 0, 0, 2, 2, 2, 1],
            [.3, 30, 0, 0, 2, 2, 2, 1],
        ], np.float32)
        np.testing.assert_array_equal(nms_indices_numpy(dets, .1), [1, 2, 4, 6])
        np.testing.assert_array_equal(nms_indices_numpy(dets[:1], .1), [0])
        np.testing.assert_array_equal(
            nms_indices_numpy(dets, .1, order=[0, 1, 2, 3, 4, 5, 6]), [0, 2, 4, 6])
        self.assertEqual(nms_indices_numpy(np.empty((0, 8)), .1).shape, (0,))

    def test_nms_rejects_invalid_shape(self):
        for shape in ((14,), (7, 6), (1, 1, 7)):
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, 'shape'):
                nms_indices_numpy(np.zeros(shape), .1)


if __name__ == '__main__':
    unittest.main()
