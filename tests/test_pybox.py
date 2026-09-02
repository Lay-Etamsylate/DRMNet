import unittest

import numpy as np
import torch

from utils.pybox import py_box_overlap, py_nms, torch_nms


class PyBoxTest(unittest.TestCase):
    def test_overlap_identity_and_disjoint(self):
        boxes = np.asarray(
            [[0, 0, 0, 10, 10, 10], [50, 50, 50, 5, 5, 5]],
            dtype=np.float32,
        )
        overlap = py_box_overlap(boxes, boxes)
        np.testing.assert_allclose(np.diag(overlap), 1.0)
        self.assertEqual(float(overlap[0, 1]), 0.0)

    def test_nms_suppresses_overlapping_lower_score(self):
        detections = np.asarray(
            [
                [0.9, 0, 0, 0, 10, 10, 10],
                [0.8, 0, 0, 0, 10, 10, 10],
                [0.7, 50, 50, 50, 5, 5, 5],
            ],
            dtype=np.float32,
        )
        filtered, keep = py_nms(detections, 0.1)
        np.testing.assert_array_equal(keep, [0, 2])
        np.testing.assert_allclose(filtered[:, 0], [0.9, 0.7])

    def test_torch_wrapper(self):
        detections = torch.tensor(
            [[0.9, 0, 0, 0, 10, 10, 10], [0.8, 0, 0, 0, 10, 10, 10]]
        )
        filtered, keep = torch_nms(detections, 0.1)
        self.assertIsInstance(filtered, torch.Tensor)
        self.assertEqual(keep.tolist(), [0])


if __name__ == "__main__":
    unittest.main()
