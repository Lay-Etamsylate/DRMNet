"""Portable 3-D box overlap and non-maximum suppression helpers.

The original project imported a C extension named ``box``. That extension is
built with ``torch.utils.ffi``, which is unavailable in modern PyTorch. The
detector moves proposals to CPU before these calls, so NumPy is sufficient.

Boxes use ``[z, y, x, d, h, w]``. NMS inputs prepend a score:
``[score, z, y, x, d, h, w, ...]``.
"""

from __future__ import annotations

import numpy as np
import torch


def _as_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def py_box_overlap(boxes1, boxes2):
    """Return pairwise 3-D IoU using center/size box coordinates."""
    a = _as_numpy(boxes1).astype(np.float32, copy=False).reshape(-1, 6)
    b = _as_numpy(boxes2).astype(np.float32, copy=False).reshape(-1, 6)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)

    a_min = a[:, None, :3] - a[:, None, 3:6] / 2.0
    a_max = a[:, None, :3] + a[:, None, 3:6] / 2.0
    b_min = b[None, :, :3] - b[None, :, 3:6] / 2.0
    b_max = b[None, :, :3] + b[None, :, 3:6] / 2.0
    intersection_size = np.maximum(
        0.0, np.minimum(a_max, b_max) - np.maximum(a_min, b_min)
    )
    intersection = np.prod(intersection_size, axis=2)
    volume_a = np.prod(np.maximum(a[:, 3:6], 0.0), axis=1)[:, None]
    volume_b = np.prod(np.maximum(b[:, 3:6], 0.0), axis=1)[None, :]
    union = volume_a + volume_b - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=np.float32),
        where=union > 0,
    )


def torch_overlap(boxes1, boxes2):
    """Compatibility alias used by the RPN/RCNN target builders."""
    return py_box_overlap(boxes1, boxes2)


def py_nms(detections, threshold):
    """Apply greedy 3-D IoU NMS and return rows and original indices."""
    dets = _as_numpy(detections).astype(np.float32, copy=False)
    if dets.ndim != 2 or dets.shape[1] < 7:
        raise ValueError(
            "detections must be [N, >=7]: [score, z, y, x, d, h, w, ...]"
        )
    if len(dets) == 0:
        return dets.copy(), np.empty((0,), dtype=np.int64)

    order = np.argsort(-dets[:, 0], kind="stable")
    keep = []
    boxes = dets[:, 1:7]
    while order.size:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        remaining = order[1:]
        overlaps = py_box_overlap(boxes[current : current + 1], boxes[remaining])[0]
        # The legacy C implementation suppresses IoU >= threshold.
        order = remaining[overlaps < threshold]

    keep_array = np.asarray(keep, dtype=np.int64)
    return dets[keep_array], keep_array


def torch_nms(detections, threshold):
    """Return CPU tensors with the same interface as the legacy extension."""
    filtered, keep = py_nms(detections, threshold)
    return torch.from_numpy(filtered).float(), torch.from_numpy(keep).long()


__all__ = ["py_box_overlap", "py_nms", "torch_overlap", "torch_nms"]
