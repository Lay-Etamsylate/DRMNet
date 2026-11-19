"""CPU fallback for the legacy box extension (continuous center-size boxes).

Matches the C kernels' >= NMS suppression threshold; no +1 voxel term.
Only used for non-differentiable target matching and proposal suppression.
"""

import numpy as np


def overlap_numpy(boxes1, boxes2, chunk_size=2048):
    boxes1 = np.asarray(boxes1, dtype=np.float32).reshape(-1, 6)
    boxes2 = np.asarray(boxes2, dtype=np.float32).reshape(-1, 6)
    result = np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)
    if not len(boxes1) or not len(boxes2):
        return result
    lo2 = boxes2[:, :3] - boxes2[:, 3:] / 2
    hi2 = lo2 + boxes2[:, 3:]
    vol2 = boxes2[:, 3:].prod(axis=1)
    for start in range(0, len(boxes1), chunk_size):
        a = boxes1[start:start + chunk_size]
        lo1 = a[:, :3] - a[:, 3:] / 2
        hi1 = lo1 + a[:, 3:]
        extent = np.maximum(0, np.minimum(hi1[:, None], hi2)
                            - np.maximum(lo1[:, None], lo2))
        inter = extent.prod(axis=2)
        union = a[:, 3:].prod(axis=1)[:, None] + vol2 - inter
        np.divide(inter, union, out=result[start:start + len(a)], where=union > 0)
    return result


def nms_indices_numpy(dets, threshold, order=None):
    """Suppress using score + six box columns; ignore trailing metadata.

    RPN rows have seven columns, while RCNN appends a class ID. Returned
    indices always refer to the original rows, as in the legacy C kernel.
    """
    dets = np.asarray(dets, dtype=np.float32)
    if dets.ndim != 2 or dets.shape[1] < 7:
        raise ValueError('dets must have shape (N, >=7): score,z,y,x,d,h,w[,metadata]')
    boxes = dets[:, 1:7]
    if order is None:
        order = np.argsort(-dets[:, 0], kind='stable')
    else:
        order = np.asarray(order, dtype=np.int64).copy()
    keep = []
    while len(order):
        index = int(order[0])
        keep.append(index)
        rest = order[1:]
        overlaps = overlap_numpy(boxes[index:index + 1], boxes[rest])[0]
        order = rest[overlaps < threshold]
    return np.asarray(keep, dtype=np.int64)
