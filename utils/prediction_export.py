"""Export saved per-scan detection arrays, including scans with zero detections."""

import csv
import json
from pathlib import Path

import numpy as np

KINDS = ('rpn', 'rcnn', 'ensemble')
COLUMNS = ['seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm', 'probability']


def export_saved_predictions(save_dir, seriesuids):
    """Require all scans/kinds before publishing a complete coverage manifest.

    NPY columns: score,z,y,x,d,h,w. Export coordinates remain cropped voxels;
    the historical diameter_mm column also remains in voxel units here.
    This checks file coverage, not the checkpoint provenance of existing arrays.
    """
    save_dir = Path(save_dir)
    ids = [str(uid) for uid in seriesuids]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('Expected a nonempty seriesuid list without duplicates')
    if any(Path(uid).name != uid or uid in ('.', '..') for uid in ids):
        raise ValueError('seriesuid must be a filename component')
    out = save_dir / 'FROC'
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / 'inference_manifest.json'
    # An interrupted/failed rebuild must not leave a stale complete manifest.
    manifest_path.unlink(missing_ok=True)
    counts = {kind: {} for kind in KINDS}
    rows = {kind: [] for kind in KINDS}
    for uid in ids:
        for kind in KINDS:
            path = save_dir / f'{uid}_{kind}s.npy'
            if not path.is_file():
                raise FileNotFoundError(f'Missing prediction file: {path}; absence is not zero detections')
            boxes = np.load(path, allow_pickle=False)
            # Support legacy empty arrays, but never reshape nonempty malformed data.
            if boxes.shape == (0,):
                boxes = boxes.reshape(0, 7)
            if boxes.ndim != 2 or boxes.shape[1] != 7:
                raise ValueError(f'{path}: expected (N, 7), got {boxes.shape}')
            if not np.issubdtype(boxes.dtype, np.number) or not np.isfinite(boxes).all():
                raise ValueError(f'{path}: expected finite numeric predictions')
            counts[kind][uid] = len(boxes)
            # Zero detections contribute no CSV rows, but remain in the manifest.
            rows[kind].extend([[uid, *box] for box in boxes[:, [3, 2, 1, 4, 0]].tolist()])
    for kind in KINDS:
        path = out / f'submission_{kind}.csv'
        temporary = path.with_suffix('.csv.tmp')
        with temporary.open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(COLUMNS)
            writer.writerows(rows[kind])
        temporary.replace(path)
    manifest = dict(status='complete', coordinate_system='cropped_voxel_xyz',
                    seriesuids=ids, prediction_counts=counts,
                    export_source='saved_per_scan_arrays')
    temporary = manifest_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(manifest, indent=2) + '\n')
    temporary.replace(manifest_path)
    return manifest
