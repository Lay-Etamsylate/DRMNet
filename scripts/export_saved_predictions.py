"""Rebuild prediction CSVs after inference, without loading a model or using CUDA."""

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.prediction_export import export_saved_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save-dir', type=Path, required=True,
                        help='epoch result directory containing *_rpns/rcnns/ensembles.npy')
    parser.add_argument('--seriesuids', type=Path, required=True, help='the exact inference split .npy')
    args = parser.parse_args()
    ids = np.load(args.seriesuids, allow_pickle=True)
    if ids.ndim != 1:
        parser.error('--seriesuids must contain a one-dimensional scan list')
    manifest = export_saved_predictions(args.save_dir, ids)
    print(f'Exported {len(ids)} scans to {args.save_dir / "FROC"} (cropped voxel coordinates)')
    for kind, counts in manifest['prediction_counts'].items():
        print(f'{kind}: {sum(counts.values())} predictions; '
              f'{sum(n == 0 for n in counts.values())} scans with zero detections')


if __name__ == '__main__':
    main()
