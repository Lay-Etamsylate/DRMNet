"""Compute a scan-level confusion matrix from saved LUNA16 predictions."""

import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_luna16 import KINDS, check_coverage, read_csv, read_test_ids


def scan_confusion(ids, annotations, predictions, threshold):
    """A scan is positive if it has an included nodule or a passing prediction, respectively."""
    actual = {row['seriesuid'] for row in annotations} & ids
    predicted = set()
    for row in predictions:
        score = float(row['probability'])
        if not math.isfinite(score):
            raise ValueError('Non-finite prediction score')
        if score >= threshold:
            predicted.add(row['seriesuid'])

    tp = len(actual & predicted)
    fp = len(predicted - actual)
    fn = len(actual - predicted)
    tn = len(ids - (actual | predicted))
    return dict(TN=tn, FP=fp, FN=fn, TP=tp, matrix=[[tn, fp], [fn, tp]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pred-dir', type=Path, required=True, help='FROC directory from test.py')
    parser.add_argument('--test-set', type=Path, required=True,
                        help='complete test.npy or a headerless list of scan IDs')
    parser.add_argument('--annotations', type=Path, required=True, help='official LUNA16 annotations.csv')
    parser.add_argument('--head', choices=KINDS, default='ensemble')
    parser.add_argument('--threshold', type=float, default=0.5, help='minimum candidate score (default: 0.5)')
    parser.add_argument('--assume-complete', action='store_true',
                        help='assert that a legacy run without a manifest processed all test scans')
    parser.add_argument('--out', type=Path, help='optional JSON output path')
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error('--threshold must be in [0, 1]')

    ids = set(read_test_ids(args.test_set))
    prediction_path = args.pred_dir / f'submission_{args.head}.csv'
    predictions = read_csv(prediction_path, ['seriesuid', 'probability'])
    outside = {row['seriesuid'] for row in predictions} - ids
    if outside:
        raise ValueError(f'Predictions outside test split: {sorted(outside)[:3]}')
    coverage = check_coverage(args.pred_dir, ids, {args.head: predictions}, args.assume_complete)
    annotations = read_csv(args.annotations, ['seriesuid'])
    report = dict(unit='scan', head=args.head, score_threshold=args.threshold, scans=len(ids),
                  coverage=coverage, labels=['negative', 'positive'],
                  matrix_axes=['actual', 'predicted'],
                  predictions=str(prediction_path.resolve()), test_set=str(args.test_set.resolve()),
                  annotations=str(args.annotations.resolve()),
                  **scan_confusion(ids, annotations, predictions, args.threshold))
    output = json.dumps(report, indent=2) + '\n'
    if args.out is not None:
        with args.out.open('x') as stream:
            stream.write(output)
    print(output, end='')


if __name__ == '__main__':
    main()
