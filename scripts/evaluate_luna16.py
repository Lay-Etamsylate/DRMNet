"""Convert saved DRMNet voxel predictions to world coordinates and run FROC.

Uses the full test list, including scans without detections. Original CSVs
are never overwritten. Run --prepare-only to validate/convert without sklearn.
"""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.luna_eval import FROC_POINTS, cropped_xyz_to_world

WORLD_FIELDS = ['seriesuid', 'coordX', 'coordY', 'coordZ', 'probability']
ANNO_FIELDS = ['seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm']
KINDS = ('rpn', 'rcnn', 'ensemble')


def read_csv(path, required):
    with Path(path).open(newline='') as f:
        reader = csv.DictReader(f)
        if not set(required) <= set(reader.fieldnames or []):
            raise ValueError(f'{path}: missing columns {set(required) - set(reader.fieldnames or [])}')
        return list(reader)


def write_csv(path, fields, rows):
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def validate_annotations(rows, official_rows, ids, metadata, tolerance=1e-3):
    """Check both voxel->world round trip and agreement with official truth."""
    actual, expected = defaultdict(list), defaultdict(list)
    errors = []
    for row in rows:
        uid = row['pid']
        if uid not in ids:
            raise ValueError(f'Annotation outside test split: {uid}')
        center = [[(float(row[f'{axis}min']) + float(row[f'{axis}max'])) / 2
                   for axis in ('x', 'y', 'z')]]
        world = cropped_xyz_to_world(center, metadata[uid])[0]
        reference = np.array([float(row[k]) for k in ANNO_FIELDS[1:4]])
        errors.append(float(np.max(np.abs(world - reference))))
        actual[uid].append([float(row[k]) for k in ANNO_FIELDS[1:]])
    for row in official_rows:
        if row['seriesuid'] in ids:
            expected[row['seriesuid']].append([float(row[k]) for k in ANNO_FIELDS[1:]])
    if not errors or not np.isfinite(errors).all() or max(errors) > tolerance:
        raise ValueError(f'Annotation coordinate check failed; max error={max(errors, default=None)} mm')
    for uid in ids:
        a, b = sorted(actual[uid]), sorted(expected[uid])
        if len(a) != len(b) or (a and not np.allclose(a, b, atol=tolerance, rtol=0)):
            raise ValueError(f'Preprocessed annotations differ from official truth for {uid}')
    return len(errors), max(errors)


def check_coverage(pred_dir, ids, predictions, assume_complete=False):
    manifest_path = pred_dir / 'inference_manifest.json'
    if not manifest_path.is_file():
        if not assume_complete:
            raise ValueError('No inference_manifest.json: prediction CSVs cannot prove that all scans ran. '
                             'Use updated test.py, or --assume-complete ONLY after checking the old run completed every test scan.')
        return 'user_asserted_complete_legacy_run'
    manifest = json.loads(manifest_path.read_text())
    completed = manifest.get('seriesuids', [])
    if (manifest.get('status') != 'complete' or set(completed) != ids
            or len(completed) != len(ids) or manifest.get('coordinate_system') != 'cropped_voxel_xyz'):
        raise ValueError('Inference manifest is incomplete, has the wrong split, or uses unexpected coordinates')
    for kind, rows in predictions.items():
        counts = dict.fromkeys(ids, 0)
        for row in rows:
            counts[row['seriesuid']] += 1
        if manifest.get('prediction_counts', {}).get(kind) != counts:
            raise ValueError(f'{kind}: CSV counts do not match the inference manifest')
    return 'manifest_verified'


def read_test_ids(path):
    if path.suffix == '.npy':
        ids_array = np.load(path, allow_pickle=True)
        if ids_array.ndim != 1:
            raise ValueError('test-set must be a one-dimensional seriesuid array')
        uid_list = [str(uid) for uid in ids_array]
    else:
        uid_list = path.read_text().splitlines()
    uid_list = [uid.strip() for uid in uid_list]
    if not uid_list or '' in uid_list or len(set(uid_list)) != len(uid_list):
        raise ValueError('test-set is empty or has blank or duplicate scan IDs')
    return uid_list


def read_exclusions(annotations, excluded_path=None):
    if excluded_path is None:
        candidates = sorted((annotations.parent / 'evaluationScript').rglob('annotations_excluded.csv'))
        candidates = [p for p in candidates if read_csv(p, ANNO_FIELDS)]
        if len(candidates) != 1:
            raise ValueError('Cannot locate one nonempty official annotations_excluded.csv. '
                             'Pass --excluded /path/to/the/file from the LUNA16 evaluation download.')
        excluded_path = candidates[0]
    excluded = read_csv(excluded_path, ANNO_FIELDS)
    if not excluded:
        raise ValueError(f'{excluded_path} has no exclusion annotations (the repository file is a placeholder). '
                         'Use the nonempty file from the LUNA16 evaluation download.')
    return excluded_path, excluded


def prepare(args):
    uid_list = read_test_ids(args.test_set)
    ids = set(uid_list)
    metadata = {}
    for uid in uid_list:
        with np.load(args.meta_dir / f'{uid}_meta.npz', allow_pickle=False) as archive:
            metadata[uid] = {key: archive[key] for key in archive.files}
        if str(metadata[uid]['seriesuid'].item()) != uid:
            raise ValueError(f'Metadata seriesuid mismatch: {uid}')
        cropped_xyz_to_world(np.zeros((1, 3)), metadata[uid])

    preprocessed_path = args.preprocessed_annotations or args.test_set.with_name(args.test_set.stem + '_annos.csv')
    required = ['pid', 'xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'] + ANNO_FIELDS[1:]
    preprocessed = read_csv(preprocessed_path, required)
    official = read_csv(args.annotations, ANNO_FIELDS)
    count, error = validate_annotations(preprocessed, official, ids, metadata)

    excluded_path, excluded = read_exclusions(args.annotations, args.excluded)

    predictions, converted = {}, {}
    for kind in KINDS:
        rows = read_csv(args.pred_dir / f'submission_{kind}.csv', WORLD_FIELDS)
        outside = {row['seriesuid'] for row in rows} - ids
        if outside:
            raise ValueError(f'{kind}: predictions outside test split: {sorted(outside)[:3]}')
        predictions[kind] = rows
        converted[kind] = []
        for row in rows:
            uid = row['seriesuid']
            xyz = [[float(row[k]) for k in WORLD_FIELDS[1:4]]]
            xyz_world = cropped_xyz_to_world(xyz, metadata[uid])[0]
            score = float(row['probability'])
            if not np.isfinite(score):
                raise ValueError(f'{kind}: non-finite score')
            converted[kind].append(dict(zip(WORLD_FIELDS, [uid, *xyz_world, score])))

    coverage = check_coverage(args.pred_dir, ids, predictions, args.assume_complete)
    if args.out_dir.resolve() == args.pred_dir.resolve():
        raise ValueError('out-dir must differ from pred-dir to preserve voxel CSVs')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for kind in KINDS:
        write_csv(args.out_dir / f'submission_{kind}.csv', WORLD_FIELDS, converted[kind])
    write_csv(args.out_dir / 'annotations.csv', ANNO_FIELDS,
              [r for r in official if r['seriesuid'] in ids])
    write_csv(args.out_dir / 'annotations_excluded.csv', ANNO_FIELDS,
              [r for r in excluded if r['seriesuid'] in ids])
    (args.out_dir / 'seriesuids.csv').write_text(''.join(uid + '\n' for uid in uid_list))
    report = {'scans': len(ids), 'nodules': count, 'coordinate_check_max_error_mm': error,
              'coverage': coverage, 'excluded_source': str(excluded_path.resolve()),
              'excluded_test_rows': sum(r['seriesuid'] in ids for r in excluded),
              'coordinate_system': 'world_xyz_mm', 'bootstrap_samples': 1000,
              'seed': args.seed, 'max_candidates_per_scan': 100,
              'prediction_rows': {k: len(v) for k, v in predictions.items()},
              'scans_with_predictions': {k: len({r['seriesuid'] for r in v}) for k, v in predictions.items()}}
    (args.out_dir / 'evaluation_inputs.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pred-dir', type=Path, required=True, help='FROC directory containing voxel CSVs')
    parser.add_argument('--meta-dir', type=Path, required=True)
    parser.add_argument('--test-set', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, required=True, help='official LUNA16 annotations.csv')
    parser.add_argument('--preprocessed-annotations', type=Path)
    parser.add_argument('--excluded', type=Path, help='nonempty official annotations_excluded.csv')
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--assume-complete', action='store_true',
                        help='assert an old run without a manifest completed ALL test scans, including zero detections')
    parser.add_argument('--seed', type=int, default=35202)
    parser.add_argument('--matching-rule', choices=('strict', 'legacy'), default='legacy',
                        help='strict: distance < GT radius; legacy: squared distance < squared radius + 5')
    parser.add_argument('--confusion-threshold', type=float,
                        help='also report lesion-level TP, FP and FN at this score threshold')
    args = parser.parse_args()
    if args.confusion_threshold is not None:
        if not 0 <= args.confusion_threshold <= 1:
            parser.error('--confusion-threshold must be in [0, 1]')
        if args.prepare_only:
            parser.error('--confusion-threshold requires evaluation; remove --prepare-only')
    # Preserve previous legacy results for direct comparison with this rerun.
    if (args.out_dir / 'summary.json').exists():
        parser.error('--out-dir already contains summary.json; choose a new directory')
    prepare(args)
    if args.prepare_only:
        return
    from evaluationScript.noduleCADEvaluationLUNA16 import noduleCADEvaluation
    result_rows = []
    for kind in KINDS:
        target = args.out_dir / kind
        target.mkdir(exist_ok=True)
        np.random.seed(args.seed)
        print(f'\n===== {kind} =====', flush=True)
        boot, cpm_boot, raw, cpm_raw = noduleCADEvaluation(
            str(args.out_dir / 'annotations.csv'), str(args.out_dir / 'annotations_excluded.csv'),
            str(args.out_dir / 'seriesuids.csv'), str(args.out_dir / f'submission_{kind}.csv'), str(target),
            matching_rule=args.matching_rule, confusion_threshold=args.confusion_threshold)
        result_rows.append(dict(model=kind, cpm_raw=float(cpm_raw), cpm_bootstrap=float(cpm_boot),
                                raw=dict(zip(map(str, FROC_POINTS), raw)),
                                bootstrap=dict(zip(map(str, FROC_POINTS), boot))))
        print(f'{kind} CPM (raw): {cpm_raw:.9f}; CPM (bootstrap): {cpm_boot:.9f}')
    (args.out_dir / 'summary.json').write_text(json.dumps(result_rows, indent=2, allow_nan=False) + '\n')
    protocol = dict(matching_rule=args.matching_rule,
                    squared_distance_tolerance_mm2=5. if args.matching_rule == 'legacy' else 0.,
                    seed=args.seed, bootstrap_samples=1000, max_candidates_per_scan=100,
                    seven_point_sampling='linear interpolation with endpoint extension',
                    prediction_source=str(args.pred_dir.resolve()), seriesuids_source=str(args.test_set.resolve()))
    (args.out_dir / 'evaluation_protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    lines = ['Matching rule: ' + args.matching_rule, '']
    for result in result_rows:
        for curve in ('raw', 'bootstrap'):
            lines.append(f"[{result['model']} / {curve}]")
            lines.extend(f'fps: {point:g}, sensitivity: {result[curve][str(point)]:.9f}' for point in FROC_POINTS)
            cpm_key = 'cpm_raw' if curve == 'raw' else 'cpm_bootstrap'
            lines.append(f"CPM: {result[cpm_key]:.9f}\n")
    (args.out_dir / 'metrics.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
