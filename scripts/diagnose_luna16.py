"""Read-only diagnosis of saved predictions, FROC inputs and a checkpoint."""

import argparse
from collections import Counter, defaultdict, deque
import csv
import json
from pathlib import Path

import numpy as np

XYZ = ('coordX', 'coordY', 'coordZ')


def read_rows(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def describe(values):
    a = np.asarray(values, dtype=np.float64)
    finite = a[np.isfinite(a)]
    return dict(count=int(a.size), nonfinite=int(a.size - finite.size),
                min=float(finite.min()) if finite.size else None,
                median=float(np.median(finite)) if finite.size else None,
                max=float(finite.max()) if finite.size else None,
                std=float(finite.std()) if finite.size else None,
                unique=int(np.unique(finite).size))


def matching_report(predictions, annotations, seriesuids):
    """Independent all-score geometric check; does not compute ranked FROC."""
    ids = set(seriesuids)
    outside = sorted({r['seriesuid'] for r in predictions} - ids)
    grouped = defaultdict(list)
    for row in predictions:
        grouped[row['seriesuid']].append([float(row[k]) for k in XYZ])
    hits = legacy_hits = no_candidates = 0
    distances = []
    ratios = []
    for row in annotations:
        if row['seriesuid'] not in ids:
            continue
        candidates = grouped[row['seriesuid']]
        if not candidates:
            no_candidates += 1
            continue
        center = np.array([float(row[k]) for k in XYZ])
        radius = float(row['diameter_mm']) / 2
        distance2 = np.square(np.asarray(candidates) - center).sum(axis=1).min()
        distances.append(float(np.sqrt(distance2)))
        ratios.append(float(np.sqrt(distance2) / radius) if radius > 0 else np.nan)
        hits += int(distance2 < radius ** 2)
        # The inherited project evaluator currently includes this extra margin.
        legacy_hits += int(distance2 < radius ** 2 + 5)
    return dict(scans=len(ids), reference_nodules=sum(r['seriesuid'] in ids for r in annotations),
                predictions_outside_split=outside, nodules_in_scans_without_predictions=no_candidates,
                geometric_hits_strict=hits, geometric_hits_legacy_margin=legacy_hits,
                nearest_distance_mm=describe(distances), nearest_distance_over_radius=describe(ratios))


def box_multiset(rows):
    return Counter((r['seriesuid'], *(float(r[k]) for k in (*XYZ, 'diameter_mm'))) for r in rows)


def checkpoint_report(path):
    import torch
    checkpoint = torch.load(path, map_location='cpu')
    state = checkpoint['state_dict']
    cfg = checkpoint.get('model_config', {})
    report = dict(path=str(path), epoch=checkpoint.get('epoch'),
                  model_variant=cfg.get('model_variant', 'legacy/unknown'),
                  thresholds={k: v for k, v in cfg.items() if 'pre_score_threshold' in k},
                  heads={}, nonfinite_tensors=[])
    prefixes = ('rpn.conv.', 'rpn.logits.', 'rpn.deltas.', 'rcnn_head.')
    for key, value in state.items():
        value = value.detach().cpu()
        if value.is_floating_point() and not torch.isfinite(value).all():
            report['nonfinite_tensors'].append(key)
        if key.startswith(prefixes):
            a = value.float().numpy()
            finite = a[np.isfinite(a)]
            report['heads'][key] = dict(shape=list(a.shape),
                                        mean=float(finite.mean()) if finite.size else None,
                                        std=float(finite.std()) if finite.size else None,
                                        absmax=float(np.abs(finite).max()) if finite.size else None,
                                        zero_fraction=float((a == 0).mean()))
    bias = state.get('rcnn_head.logit.bias')
    if bias is not None:
        report['rcnn_probability_from_bias_alone'] = [v if np.isfinite(v) else None
                                                     for v in torch.softmax(bias.float(), dim=0).tolist()]
    for key in ('rpn.logits.bias', 'rpn.deltas.bias', 'rcnn_head.delta.bias'):
        if key in state:
            report[key] = [v if np.isfinite(v) else None for v in state[key].tolist()]
    report['optimizer'] = [{k: group[k] for k in ('lr', 'momentum', 'weight_decay') if k in group}
                           for group in checkpoint.get('optimizer', {}).get('param_groups', [])]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pred-dir', type=Path, required=True, help='original voxel FROC directory')
    parser.add_argument('--world-dir', type=Path, help='defaults to pred-dir + _world')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--train-log', type=Path)
    args = parser.parse_args()
    world_dir = args.world_dir or Path(str(args.pred_dir) + '_world')
    report = {'voxel_predictions': {}, 'world_geometry': {}}
    original = {}
    for kind in ('rpn', 'rcnn', 'ensemble'):
        rows = read_rows(args.pred_dir / f'submission_{kind}.csv')
        original[kind] = rows
        report['voxel_predictions'][kind] = dict(rows=len(rows), scans=len({r['seriesuid'] for r in rows}),
                                                  scores=describe([float(r['probability']) for r in rows]))
    report['rpn_rcnn_same_boxes_ignoring_order'] = box_multiset(original['rpn']) == box_multiset(original['rcnn'])
    for filename in ('evaluation_inputs.json', 'summary.json'):
        path = world_dir / filename
        if path.is_file():
            content = json.loads(path.read_text())
            report[filename] = ([{k: row[k] for k in ('model', 'cpm_raw', 'cpm_bootstrap')} for row in content]
                                if filename == 'summary.json' else content)
    if (world_dir / 'annotations.csv').is_file():
        annotations = read_rows(world_dir / 'annotations.csv')
        ids = [line.strip() for line in (world_dir / 'seriesuids.csv').read_text().splitlines() if line.strip()]
        for kind in ('rpn', 'rcnn', 'ensemble'):
            rows = read_rows(world_dir / f'submission_{kind}.csv')
            report['world_geometry'][kind] = matching_report(rows, annotations, ids)
            analysis_path = world_dir / kind / 'CADAnalysis.txt'
            if analysis_path.is_file():
                report['world_geometry'][kind]['CADAnalysis'] = analysis_path.read_text()
    if args.checkpoint:
        report['checkpoint'] = checkpoint_report(args.checkpoint)
    print(json.dumps(report, indent=2, allow_nan=False))
    if args.train_log:
        needles = ('Train Epoch', 'Val Epoch', 'rpn_cls ', 'rpn_stats:', 'rcnn_stats:', 'use_rcnn:')
        with args.train_log.open(errors='replace') as f:
            recent = deque((line.rstrip() for line in f if any(s in line for s in needles)), maxlen=30)
        print('\nRecent training/validation log entries:')
        print('\n'.join(recent))


if __name__ == '__main__':
    main()
