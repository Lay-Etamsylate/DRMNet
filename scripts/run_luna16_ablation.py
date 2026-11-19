"""Train DRMNet variants and evaluate a checkpoint on validation or test scans."""

import argparse
import csv
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))

from net.model_config import ABLATION_VARIANTS, VARIANTS, build_model_config
from scripts.evaluate_luna16 import ANNO_FIELDS, read_csv, read_exclusions, read_test_ids
from utils.luna_eval import FROC_POINTS


def checkpoint(run_dir, epoch):
    return run_dir / 'model' / f'{epoch:03d}.ckpt'


def prediction_dir(args, variant, split='test'):
    return args.out_dir / variant / f'{split}_predictions'


def world_dir(args, variant, split='test'):
    return prediction_dir(args, variant, split) / 'res' / str(args.epochs) / f'FROC_world_{args.matching_rule}'


def train_commands(args, variant):
    run_dir = args.out_dir / variant
    filelists = args.data_dir / 'filelists'
    common = [
        'train.py', '--variant', variant, '--out-dir', run_dir,
        '--train-set', filelists / 'train.npy', '--val-set', filelists / 'val.npy',
        '--train-anno', filelists / 'train_annos.csv', '--val-anno', filelists / 'val_annos.csv',
        '--train-data-dir', args.data_dir, '--val-data-dir', args.data_dir,
        '--seed', args.seed, '--batch-size', args.batch_size, '--num-workers', args.num_workers,
        '--epochs', args.epochs, '--epoch-rcnn', args.epoch_rcnn, '--epoch-save', 1,
        '--optimizer', 'SGD', '--momentum', 0.9, '--weight-decay', 0.0001, '--max-grad-norm', 5,
        '--head-activation', args.head_activation, '--auxiliary-losses', args.auxiliary_losses,
        '--scale-augmentation', args.scale_augmentation,
        '--rpn-train-pre-topk', 2000, '--rpn-train-post-topk', 300,
    ]
    if args.rpn_positives_per_nodule is not None:
        common.extend(['--rpn-positives-per-nodule', args.rpn_positives_per_nodule])
    warmup_epochs = args.epoch_rcnn - 1
    resume = []
    if warmup_epochs:
        yield [*common, '--init-lr', 0.001, '--stop-after-epoch', warmup_epochs]
        resume = ['--ckpt', checkpoint(run_dir, warmup_epochs)]
    yield [*common, '--init-lr', 0.0001, *resume]


def evaluation_commands(args, variant, split):
    predictions = prediction_dir(args, variant, split)
    filelists = args.data_dir / 'filelists'
    test_set = filelists / f'{split}.npy'
    test_anno = filelists / f'{split}_annos.csv'
    yield [
        'test.py', '--mode', 'eval', '--inference-only', '--variant', variant,
        '--weight', checkpoint(args.out_dir / variant, args.epochs),
        '--test-set-name', test_set, '--test-anno', test_anno,
        '--preprocessed-data-dir', args.data_dir, '--out-dir', predictions,
    ]
    excluded = ['--excluded', args.excluded] if args.excluded else []
    yield [
        'scripts/evaluate_luna16.py', '--pred-dir', predictions / 'res' / str(args.epochs) / 'FROC',
        '--test-set', test_set, '--preprocessed-annotations', test_anno, '--meta-dir', args.data_dir,
        '--annotations', args.luna_root / 'annotations.csv', *excluded,
        '--matching-rule', args.matching_rule, '--out-dir', world_dir(args, variant, split),
    ]


def run_commands(commands, env, dry_run):
    for command in commands:
        argv = [sys.executable, '-u', *map(str, command)]
        print(shlex.join(argv), flush=True)
        if not dry_run:
            subprocess.run(argv, check=True, cwd=REPO_DIR, env=env)


def require_file(path):
    if not path.is_file():
        raise FileNotFoundError(f'Missing file: {path}')


def check_cohort(root, split, ids):
    subsets, size = {'train': (range(2, 10), 710), 'val': ([1], 89), 'test': ([0], 89)}[split]
    if len(ids) != size:
        raise ValueError(f'{split}: expected {size} scans, found {len(ids)}')
    directories = [root / f'subset{i}' for i in subsets]
    if any(directory.is_dir() for directory in directories):
        raw_ids = [path.stem for directory in directories for path in directory.rglob('*.mhd')]
        if len(raw_ids) != size or set(raw_ids) != set(ids):
            raise ValueError(f'{split}: filelist does not match the scans in {directories}')
    else:
        print(f'{split}: {size} unique scans; raw subset directories unavailable for UID verification')


def check_inputs(args, variants, training, eval_split):
    filelists = args.data_dir / 'filelists'
    active_splits = (('train', 'val') if training else ()) + ((eval_split,) if eval_split else ())
    used = set()
    for split in ('train', 'val', 'test'):
        split_file = filelists / f'{split}.npy'
        if split not in active_splits and not split_file.is_file():
            continue
        ids = read_test_ids(split_file)
        overlap = used.intersection(ids)
        if overlap:
            raise ValueError(f'{split} overlaps another split ({len(overlap)} scans)')
        used.update(ids)
        if split not in active_splits:
            continue
        check_cohort(args.luna_root, split, ids)
        require_file(filelists / f'{split}_annos.csv')
        for uid in ids:
            require_file(args.data_dir / f'{uid}_clean.npy')
            if split == eval_split:
                require_file(args.data_dir / f'{uid}_meta.npz')
    if eval_split:
        annotations = args.luna_root / 'annotations.csv'
        read_csv(annotations, ANNO_FIELDS)
        args.excluded, _ = read_exclusions(annotations, args.excluded)
    for variant in variants:
        run_dir = args.out_dir / variant
        if training and any((run_dir / 'model').glob('*.ckpt')):
            raise ValueError(f'{run_dir} already has checkpoints. Use test, a new --out-dir, or train.py --ckpt.')
        if eval_split and not training:
            require_file(checkpoint(run_dir, args.epochs))
        if eval_split and (world_dir(args, variant, eval_split) / 'summary.json').exists():
            raise ValueError(f'{world_dir(args, variant, eval_split)} is already evaluated.')


def write_summary(args, variants=ABLATION_VARIANTS, split='test'):
    directories = [world_dir(args, variant, split) for variant in variants]
    if not all((directory / 'summary.json').is_file() for directory in directories):
        return
    for name in ('seriesuids.csv', 'annotations.csv', 'annotations_excluded.csv'):
        first = (directories[0] / name).read_bytes()
        if any((directory / name).read_bytes() != first for directory in directories[1:]):
            raise ValueError(f'Cannot aggregate: evaluation inputs differ for {name}')

    columns = ['variant', 'multidirection', 'freqfusion', 'head', 'epoch', 'matching_rule',
               'cpm_raw', 'cpm_bootstrap', *(f'sensitivity_at_{p:g}_fp_per_scan' for p in FROC_POINTS)]
    rows = []
    for variant, directory in zip(variants, directories):
        cfg = build_model_config({}, variant)
        metrics = json.loads((directory / 'summary.json').read_text())
        for result in metrics:
            rows.append([
                variant, int(bool(cfg['multidirection_stages'])), int(bool(cfg['freqfusion_stages'])),
                result['model'], args.epochs, args.matching_rule, result['cpm_raw'], result['cpm_bootstrap'],
                *(result['raw'][str(p)] for p in FROC_POINTS),
            ])
    prefix = variants[0] if len(variants) == 1 else 'ablation'
    path = args.out_dir / f'{prefix}_{split}_{args.epochs:03d}_{args.matching_rule}.csv'
    with path.open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        writer.writerows(rows)
    print(f'{split} comparison: {path}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('train', 'val', 'train-val', 'test', 'all'))
    parser.add_argument('--luna-root', required=True, type=Path,
                        help='LUNA16 directory containing official annotations.csv')
    parser.add_argument('--data-dir', type=Path, help='defaults to LUNA_ROOT/drmnet_preprocessed')
    parser.add_argument('--excluded', type=Path, help='official annotations_excluded.csv')
    parser.add_argument('--out-dir', type=Path, default=Path('results/luna16_ablation'))
    parser.add_argument('--variant', choices=VARIANTS, help='run one variant (default: the four original ablations)')
    parser.add_argument('--rpn-positives-per-nodule', type=int, help='train with up to this many anchors per nodule')
    parser.add_argument('--epochs', type=int, default=80, help='total epochs and final checkpoint to test')
    parser.add_argument('--epoch-rcnn', type=int, default=21, help='first epoch that trains RCNN')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=35202)
    parser.add_argument('--head-activation', choices=('relu', 'leaky_relu'), default='leaky_relu')
    parser.add_argument('--auxiliary-losses', choices=('on', 'off'), default='on')
    parser.add_argument('--scale-augmentation', choices=('on', 'off'), default='off')
    parser.add_argument('--matching-rule', choices=('strict', 'legacy'), default='strict')
    parser.add_argument('--dry-run', action='store_true', help='print the commands without running them')
    args = parser.parse_args()
    training = args.stage in ('train', 'train-val', 'all')
    if args.rpn_positives_per_nodule is not None and args.rpn_positives_per_nodule < 0:
        parser.error('--rpn-positives-per-nodule must be nonnegative')
    if args.epochs < 1 or (training and not 1 <= args.epoch_rcnn <= args.epochs):
        parser.error('require --epochs >= 1 and, for training, 1 <= --epoch-rcnn <= --epochs')
    if args.batch_size < 1 or args.num_workers < 0 or not 0 <= args.seed < 2**32:
        parser.error('require --batch-size >= 1, --num-workers >= 0 and 0 <= --seed < 2**32')

    args.luna_root = args.luna_root.expanduser().resolve()
    args.data_dir = (args.data_dir or args.luna_root / 'drmnet_preprocessed').expanduser().resolve()
    args.out_dir = args.out_dir.expanduser().resolve()
    if args.excluded:
        args.excluded = args.excluded.expanduser().resolve()
    variants = (args.variant,) if args.variant else ABLATION_VARIANTS
    eval_split = {'val': 'val', 'train-val': 'val', 'test': 'test', 'all': 'test', 'train': None}[args.stage]
    env = os.environ.copy()
    env.setdefault('MPLBACKEND', 'Agg')
    env.setdefault('CUDA_VISIBLE_DEVICES', '0')
    try:
        if not args.dry_run:
            check_inputs(args, variants, training, eval_split)
        if training:
            for variant in variants:
                run_commands(train_commands(args, variant), env, args.dry_run)
        if eval_split:
            for variant in variants:
                run_commands(evaluation_commands(args, variant, eval_split), env, args.dry_run)
        if eval_split and not args.dry_run:
            write_summary(args, variants, eval_split)
            if args.variant in ABLATION_VARIANTS:
                write_summary(args, ABLATION_VARIANTS, eval_split)
    except subprocess.CalledProcessError as exc:
        parser.exit(max(1, exc.returncode), f'Command failed: {shlex.join(exc.cmd)}\n')
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
