import csv
import json
import shlex
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.run_luna16_ablation import ABLATION_VARIANTS, check_cohort, main, world_dir, write_summary
from scripts.evaluate_luna16 import KINDS
from utils.luna_eval import FROC_POINTS


def test_dry_run_pairs_four_fresh_runs_with_their_own_checkpoints(tmp_path, capsys, monkeypatch):
    out_dir = tmp_path / 'runs with spaces'
    monkeypatch.setattr(sys, 'argv', [
        'run_luna16_ablation.py', 'all', '--luna-root', str(tmp_path),
        '--out-dir', str(out_dir), '--dry-run',
    ])
    main()
    commands = [shlex.split(line)[2:] for line in capsys.readouterr().out.splitlines()]
    assert len(commands) == 16
    assert not out_dir.exists()
    for i, variant in enumerate(ABLATION_VARIANTS):
        fresh, joint = commands[2 * i:2 * i + 2]
        inference, evaluation = commands[8 + 2 * i:10 + 2 * i]

        def option(command, flag):
            return command[command.index(flag) + 1]

        assert [cmd[0] for cmd in (fresh, joint, inference, evaluation)] == [
            'train.py', 'train.py', 'test.py', 'scripts/evaluate_luna16.py',
        ]
        assert all(option(cmd, '--variant') == variant for cmd in (fresh, joint, inference))
        assert '--ckpt' not in fresh and '--init-from' not in fresh
        assert option(fresh, '--stop-after-epoch') == '20'
        assert option(joint, '--ckpt') == str(out_dir / variant / 'model' / '020.ckpt')
        assert option(inference, '--weight') == str(out_dir / variant / 'model' / '080.ckpt')
        assert option(inference, '--test-set-name') == option(evaluation, '--test-set')
        assert option(evaluation, '--matching-rule') == 'strict'


def test_summary_uses_raw_froc_points_and_checks_shared_cohort(tmp_path):
    args = SimpleNamespace(out_dir=tmp_path, epochs=80, matching_rule='strict')
    for variant in ABLATION_VARIANTS:
        target = world_dir(args, variant)
        target.mkdir(parents=True)
        for name in ('seriesuids.csv', 'annotations.csv', 'annotations_excluded.csv'):
            (target / name).write_text('same data\n')
        rows = [dict(model=head, cpm_raw=0.6, cpm_bootstrap=0.5,
                     raw={str(p): 0.6 for p in FROC_POINTS}) for head in KINDS]
        (target / 'summary.json').write_text(json.dumps(rows))
    write_summary(args)
    with (tmp_path / 'ablation_test_080_strict.csv').open(newline='') as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 12
    assert [(row['variant'], row['multidirection'], row['freqfusion']) for row in rows
            if row['head'] == 'ensemble'] == [
        ('baseline', '0', '0'), ('mamba', '1', '0'), ('freqfusion', '0', '1'), ('fusion', '1', '1'),
    ]
    assert rows[-1]['sensitivity_at_4_fp_per_scan'] == '0.6'
    (world_dir(args, 'fusion') / 'seriesuids.csv').write_text('another cohort\n')
    with pytest.raises(ValueError, match='evaluation inputs differ'):
        write_summary(args)


def test_validation_of_new_variant_uses_only_subset1_predictions_and_truth(tmp_path, capsys, monkeypatch):
    common = ['--luna-root', str(tmp_path), '--out-dir', str(tmp_path / 'exp'),
              '--variant', 'fusion_cross_scale', '--dry-run']
    monkeypatch.setattr(sys, 'argv', ['runner', 'train-val', *common, '--rpn-positives-per-nodule', '3'])
    main()
    commands = capsys.readouterr().out.splitlines()
    assert len(commands) == 4
    assert all('--rpn-positives-per-nodule 3' in line for line in commands[:2])
    assert all('val.npy' in line and 'test.npy' not in line for line in commands)
    monkeypatch.setattr(sys, 'argv', ['runner', 'val', *common, '--epochs', '40'])
    main()
    validation = capsys.readouterr().out.splitlines()
    assert len(validation) == 2
    assert 'fusion_cross_scale/model/040.ckpt' in validation[0]
    assert all('val.npy' in cmd and 'val_annos.csv' in cmd and 'val_predictions' in cmd
               and 'test.npy' not in cmd for cmd in validation)


def test_cohort_rejects_interchanged_subsets_even_with_correct_size(tmp_path):
    directory = tmp_path / 'subset1'
    directory.mkdir()
    ids = [f'val{i}' for i in range(89)]
    for uid in ids:
        (directory / f'{uid}.mhd').touch()
    check_cohort(tmp_path, 'val', ids)
    with pytest.raises(ValueError, match='does not match'):
        check_cohort(tmp_path, 'val', ['test0', *ids[1:]])
    with pytest.raises(ValueError, match='expected 710'):
        check_cohort(tmp_path, 'train', ids)


def test_seeded_reader_crops_are_independent_of_global_random_state(tmp_path):
    pytest.importorskip('torch')
    from dataset.bbox_reader import BboxReader

    scan = np.arange(32**3, dtype=np.float32).reshape(1, 32, 32, 32)
    np.save(tmp_path / 'scan_clean.npy', scan)
    split = tmp_path / 'train.npy'
    np.save(split, ['scan'])
    annotations = tmp_path / 'train_annos.csv'
    annotations.write_text('pid,zmin,zmax,ymin,ymax,xmin,xmax\nscan,14,18,14,18,14,18\n')
    cfg = dict(crop_seed=35202, blacklist=[], r_rand_crop=0., pad_value=170, stride=4,
               crop_size=[16, 16, 16], bound_size=4, bbox_border=2, batch_size=1,
               augtype=dict(scale=False, flip=True, rotate=False, swap=False))
    reader = BboxReader(str(tmp_path), str(split), str(annotations), cfg, 1, mode='train')
    reader.epoch = 1
    first = reader[0][0].numpy()
    np.random.rand(100)
    np.testing.assert_array_equal(reader[0][0].numpy(), first)
    reader.epoch = 2
    assert not np.array_equal(reader[0][0].numpy(), first)
