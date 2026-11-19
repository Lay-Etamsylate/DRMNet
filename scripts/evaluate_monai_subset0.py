"""Run the MONAI lung nodule bundle on raw LUNA16 subset0 and score CT scans."""

import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys


def series_uid(image):
    name = Path(image).name
    for suffix in ('.nii.gz', '.nii', '.mhd'):
        if name.endswith(suffix):
            return name.removesuffix(suffix)
    raise ValueError(f'Unexpected image filename: {image}')


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, data):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2)
        stream.write('\n')


def summarize(scans, positive, predictions, threshold):
    seen = set()
    predicted_positive = set()
    for row in predictions:
        uid = series_uid(row['image'])
        if uid in seen or uid not in scans:
            raise ValueError(f'Duplicate prediction record or scan outside subset0: {uid}')
        seen.add(uid)
        scores = row['label_scores']
        if any(not math.isfinite(score) or not 0 <= score <= 1 for score in scores):
            raise ValueError(f'Invalid detection score for {uid}')
        if any(score >= threshold for score in scores):
            predicted_positive.add(uid)

    if seen != scans:
        missing = sorted(scans - seen)
        raise ValueError(f'Missing prediction records for {len(missing)} scans: {missing[:3]}')

    tn = len(scans - (positive | predicted_positive))
    fp = len(predicted_positive - positive)
    fn = len(positive - predicted_positive)
    tp = len(positive & predicted_positive)
    return {
        'unit': 'scan',
        'subset': 'subset0',
        'score_threshold': threshold,
        'scans': len(scans),
        'scans_with_prediction_records': len(seen),
        'matrix_axes': ['actual', 'predicted'],
        'labels': ['negative', 'positive'],
        'matrix': [[tn, fp], [fn, tp]],
        'TN': tn,
        'FP': fp,
        'FN': fn,
        'TP': tp,
    }


def run_bundle(bundle, root, images, output):
    config = load_json(bundle / 'configs/inference.json')
    datalist = output / 'subset0.json'
    write_json(datalist, {'validation': [{'image': str(image)} for image in images]})
    predictions = output / 'predictions.json'
    config.update(
        bundle_root=str(bundle),
        dataset_dir=str(root),
        data_list_file_path=str(datalist),
        whether_raw_luna16=True,
        load_pretrain=True,
        output_dir=str(output),
        output_filename=predictions.name,
    )
    run_config = output / 'inference.json'
    write_json(run_config, config)
    subprocess.run(
        [sys.executable, '-m', 'monai.bundle', 'run', '--config_file', str(run_config)],
        cwd=bundle,
        check=True,
    )
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--luna-root', type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--bundle', type=Path, help='unpacked MONAI lung_nodule_ct_detection bundle')
    source.add_argument('--predictions', type=Path, help='existing MONAI DetectionSaver JSON for subset0')
    parser.add_argument('--out-dir', type=Path, required=True, help='new directory for this run')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--fold-json', type=Path, help='official MONAI dataset_fold0.json to check split overlap')
    args = parser.parse_args()
    if not 0.02 <= args.threshold <= 1:
        parser.error('--threshold must be in [0.02, 1] for the official bundle score filter')

    root = args.luna_root.resolve()
    images = sorted((root / 'subset0').rglob('*.mhd'))
    scans = {series_uid(image) for image in images}
    if not images or len(scans) != len(images):
        parser.error('Expected a nonempty subset0 directory with unique .mhd scan IDs')
    annotations = root / 'annotations.csv'
    with annotations.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if 'seriesuid' not in (reader.fieldnames or []):
            parser.error(f'Missing seriesuid column in {annotations}')
        positive = {row['seriesuid'] for row in reader} & scans
    if not positive:
        parser.error(f'No subset0 scan IDs match {annotations}')

    overlap = None
    if args.fold_json:
        fold = load_json(args.fold_json)
        overlap = {
            split: len(scans & {series_uid(row['image']) for row in fold[split]})
            for split in ('training', 'validation')
        }

    bundle = args.bundle.resolve() if args.bundle else None
    if bundle and not (bundle / 'models/model.pt').is_file():
        parser.error(f'Missing checkpoint: {bundle / "models/model.pt"}')
    output = args.out_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    predictions = run_bundle(bundle, root, images, output) if bundle else args.predictions.resolve()
    result = summarize(scans, positive, load_json(predictions), args.threshold)
    result.update(predictions=str(predictions), annotations=str(annotations), monai_fold0_overlap=overlap)
    if bundle:
        result['bundle_version'] = load_json(bundle / 'configs/metadata.json')['version']
        result['checkpoint'] = str(bundle / 'models/model.pt')
    write_json(output / 'confusion_matrix.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
