# LUNA16 reproduction guide

This guide reproduces a single-domain LUNA16 detector and official FROC
evaluation. It does not reproduce the paper's multi-source domain-
generalization experiment, which also requires the other GPND source domains.

## 1. Environment

The tested reproduction target is Python 3.10, PyTorch 2.1.2+cu118,
`mamba-ssm==1.2.0.post1`, and `causal-conv1d==1.2.0.post2`. Install the matching
Mamba release wheels before installing `requirements-repro.txt`.

The legacy extensions below `net/lib/box` are not needed and must not be built.
`utils/pybox.py` provides the portable implementation used by training and
inference.

Verify the synchronized repository on the server:

```bash
cd /data/home/zpc/jc/DRMNet
CUDA_VISIBLE_DEVICES=0 python tools/smoke_test_model.py
```

Add `--backward` after the forward-only check succeeds.

## 2. Preprocess LUNA16

The command below uses subset0 as test, subset1 as validation, and the other
eight official subsets for training.

```bash
cd /data/home/zpc/jc/DRMNet
conda activate drmnet

python tools/prepare_luna16.py \
  --data-root /data/home/zpc/jc/data/LUNA16 \
  --output-dir /data/home/zpc/jc/data/LUNA16/drmnet_preprocessed \
  --test-fold 0 \
  --val-fold 1 \
  --workers 8
```

Expected outputs include:

```text
drmnet_preprocessed/
  <seriesuid>_clean.npy
  <seriesuid>_meta.npz
  annotations_preprocessed.csv
  scan_shapes.csv
  filelists/
    train.npy
    train_annos.csv
    val.npy
    val_annos.csv
    test.npy
    test_annos.csv
    test_seriesuids.csv
    split.json
```

Rerunning the command skips completed scans. Pass `--overwrite` only when the
preprocessing definition changes.

Validate all files, splits, boxes, and coordinate transforms before training:

```bash
python tools/validate_luna16_preprocessed.py \
  --preprocessed-dir /data/home/zpc/jc/data/LUNA16/drmnet_preprocessed
```

## 3. Train

The original RCNN feature distribution logic is not safe under its internal
multi-GPU `data_parallel` scheme. Start with one L20; its 46 GB memory is enough
for the default 128-cube, batch-size-2 configuration.

First run two training and validation batches as an end-to-end smoke test:

```bash
PREP=/data/home/zpc/jc/data/LUNA16/drmnet_preprocessed

CUDA_VISIBLE_DEVICES=3 python train.py \
  --train-data-dir "$PREP" \
  --train-set "$PREP/filelists/train.npy" \
  --train-anno "$PREP/filelists/train_annos.csv" \
  --val-data-dir "$PREP" \
  --val-set "$PREP/filelists/val.npy" \
  --val-anno "$PREP/filelists/val_annos.csv" \
  --out-dir results/luna16_smoke \
  --batch-size 2 \
  --num-workers 2 \
  --epochs 1 \
  --max-train-batches 2 \
  --max-val-batches 2
```

Do not set `--epoch-rcnn 1` for this smoke test: the RCNN stage is designed to
start after the RPN has trained for 20 epochs.

```bash
cd /data/home/zpc/jc/DRMNet
conda activate drmnet

PREP=/data/home/zpc/jc/data/LUNA16/drmnet_preprocessed

CUDA_VISIBLE_DEVICES=0 python train.py \
  --train-data-dir "$PREP" \
  --train-set "$PREP/filelists/train.npy" \
  --train-anno "$PREP/filelists/train_annos.csv" \
  --val-data-dir "$PREP" \
  --val-set "$PREP/filelists/val.npy" \
  --val-anno "$PREP/filelists/val_annos.csv" \
  --out-dir results/luna16_fold0 \
  --batch-size 2 \
  --num-workers 8 \
  --epochs 80 \
  --epoch-rcnn 21
```

Checkpoints are written to `results/luna16_fold0/model/`. To resume, add:

```bash
--ckpt results/luna16_fold0/model/040.ckpt
```

## 4. Smoke-test sliding-window inference

First evaluate one scan and skip FROC:

```bash
CUDA_VISIBLE_DEVICES=0 python eval_luna16.py \
  --checkpoint results/luna16_fold0/model/080.ckpt \
  --data-root /data/home/zpc/jc/data/LUNA16 \
  --preprocessed-dir "$PREP" \
  --output-dir results/luna16_fold0/eval_080_smoke \
  --max-scans 1 \
  --skip-froc
```

## 5. Full test-fold FROC

```bash
CUDA_VISIBLE_DEVICES=0 python eval_luna16.py \
  --checkpoint results/luna16_fold0/model/080.ckpt \
  --data-root /data/home/zpc/jc/data/LUNA16 \
  --preprocessed-dir "$PREP" \
  --output-dir results/luna16_fold0/eval_080
```

The evaluator writes world-coordinate submissions and separate RPN, RCNN, and
ensemble FROC outputs below `eval_080/FROC/`.
