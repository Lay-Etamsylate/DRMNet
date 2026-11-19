#!/usr/bin/env bash
# Run from the activated drmnet-l20 environment. GPU work runs on the server.
set -euo pipefail

MODE="${1:-resume}"
case "$MODE" in
  resume|fresh) ;;
  *) echo "Usage: bash scripts/run_luna16_80epochs.sh [resume|fresh]" >&2; exit 2 ;;
esac

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
export MPLBACKEND=Agg
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
LUNA_ROOT="${LUNA_ROOT:-/data/home/zpc/jc/data/LUNA16}"
DATA_DIR="${DATA_DIR:-$LUNA_ROOT/drmnet_preprocessed}"
RUN_DIR="${RUN_DIR:-$REPO_DIR/results/luna16_fusion_full80_$MODE}"
SOURCE_CKPT="${SOURCE_CKPT:-$REPO_DIR/results/luna16_fusion_joint_lr5e5/model/040.ckpt}"

# A separate output directory preserves every previous experiment.
if [[ -e "$RUN_DIR" ]]; then
  echo "Output already exists: $RUN_DIR. Set RUN_DIR to a new directory." >&2
  exit 1
fi
for split in train val test; do
  for file in "$DATA_DIR/filelists/$split.npy" "$DATA_DIR/filelists/${split}_annos.csv"; do
    [[ -f "$file" ]] || { echo "Missing: $file" >&2; exit 1; }
  done
done
if [[ "$MODE" == resume && ! -f "$SOURCE_CKPT" ]]; then
  echo "Missing checkpoint: $SOURCE_CKPT" >&2
  exit 1
fi

# Optional CLI flag exists only in newer evaluator versions.
EVAL_HELP="$(python scripts/evaluate_luna16.py --help)"
MATCHING_ARGS=()
if [[ "$EVAL_HELP" == *--matching-rule* ]]; then
  MATCHING_ARGS=(--matching-rule legacy)
fi

train_stage() {
  python -u train.py \
    --variant fusion --head-activation leaky_relu \
    --auxiliary-losses off --scale-augmentation off \
    --train-set "$DATA_DIR/filelists/train.npy" \
    --val-set "$DATA_DIR/filelists/val.npy" \
    --train-data-dir "$DATA_DIR" --val-data-dir "$DATA_DIR" \
    --train-anno "$DATA_DIR/filelists/train_annos.csv" \
    --val-anno "$DATA_DIR/filelists/val_annos.csv" \
    --batch-size 1 --num-workers 4 \
    --optimizer SGD --momentum 0.9 --weight-decay 0.0001 \
    --max-grad-norm 5 \
    --rpn-train-pre-topk 2000 --rpn-train-post-topk 300 \
    --epochs 80 --epoch-rcnn 21 --epoch-save 1 \
    --report-head-health --out-dir "$RUN_DIR" "$@"
}

if [[ "$MODE" == fresh ]]; then
  # Match the staged learning rates used during the successful detector run.
  train_stage --init-lr 0.001 --stop-after-epoch 20
  train_stage --ckpt "$RUN_DIR/model/020.ckpt" --init-lr 0.0001
else
  train_stage --ckpt "$SOURCE_CKPT" --init-lr 0.00005
fi

# Evaluate the final epoch, separately for validation and held-out test scans.
for split in val test; do
  PRED_DIR="$RUN_DIR/${split}_predictions"
  python -u test.py \
    --mode eval --weight "$RUN_DIR/model/080.ckpt" \
    --test-set-name "$DATA_DIR/filelists/$split.npy" \
    --test-anno "$DATA_DIR/filelists/${split}_annos.csv" \
    --preprocessed-data-dir "$DATA_DIR" \
    --out-dir "$PRED_DIR" --inference-only

  python -u scripts/evaluate_luna16.py \
    --pred-dir "$PRED_DIR/res/80/FROC" \
    --meta-dir "$DATA_DIR" \
    --test-set "$DATA_DIR/filelists/$split.npy" \
    --preprocessed-annotations "$DATA_DIR/filelists/${split}_annos.csv" \
    --annotations "$LUNA_ROOT/annotations.csv" \
    "${MATCHING_ARGS[@]}" \
    --out-dir "$PRED_DIR/res/80/FROC_world"
done

echo "Finished: $RUN_DIR"
echo "Validation: $RUN_DIR/val_predictions/res/80/FROC_world/metrics.txt"
echo "Test:       $RUN_DIR/test_predictions/res/80/FROC_world/metrics.txt"
