#!/usr/bin/env bash
# Rigid motion-segmentation pipeline for a trained 4DGS multipleview scene.
# Runs in the same CUDA/PyTorch environment as train_pump.sh (extract_trajectories.py needs
# the GPU; segment_rigid.py and evaluate_segmentation.py are pure numpy/scipy).
#
#   ./motion_seg/run.sh pump01
#   ./motion_seg/run.sh pump01 --threshold-mult 2 --opacity-thresh 0.2   # re-tune, skip re-extract
#
# Extra args after the scene name are forwarded to segment_rigid.py (--k, --min-size,
# --threshold-mult, --opacity-thresh) so you can re-tune without re-running the GPU extraction
# step. Set SKIP_EXTRACT=1 to reuse an existing trajectories.npz.
set -euo pipefail

NAME="${1:-pump01}"
shift || true
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

MODEL_DIR="output/multipleview/$NAME"
DATA_DIR="data/multipleview/$NAME"
CONFIG="arguments/multipleview/$NAME.py"

test -d "$MODEL_DIR" || { echo "ERROR: $MODEL_DIR not found — train the scene first (train_pump.sh $NAME)"; exit 1; }
test -f "$CONFIG" || { echo "ERROR: $CONFIG not found"; exit 1; }

if [ "${SKIP_EXTRACT:-0}" = "1" ] && [ -f "$MODEL_DIR/trajectories.npz" ]; then
    echo "[1/3] SKIP_EXTRACT=1 — reusing existing $MODEL_DIR/trajectories.npz"
else
    echo "[1/3] extracting per-Gaussian trajectories (GPU) -> $MODEL_DIR/trajectories.npz"
    uv run python -m motion_seg.extract_trajectories --model_path "$MODEL_DIR" --configs "$CONFIG"
fi

echo "[2/3] rigid motion segmentation -> $MODEL_DIR/segmentation.npz"
uv run python -m motion_seg.segment_rigid \
    --trajectories "$MODEL_DIR/trajectories.npz" \
    --out "$MODEL_DIR/segmentation.npz" \
    "$@"

if [ -f "$DATA_DIR/gt_segmentation.npz" ]; then
    echo "[3/3] evaluating against GT ($DATA_DIR/gt_segmentation.npz)"
    uv run python -m motion_seg.evaluate_segmentation \
        --pred "$MODEL_DIR/segmentation.npz" \
        --gt "$DATA_DIR/gt_segmentation.npz" \
        --recolored-ply "$MODEL_DIR/segmentation_preview.ply"
else
    echo "[3/3] no $DATA_DIR/gt_segmentation.npz found — skipping evaluation"
fi

echo "[done] segmentation: $MODEL_DIR/segmentation.npz"
echo "  segment preview:    $MODEL_DIR/segmentation_preview.png"
echo "  GT-vs-pred preview:  $MODEL_DIR/segmentation_vs_gt.png"
