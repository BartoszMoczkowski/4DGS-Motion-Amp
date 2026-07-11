#!/usr/bin/env bash
# 4DGS training + render for the synthetic pump scene.
# Runs in the CUDA/PyTorch environment (the repo's existing .devcontainer or your GPU env),
# NOT the Isaac Sim container. Assumes you've already run:
#   omni_capture.py  ->  omni_to_4dgs.py --name pump01   (=> data/multipleview/pump01)
set -euo pipefail

NAME="${1:-pump01}"
PORT="${2:-6017}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

test -d "data/multipleview/$NAME" || { echo "ERROR: data/multipleview/$NAME not found — run omni_to_4dgs.py first"; exit 1; }
test -f "arguments/multipleview/$NAME.py" || { echo "ERROR: arguments/multipleview/$NAME.py not found"; exit 1; }

echo "[train] 4DGS on data/multipleview/$NAME"
uv run train.py -s "data/multipleview/$NAME" --port "$PORT" \
    --expname "multipleview/$NAME" --configs "arguments/multipleview/$NAME.py"

echo "[render] test + video views"
uv run render.py --model_path "output/multipleview/$NAME/" --skip_train \
    --configs "arguments/multipleview/$NAME.py"

echo "[done] trained model in output/multipleview/$NAME/"
echo "  next: segmentation (MBS / trajectory clustering) then per-segment motion amp (render_amp.py)."
echo "  GT motion-segment labels: data/multipleview/$NAME/gt_segmentation.npz  (+ *_motion_groups.json)"
