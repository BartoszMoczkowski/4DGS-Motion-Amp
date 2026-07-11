#!/usr/bin/env bash
# Run the pump capture inside the Isaac Sim dev container.
# Usage:  ./run_capture.sh [extra omni_capture.py flags]
#   smoke test first:  ./run_capture.sh --n-cameras 2 --frames 2
set -euo pipefail

REPO=/workspace
CFG="$REPO/omniverse_pipeline/capture_config_pump.container.yaml"

echo "[run_capture] Isaac Sim python: /isaac-sim/python.sh"
echo "[run_capture] config: $CFG"
test -f /omniverse/assets/pump_radnom/CONJUNTO_BOMBAS_animated.usd \
  || { echo "ERROR: animated pump not found under /omniverse — is OMNIVERSE_DIR mounted?"; exit 1; }

cd "$REPO/omniverse_pipeline"
/isaac-sim/python.sh omni_capture.py --config "$CFG" "$@"

echo "[run_capture] done. Convert to 4DGS with:"
echo "  python3 $REPO/omniverse_pipeline/omni_to_4dgs.py --capture /omniverse/renders/capture_pump --out $REPO --name pump01"
