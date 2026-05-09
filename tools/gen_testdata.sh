#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

# Prefer the dpvo conda env if present; otherwise fall back to the current python.
DEFAULT_PY="$HOME/miniconda3/envs/dpvo/bin/python"
PYTHON_BIN=${PYTHON_BIN:-$DEFAULT_PY}
if [ ! -x "$PYTHON_BIN" ]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN=$(command -v python)
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=$(command -v python3)
  else
    echo "[ERROR] Could not find a usable Python interpreter." >&2
    exit 1
  fi
fi

WEIGHTS=${WEIGHTS:-"$REPO_ROOT/dpvo.pth"}
IMAGES=${IMAGES:-"$REPO_ROOT/subset_0493"}
CALIB=${CALIB:-"$REPO_ROOT/calib/iphone.txt"}
TESTDATA_ROOT=${TESTDATA_ROOT:-"$REPO_ROOT/testdata"}
MODE=${1:-0}

usage() {
  echo "Usage: $0 [MODE]"
  echo "  MODE=0  Generate all testdata"
  echo "  MODE=1  Generate dpvo_runner_parity_small"
  echo "  MODE=2  Generate dpvo_python_medium_fast"
  echo "  MODE=3  Generate dpvo_python_medium"
}

gen_dpvo_runner_parity_small() {
  echo "[INFO] Generating dpvo_runner_parity_small"
  "$PYTHON_BIN" "$SCRIPT_DIR/generate_dpvo_runner_parity_testdata.py" \
    --weights "$WEIGHTS" \
    --images "$IMAGES" \
    --calib "$CALIB" \
    --output-root "$TESTDATA_ROOT/dpvo_runner_parity_small" \
    --width 256 \
    --height 144 \
    --frame-start 1 \
    --frame-count 4 \
    --patches-per-frame 8
}

gen_dpvo_python_medium_fast() {
  echo "[INFO] Generating dpvo_python_medium_fast"
  "$PYTHON_BIN" "$SCRIPT_DIR/generate_dpvo_python_testdata.py" \
    --output-root "$TESTDATA_ROOT/dpvo_python_medium_fast" \
    --frame-count 32 \
    --max-long-edge 256 \
    --patches-per-frame 16 \
    --buffer-size 40 \
    --removal-window 8 \
    --optimization-window 12 \
    --patch-lifetime 8 \
    --seed 7 \
    --dump-state
}

gen_dpvo_python_medium() {
  echo "[INFO] Generating dpvo_python_medium"
  "$PYTHON_BIN" "$SCRIPT_DIR/generate_dpvo_python_testdata.py" \
    --output-root "$TESTDATA_ROOT/dpvo_python_medium" \
    --frame-count 32 \
    --max-long-edge 512 \
    --patches-per-frame 64 \
    --buffer-size 96 \
    --removal-window 16 \
    --optimization-window 12 \
    --patch-lifetime 14 \
    --seed 7 \
    --dump-state
}

echo "[INFO] MODE: $MODE"
echo "[INFO] PYTHON_BIN: $PYTHON_BIN"

case "$MODE" in
  0)
    gen_dpvo_python_medium_fast
    gen_dpvo_python_medium
    gen_dpvo_runner_parity_small
    ;;
  1)
    gen_dpvo_runner_parity_small
    ;;
  2)
    gen_dpvo_python_medium_fast
    ;;
  3)
    gen_dpvo_python_medium
    ;;
  *)
    echo "[ERROR] Unsupported MODE: $MODE" >&2
    usage
    exit 1
    ;;
esac
