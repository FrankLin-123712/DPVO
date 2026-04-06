#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

# Prefer the dpvo conda env if present; otherwise fall back to the current python.
DEFAULT_PY="$HOME/miniconda3/envs/dpvo/bin/python"
PYTHON_BIN=${PYTHON_BIN:-$DEFAULT_PY}
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN=$(command -v python)
fi

OUT_DIR=${OUT_DIR:-"$REPO_ROOT/exported_models"}
WEIGHTS=${WEIGHTS:-"$REPO_ROOT/dpvo.pth"}
HEIGHT=${HEIGHT:-480}
WIDTH=${WIDTH:-640}
EDGES=${EDGES:-256}
OPSET=${OPSET:-13}

"$PYTHON_BIN" "$SCRIPT_DIR/export_models.py" \
  --weights "$WEIGHTS" \
  --out "$OUT_DIR" \
  --height "$HEIGHT" \
  --width "$WIDTH" \
  --edges "$EDGES" \
  --opset "$OPSET"
