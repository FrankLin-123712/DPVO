#!/bin/bash

# Prefer the dpvo conda env if present; otherwise fall back to current python.
DEFAULT_PY="$HOME/miniconda3/envs/dpvo/bin/python"
PYTHON_BIN=${PYTHON_BIN:-$DEFAULT_PY}
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN=$(command -v python)
fi

"$PYTHON_BIN" export_models.py \
  --weights ./dpvo.pth \
  --out ./exported_models_IR7 \
  --height 480 --width 640 \
  --edges 256 \
  --opset 13
