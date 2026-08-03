#!/usr/bin/env python3
"""Inspect a DPVO ONNX file and run the ONNX structural checker."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = REPO_ROOT / "exported_models" / "update_block.onnx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print basic metadata for an ONNX model and run onnx.checker."
    )
    parser.add_argument(
        "model",
        nargs="?",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"Path to the ONNX model. Default: {DEFAULT_MODEL}",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model = onnx.load(args.model)

    print(f"Model: {args.model}")
    print("IR version:", model.ir_version)
    print("Producer:", model.producer_name, model.producer_version)
    print("Opset imports:", {item.domain or 'ai.onnx': item.version for item in model.opset_import})
    print("Inputs:", [value.name for value in model.graph.input])
    print("Outputs:", [value.name for value in model.graph.output])

    onnx.checker.check_model(model)
    print("onnx.checker: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
