"""CLI and output policy shared by both generators; no ONNX/torch imports."""
from pathlib import Path
import json


def add_precision_arguments(parser):
    parser.add_argument("--nn-precision", choices=("fp32", "fp16"), default="fp32",
                        help="fp16 uses mixed ONNX graph reference, not whole-network half")
    parser.add_argument("--onnx-model-dir", type=Path,
                        help="Converted feature/update opset11 models; required for fp16")
    parser.add_argument("--nn-accumulation", choices=("fp32", "gemmini-dim32"), default="fp32",
                        help="FP16 NN accumulation: FP32 reference (default) or DIM32 half FMA / FP32 sums")


def resolve_precision(parser, args, default_root):
    if (args.nn_precision == "fp16") != (args.onnx_model_dir is not None):
        parser.error("--nn-precision=fp16 and --onnx-model-dir must be specified together")
    if args.nn_accumulation != "fp32" and args.nn_precision != "fp16":
        parser.error("--nn-accumulation=gemmini-dim32 requires --nn-precision=fp16")
    if args.output_root is None:
        suffix = "_fp16" if args.nn_precision == "fp16" else ""
        if args.nn_accumulation == "gemmini-dim32":
            suffix += "_dim32"
        args.output_root = default_root.parent / args.nn_precision / (default_root.name + suffix)
    return args


def prepare_reference(args):
    # Do this before either generator deletes/replaces existing output files.
    accumulation = getattr(args, "nn_accumulation", "fp32")
    root = args.output_root
    metadata = root / "metadata.json"
    if metadata.exists():
        previous = json.loads(metadata.read_text()).get("nn_reference")
        previous_precision = "fp16" if previous else "fp32"
        if previous_precision != args.nn_precision:
            raise ValueError("Output directory belongs to a different NN precision; use a separate directory")
        if previous:
            from fp16_onnx_reference import ACCUMULATIONS
            if previous.get("accumulation") != ACCUMULATIONS[accumulation]:
                raise ValueError("Output directory belongs to a different NN accumulation; use a separate directory")
    elif args.nn_precision == "fp16" and root.exists() and any(root.iterdir()):
        raise ValueError(
            f"Refusing to overwrite nonempty output without precision metadata: {root.resolve()}. "
            "metadata.json is missing (a previous generation may have been interrupted). "
            "Use a new output directory or move the existing directory aside before retrying."
        )
    if args.nn_precision == "fp32":
        return None
    from fp16_onnx_reference import MixedReference
    return MixedReference(args.onnx_model_dir, accumulation=accumulation)


def require_cuda_dpvo():
    """Fail before replacing output when the full generators cannot run."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Full/component generation needs PyTorch and CUDA DPVO") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("Full/component generation requires CUDA for DPVO/fastba")
    from dpvo import fastba, altcorr  # noqa: F401


def check_finite_case(tensors):
    import numpy as np
    for name, array in tensors.items():
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name}: refusing to write nonfinite reference data")
