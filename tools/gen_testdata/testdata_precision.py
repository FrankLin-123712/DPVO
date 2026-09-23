"""CLI and output policy shared by both generators; no ONNX/torch imports."""
from pathlib import Path
import json


def add_precision_arguments(parser):
    parser.add_argument("--nn-precision", choices=("fp32", "fp16"), default="fp32",
                        help="fp16 uses mixed ONNX graph reference, not whole-network half")
    parser.add_argument("--onnx-model-dir", type=Path,
                        help="Converted feature/update opset11 models; required for fp16")


def resolve_precision(parser, args, default_root):
    if (args.nn_precision == "fp16") != (args.onnx_model_dir is not None):
        parser.error("--nn-precision=fp16 and --onnx-model-dir must be specified together")
    if args.output_root is None:
        args.output_root = Path(str(default_root) + ("_fp16" if args.nn_precision == "fp16" else ""))
    return args


def prepare_reference(args):
    # Do this before either generator deletes/replaces existing output files.
    root = args.output_root
    metadata = root / "metadata.json"
    if metadata.exists():
        previous = json.loads(metadata.read_text()).get("nn_reference")
        previous_precision = "fp16" if previous else "fp32"
        if previous_precision != args.nn_precision:
            raise ValueError("Output directory belongs to a different NN precision; use a separate directory")
    elif args.nn_precision == "fp16" and root.exists() and any(root.iterdir()):
        raise ValueError("Refusing to overwrite nonempty output without precision metadata")
    if args.nn_precision == "fp32":
        return None
    from fp16_onnx_reference import MixedReference
    return MixedReference(args.onnx_model_dir)


def require_cuda_dpvo():
    """Fail before replacing output when the full generators cannot run."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Full/component generation needs PyTorch and CUDA DPVO; use fp16_onnx_reference.py for CPU-only NN cases") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("Full/component generation requires CUDA for DPVO/fastba")
    from dpvo import fastba, altcorr  # noqa: F401


def check_finite_case(tensors):
    import numpy as np
    for name, array in tensors.items():
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name}: refusing to write nonfinite reference data")
