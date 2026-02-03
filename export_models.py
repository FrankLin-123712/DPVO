"""
Utility to export DPVO feature encoders and update block to TorchScript/ONNX.

Notes / caveats:
- The DPVO graph relies on custom CUDA ops (fastba, altcorr, torch_scatter). TorchScript
  can capture these extensions as calls, but ONNX export will fall back to custom ops;
  you will need runtime kernels for them on the target.
- The patchify step in Python uses `altcorr.patchify` to crop feature/patch tensors.
  That op is NOT exported here. C++ must replicate the same sampling logic.
- The exported update block expects pre-computed correlation features and context
  tensors laid out identically to the Python pipeline (see dpvo_runner TODOs).
"""

import argparse
from pathlib import Path

import torch

# Ensure repository root (where the `dpvo` package lives) is on the path so the
# script works when run directly from the repo checkout.
import sys
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from dpvo.net import VONet, DIM  # noqa: E402
import dpvo.fastba as fastba  # noqa: E402
from dpvo.blocks import GradientClip  # noqa: E402
from torch.onnx import register_custom_op_symbolic  # noqa: E402
from torch.onnx import symbolic_helper as sym_help  # noqa: E402


def _cpu_neighbors(ii: torch.Tensor, jj: torch.Tensor):
    """CPU-safe replacement for cuda_ba.neighbors used during export on machines without GPUs."""
    ii_cpu = ii.cpu()
    jj_cpu = jj.cpu()

    uniq, inverse = torch.unique(ii_cpu, return_inverse=True)
    ix = torch.full_like(ii_cpu, -1)
    jx = torch.full_like(ii_cpu, -1)

    for idx in range(uniq.numel()):
        mask = inverse == idx
        edge_idxs = torch.nonzero(mask, as_tuple=False).squeeze(1)
        if edge_idxs.numel() == 0:
            continue

        jj_vals = jj_cpu[edge_idxs]
        order = torch.argsort(jj_vals)
        sorted_edges = edge_idxs[order]

        if sorted_edges.numel() > 1:
            ix[sorted_edges[1:]] = sorted_edges[:-1]
            jx[sorted_edges[:-1]] = sorted_edges[1:]

    # Always return CPU tensors to keep tracing on CPU.
    return ix, jx

# Force CPU-safe neighbor lookup during export to avoid CUDA/custom op usage.
fastba.neighbors = _cpu_neighbors


def register_dpvo_custom_ops(opset: int):
    """
    Register custom ONNX symbolic for torch_scatter::scatter_max so export can emit
    a custom-domain node `dpvo::scatter_max` instead of failing.
    """

    def scatter_max_symbolic(g, src, index, dim=None, out=None, dim_size=None, fill_value=None):
        inputs = [src, index]
        attrs = {}
        # dim is usually an int, but can be a graph value; handle both.
        if dim is not None:
            if isinstance(dim, torch._C.Value):
                inputs.append(dim)
            else:
                attrs["dim_i"] = int(dim)

        node = g.op("dpvo::scatter_max", *inputs, outputs=2, **attrs)
        values, argmax = node[0], node[1]

        # Best-effort shape/type propagation to quiet warnings.
        src_type = src.type()
        if isinstance(src_type, torch._C.TensorType):
            values.setType(src_type)
            arg_t = torch._C.TensorType.get()
            if src_type.sizes() is not None:
                arg_t = arg_t.with_sizes(src_type.sizes())
            arg_t = arg_t.with_dtype(torch.int64)
            argmax.setType(arg_t)

        return values, argmax

    try:
        register_custom_op_symbolic("torch_scatter::scatter_max", scatter_max_symbolic, opset)
    except RuntimeError:
        # Already registered for this opset; ignore.
        pass


class FeatureExtractor(torch.nn.Module):
    """Wrapper that returns both fmap (gmap source) and imap encoders."""

    def __init__(self, model: VONet):
        super().__init__()
        self.fnet = model.patchify.fnet
        self.inet = model.patchify.inet

    def forward(self, images):
        """
        images: float tensor [B, N, 3, H, W] in range [-0.5, 0.5] like Python code.
        returns (fmap, imap) matching DPVO patchify encoders (no cropping done here).
        """
        fmap = self.fnet(images)
        imap = self.inet(images)
        return fmap, imap


class UpdateWrapper(torch.nn.Module):
    """Expose only the GRU-style update block."""

    def __init__(self, model: VONet):
        super().__init__()
        self.update = model.update

    def forward(self, net, ctx, corr, flow, ii, jj, kk):
        net_out, (delta, weight, _) = self.update(net, ctx, corr, flow, ii, jj, kk)
        return net_out, delta, weight


def load_weights(weights_path: Path) -> VONet:
    state = torch.load(weights_path, map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]
    # Strip "module." prefix if present
    cleaned = {k.replace("module.", ""): v for k, v in state.items() if "update.lmbda" not in k}
    model = VONet()
    model.load_state_dict(cleaned, strict=False)
    _replace_grad_clip(model.update)
    model.eval()
    return model


def _replace_grad_clip(module: torch.nn.Module):
    """Recursively replace GradientClip layers with identity for export (forward is already identity)."""
    for name, child in list(module.named_children()):
        if isinstance(child, GradientClip):
            setattr(module, name, torch.nn.Identity())
        else:
            _replace_grad_clip(child)


def export_feature(model: VONet, out_dir: Path, h: int, w: int, opset: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    feat = FeatureExtractor(model).eval()

    dummy = torch.randn(1, 1, 3, h, w)
    ts_path = out_dir / "feature_extractor.ts"
    onnx_path = out_dir / "feature_extractor.onnx"

    feat_ts = torch.jit.trace(feat, dummy)
    feat_ts.save(ts_path)

    torch.onnx.export(
        feat,
        dummy,
        onnx_path,
        input_names=["images"],
        output_names=["fmap", "imap"],
        opset_version=opset,
        dynamic_axes={"images": {0: "batch", 1: "frames", 3: "h", 4: "w"}},
    )
    print(f"Saved feature extractor: {ts_path} and {onnx_path}")


def export_update(model: VONet, out_dir: Path, K: int, opset: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    upd = UpdateWrapper(model).eval()
    # Ensure everything is on CPU for export; avoids CUDA availability checks.
    upd = upd.cpu()

    net = torch.zeros(1, K, DIM)
    ctx = torch.zeros(1, K, DIM)
    # corr size matches Python: 2 * 49 * P * P with P=3 => 882
    corr = torch.zeros(1, K, 2 * 49 * model.P * model.P)
    flow = torch.zeros(1, K, 2)
    ii = torch.zeros(K, dtype=torch.long)
    jj = torch.zeros(K, dtype=torch.long)
    kk = torch.zeros(K, dtype=torch.long)

    ts_path = out_dir / "update_block.ts"
    onnx_path = out_dir / "update_block.onnx"

    upd_ts = torch.jit.trace(upd, (net, ctx, corr, flow, ii, jj, kk))
    upd_ts.save(ts_path)

    try:
        torch.onnx.export(
            upd,
            (net, ctx, corr, flow, ii, jj, kk),
            onnx_path,
            input_names=["net", "ctx", "corr", "flow", "ii", "jj", "kk"],
            output_names=["net_out", "delta", "weight"],
            opset_version=opset,
            dynamic_axes={
                "net": {0: "batch", 1: "edges"},
                "ctx": {0: "batch", 1: "edges"},
                "corr": {0: "batch", 1: "edges"},
                "flow": {0: "batch", 1: "edges"},
                "ii": {0: "edges"},
                "jj": {0: "edges"},
                "kk": {0: "edges"},
            },
        )
        print(f"Saved update block: {ts_path} and {onnx_path}")
    except Exception as e:
        print(f"ONNX export of update block failed (TorchScript saved): {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True, help="Path to dpvo.pth checkpoint")
    parser.add_argument("--out", type=Path, default=Path("exported_models"))
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--edges", type=int, default=64, help="Dummy edge count for update export")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    register_dpvo_custom_ops(args.opset)
    model = load_weights(args.weights)
    export_feature(model, args.out, args.height, args.width, args.opset)
    export_update(model, args.out, args.edges, args.opset)


if __name__ == "__main__":
    main()
