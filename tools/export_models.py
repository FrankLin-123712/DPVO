#!/usr/bin/env python3
"""
Export DPVO ONNX models.

This script supports the two ONNX artifacts needed by the DPVO pipeline:
- feature_extractor.onnx
- update_block.onnx

The update block export uses explicit neighbor indices (ix/jx) as inputs.
That keeps the exported graph free of the fragile fastba.neighbors lowering
used by the older exporter.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from torch.onnx import register_custom_op_symbolic
from torch.onnx import symbolic_helper as sym_help

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
sys.path.insert(0, str(REPO_ROOT))

from dpvo.blocks import GradientClip  # noqa: E402
from dpvo.net import DIM, VONet  # noqa: E402

SCATTER_SUM_PATCHES = {
    "/agg_kk/Add": (
        "/agg_kk/Exp_output_0",
        "/agg_kk/Expand_1_output_0",
    ),
    "/agg_kk/Add_1": (
        "/agg_kk/Mul_2_output_0",
        "/agg_kk/Expand_2_output_0",
    ),
    "/agg_ij/Add": (
        "/agg_ij/Exp_output_0",
        "/agg_ij/Expand_1_output_0",
    ),
    "/agg_ij/Add_1": (
        "/agg_ij/Mul_2_output_0",
        "/agg_ij/Expand_2_output_0",
    ),
}


class FeatureExtractor(torch.nn.Module):
    """Return the feature and context encoders used by DPVO patchify."""

    def __init__(self, model: VONet):
        super().__init__()
        self.fnet = model.patchify.fnet
        self.inet = model.patchify.inet

    def forward(self, images):
        fmap = self.fnet(images)
        imap = self.inet(images)
        return fmap, imap


class UpdateWrapperExplicitNeighbors(torch.nn.Module):
    """Expose the update block with host-provided neighbor indices."""

    def __init__(self, model: VONet):
        super().__init__()
        self.update = model.update

    def forward(self, net, ctx, corr, ii, jj, kk, ix, jx):
        net = net + ctx + self.update.corr(corr)
        net = self.update.norm(net)

        mask_ix = (ix >= 0).float().reshape(1, -1, 1)
        mask_jx = (jx >= 0).float().reshape(1, -1, 1)

        net = net + self.update.c1(mask_ix * net[:, ix])
        net = net + self.update.c2(mask_jx * net[:, jx])
        net = net + self.update.agg_kk(net, kk)
        net = net + self.update.agg_ij(net, ii * 12345 + jj)
        net = self.update.gru(net)
        return net, self.update.d(net), self.update.w(net)


def compute_neighbor_indices(group_ids: torch.Tensor, order_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute previous/next neighbor links within each group on CPU."""
    group_cpu = group_ids.cpu()
    order_cpu = order_ids.cpu()

    unique_groups, inverse = torch.unique(group_cpu, return_inverse=True)
    ix = torch.full_like(group_cpu, -1)
    jx = torch.full_like(group_cpu, -1)

    for idx in range(unique_groups.numel()):
        mask = inverse == idx
        edge_idxs = torch.nonzero(mask, as_tuple=False).squeeze(1)
        if edge_idxs.numel() == 0:
            continue

        order = torch.argsort(order_cpu[edge_idxs])
        sorted_edges = edge_idxs[order]

        if sorted_edges.numel() > 1:
            ix[sorted_edges[1:]] = sorted_edges[:-1]
            jx[sorted_edges[:-1]] = sorted_edges[1:]

    return ix, jx


def register_dpvo_custom_ops(opset: int) -> None:
    """Register ONNX symbolics needed by DPVO custom operators."""

    def scatter_max_symbolic(g, src, index, dim=None, out=None, dim_size=None, fill_value=None):
        inputs = [src, index]
        attrs = {}
        if dim is not None:
            dim_const = sym_help._maybe_get_const(dim, "i")
            attrs["dim_i"] = int(dim_const)

        node = g.op("dpvo::scatter_max", *inputs, outputs=2, **attrs)
        values, argmax = node[0], node[1]

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
        pass


def load_weights(weights_path: Path) -> VONet:
    state = torch.load(weights_path, map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]

    cleaned = {
        key.replace("module.", ""): value
        for key, value in state.items()
        if "update.lmbda" not in key
    }

    model = VONet()
    model.load_state_dict(cleaned, strict=False)
    _replace_grad_clip(model.update)
    model.eval()
    return model


def _replace_grad_clip(module: torch.nn.Module) -> None:
    """Replace GradientClip layers with identity for stable export."""
    for name, child in list(module.named_children()):
        if isinstance(child, GradientClip):
            setattr(module, name, torch.nn.Identity())
        else:
            _replace_grad_clip(child)


def patch_legacy_scatter_sum(path: Path) -> int:
    """Rewrite legacy scatter_sum subgraphs into dpvo::scatter_sum nodes."""
    import onnx

    model = onnx.load(path)
    patch_count = 0

    for node in model.graph.node:
        replacement_inputs = SCATTER_SUM_PATCHES.get(node.name)
        if replacement_inputs is None:
            continue

        del node.input[:]
        node.input.extend(replacement_inputs)
        node.op_type = "scatter_sum"
        node.domain = "dpvo"
        del node.attribute[:]
        node.attribute.extend([onnx.helper.make_attribute("dim", 1)])
        patch_count += 1

    onnx.save(model, path)
    return patch_count


def export_feature(model: VONet, out_dir: Path, height: int, width: int, opset: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    feature = FeatureExtractor(model).cpu().eval()
    onnx_path = out_dir / "feature_extractor.onnx"
    dummy_images = torch.randn(1, 1, 3, height, width)

    torch.onnx.export(
        feature,
        dummy_images,
        onnx_path,
        input_names=["images"],
        output_names=["fmap", "imap"],
        opset_version=opset,
        dynamic_axes={"images": {0: "batch", 1: "frames", 3: "height", 4: "width"}},
    )
    return onnx_path


def export_update(model: VONet, out_dir: Path, edge_count: int, opset: int) -> tuple[Path, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    update = UpdateWrapperExplicitNeighbors(model).cpu().eval()
    onnx_path = out_dir / "update_block.onnx"

    net = torch.zeros(1, edge_count, DIM)
    ctx = torch.zeros(1, edge_count, DIM)
    corr = torch.zeros(1, edge_count, 2 * 49 * model.P * model.P)
    ii = torch.zeros(edge_count, dtype=torch.long)
    jj = torch.arange(1, edge_count + 1, dtype=torch.long)
    kk = torch.arange(edge_count, dtype=torch.long)
    ix = torch.full((edge_count,), -1, dtype=torch.long)
    jx = torch.full((edge_count,), -1, dtype=torch.long)

    torch.onnx.export(
        update,
        (net, ctx, corr, ii, jj, kk, ix, jx),
        onnx_path,
        input_names=["net", "ctx", "corr", "ii", "jj", "kk", "ix", "jx"],
        output_names=["net_out", "delta", "weight"],
        opset_version=opset,
        dynamic_axes={
            "net": {0: "batch", 1: "edges"},
            "ctx": {0: "batch", 1: "edges"},
            "corr": {0: "batch", 1: "edges"},
            "ii": {0: "edges"},
            "jj": {0: "edges"},
            "kk": {0: "edges"},
            "ix": {0: "edges"},
            "jx": {0: "edges"},
        },
    )
    patch_count = patch_legacy_scatter_sum(onnx_path)
    return onnx_path, patch_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export DPVO ONNX models. The update block export uses explicit "
            "neighbor indices ix/jx and is the supported ONNX path."
        )
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=REPO_ROOT / "dpvo.pth",
        help="Path to the DPVO checkpoint.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "exported_models",
        help="Output directory for exported ONNX files.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Dummy image height used for feature extractor export.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Dummy image width used for feature extractor export.",
    )
    parser.add_argument(
        "--edges",
        type=int,
        default=256,
        help="Dummy edge count used to trace the update block export.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=13,
        help="ONNX opset version.",
    )
    parser.add_argument(
        "--skip-feature",
        action="store_true",
        help="Skip feature extractor export.",
    )
    parser.add_argument(
        "--skip-update",
        action="store_true",
        help="Skip update block export.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    register_dpvo_custom_ops(args.opset)
    model = load_weights(args.weights).cpu().eval()

    if not args.skip_feature:
        feature_path = export_feature(model, args.out, args.height, args.width, args.opset)
        print(f"Saved feature extractor ONNX to {feature_path}")

    if not args.skip_update:
        update_path, patch_count = export_update(model, args.out, args.edges, args.opset)
        print(f"Saved update block ONNX to {update_path}")
        print(f"Patched {patch_count} scatter_sum node(s) in-place")
        print("Update block inputs: net, ctx, corr, ii, jj, kk, ix, jx")

    if args.skip_feature and args.skip_update:
        print("Nothing to export because both --skip-feature and --skip-update were set.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
