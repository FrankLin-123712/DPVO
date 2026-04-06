#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(TOOLS_DIR))

from export_models import (  # noqa: E402
    DIM,
    UpdateWrapperExplicitNeighbors,
    compute_neighbor_indices,
    load_weights,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a tiny deterministic update_block parity case using only "
            "PyTorch inference on DPVO's update module."
        )
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=REPO_ROOT / "dpvo.pth",
        help="Path to the DPVO checkpoint used to build the update block.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/dpvo_update_block_case_small"),
        help="Output directory containing manifest.txt and raw tensor binaries.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed used to generate deterministic ctx/corr tensors.",
    )
    parser.add_argument(
        "--groups",
        type=int,
        default=4,
        help="Number of kk groups to generate.",
    )
    parser.add_argument(
        "--edges-per-group",
        type=int,
        default=3,
        help="Number of edges in each kk group.",
    )
    parser.add_argument(
        "--ctx-scale",
        type=float,
        default=0.5,
        help="Scale applied to the random ctx tensor.",
    )
    parser.add_argument(
        "--corr-scale",
        type=float,
        default=0.8,
        help="Scale applied to the random corr tensor.",
    )
    return parser.parse_args()


def build_indices(groups: int, edges_per_group: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if groups <= 0:
        raise ValueError("--groups must be positive")
    if edges_per_group < 2:
        raise ValueError("--edges-per-group must be at least 2 to exercise neighbor links")

    kk = torch.arange(groups, dtype=torch.long).repeat_interleave(edges_per_group)
    jj = torch.arange(1, edges_per_group + 1, dtype=torch.long).repeat(groups)
    ii = torch.zeros_like(kk)
    return ii, jj, kk


def write_manifest(output_dir: Path, tensors: dict[str, np.ndarray]) -> None:
    lines = [
        "# update_block_case_v1",
        "# filename is inferred as <tensor_name>.bin",
    ]
    for name, array in tensors.items():
        shape = " ".join(str(dim) for dim in array.shape)
        lines.append(f"{name} {array.dtype.name} {shape}".rstrip())
    (output_dir / "manifest.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tensor(output_dir: Path, name: str, array: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(array)
    contiguous.tofile(output_dir / f"{name}.bin")


def main() -> int:
    args = parse_args()

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)

    model = load_weights(args.weights).cpu().eval()
    update = UpdateWrapperExplicitNeighbors(model).cpu().eval()

    ii, jj, kk = build_indices(args.groups, args.edges_per_group)
    ix, jx = compute_neighbor_indices(kk, jj)
    edge_count = kk.numel()

    net = torch.zeros(1, edge_count, DIM, dtype=torch.float32)
    ctx = torch.randn(1, edge_count, DIM, generator=generator, dtype=torch.float32)
    corr = torch.randn(
        1,
        edge_count,
        2 * 49 * model.P * model.P,
        generator=generator,
        dtype=torch.float32,
    )
    ctx.mul_(args.ctx_scale)
    corr.mul_(args.corr_scale)

    with torch.inference_mode():
        golden_net, golden_delta, golden_weight = update(net, ctx, corr, ii, jj, kk, ix, jx)

    tensors = {
        "net": net.numpy().astype(np.float32, copy=False),
        "ctx": ctx.numpy().astype(np.float32, copy=False),
        "corr": corr.numpy().astype(np.float32, copy=False),
        "ii": ii.numpy().astype(np.int64, copy=False),
        "jj": jj.numpy().astype(np.int64, copy=False),
        "kk": kk.numpy().astype(np.int64, copy=False),
        "ix": ix.numpy().astype(np.int64, copy=False),
        "jx": jx.numpy().astype(np.int64, copy=False),
        "golden_net": golden_net.numpy().astype(np.float32, copy=False),
        "golden_delta": golden_delta.numpy().astype(np.float32, copy=False),
        "golden_weight": golden_weight.numpy().astype(np.float32, copy=False),
    }

    args.output.mkdir(parents=True, exist_ok=True)
    write_manifest(args.output, tensors)
    for name, array in tensors.items():
        write_tensor(args.output, name, array)

    print(f"Wrote update_block case to {args.output}")
    print(f"weights={args.weights}")
    print(f"seed={args.seed} groups={args.groups} edges_per_group={args.edges_per_group}")
    print(
        "max_abs(golden_net)=%.6f max_abs(golden_delta)=%.6f max_abs(golden_weight)=%.6f"
        % (
            float(np.max(np.abs(tensors["golden_net"]))),
            float(np.max(np.abs(tensors["golden_delta"]))),
            float(np.max(np.abs(tensors["golden_weight"]))),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
