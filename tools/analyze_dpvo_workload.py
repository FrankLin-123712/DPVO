#!/usr/bin/env python3
"""Static DPVO workload estimator.

The estimator is intentionally dependency-free. It models the main online DPVO
modules from the repository implementation and reports a hardware-oriented
summary table: shapes, MACs, memory traffic, operational intensity, access
pattern, and likely bottleneck.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config" / "default.yaml"

DIM = 384
FNET_DIM = 128
GMAP_DIM = 128
ENCODER_BASE_DIM = 32


@dataclass
class WorkloadRow:
    module: str
    operator: str
    shape: str
    macs: int
    bytes: int
    oi: float
    access_pattern: str
    bottleneck: str

    def display(self, roofline_threshold: float | None = None) -> dict[str, str]:
        row = {
            "Module": self.module,
            "Operator": self.operator,
            "Shape": self.shape,
            "MACs": human_count(self.macs),
            "Bytes": human_bytes(self.bytes),
            "OI": f"{self.oi:.2f}",
            "Access pattern": self.access_pattern,
            "Bottleneck": self.bottleneck,
        }
        if roofline_threshold is not None:
            row["Roofline bound"] = classify_roofline(self, roofline_threshold)
        return row


@dataclass
class ConvLayer:
    name: str
    cin: int
    cout: int
    kernel: int
    stride: int
    padding: int
    hin: int
    win: int

    @property
    def hout(self) -> int:
        return conv_out(self.hin, self.kernel, self.stride, self.padding)

    @property
    def wout(self) -> int:
        return conv_out(self.win, self.kernel, self.stride, self.padding)

    @property
    def macs(self) -> int:
        return self.hout * self.wout * self.cout * self.cin * self.kernel * self.kernel

    def bytes(self, dtype_bytes: int) -> int:
        input_bytes = self.hin * self.win * self.cin * dtype_bytes
        output_bytes = self.hout * self.wout * self.cout * dtype_bytes
        weight_bytes = self.cout * self.cin * self.kernel * self.kernel * dtype_bytes
        bias_bytes = self.cout * dtype_bytes
        return input_bytes + output_bytes + weight_bytes + bias_bytes


def conv_out(size: int, kernel: int, stride: int, padding: int, dilation: int = 1) -> int:
    return (size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if value.startswith(("'", '"')) and value.endswith(("'", '"')):
        return value[1:-1]
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        if any(ch in value for ch in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def load_simple_yaml(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    if not path.exists():
        return data

    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key and value:
            data[key] = parse_scalar(value)
    return data


def human_count(value: int) -> str:
    units = ("", "K", "M", "G", "T", "P")
    number = float(value)
    for unit in units:
        if abs(number) < 1000.0 or unit == units[-1]:
            return f"{number:.2f}{unit}" if unit else str(int(number))
        number /= 1000.0
    return str(value)


def human_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")
    number = float(value)
    for unit in units:
        if abs(number) < 1024.0 or unit == units[-1]:
            return f"{number:.2f} {unit}" if unit != "B" else f"{int(number)} B"
        number /= 1024.0
    return f"{value} B"


def parse_si_number(text: str) -> float:
    """Parse numbers with optional SI suffixes such as 1T or 100GB/s."""
    value = text.strip().replace("_", "")
    for suffix in ("/s", "ps", "B", "b"):
        if value.endswith(suffix):
            value = value[: -len(suffix)]

    suffixes = {
        "K": 1e3,
        "M": 1e6,
        "G": 1e9,
        "T": 1e12,
        "P": 1e15,
    }
    if value and value[-1].upper() in suffixes:
        scale = suffixes[value[-1].upper()]
        value = value[:-1]
    else:
        scale = 1.0

    try:
        parsed = float(value) * scale
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid SI number: {text}") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"value must be positive: {text}")
    return parsed


def classify_roofline(row: WorkloadRow, threshold: float) -> str:
    if math.isclose(row.oi, threshold, rel_tol=0.05):
        return "balanced"
    if row.oi < threshold:
        return "memory-bound"
    return "compute-bound"


def linear_macs(samples: int, in_dim: int, out_dim: int) -> int:
    return samples * in_dim * out_dim


def linear_bytes(samples: int, in_dim: int, out_dim: int, dtype_bytes: int) -> int:
    input_bytes = samples * in_dim * dtype_bytes
    output_bytes = samples * out_dim * dtype_bytes
    weight_bytes = in_dim * out_dim * dtype_bytes
    bias_bytes = out_dim * dtype_bytes
    return input_bytes + output_bytes + weight_bytes + bias_bytes


def build_basic_encoder4_layers(height: int, width: int, output_dim: int, prefix: str) -> list[ConvLayer]:
    layers: list[ConvLayer] = []

    h0, w0 = height, width
    layers.append(ConvLayer(f"{prefix}.conv1", 3, 32, 7, 2, 3, h0, w0))
    h1, w1 = layers[-1].hout, layers[-1].wout

    # layer1: two ResidualBlock(32 -> 32, stride=1)
    for block in range(2):
        layers.append(ConvLayer(f"{prefix}.layer1.{block}.conv1", 32, 32, 3, 1, 1, h1, w1))
        layers.append(ConvLayer(f"{prefix}.layer1.{block}.conv2", 32, 32, 3, 1, 1, h1, w1))

    # layer2 block0: ResidualBlock(32 -> 64, stride=2) with 1x1 downsample
    layers.append(ConvLayer(f"{prefix}.layer2.0.conv1", 32, 64, 3, 2, 1, h1, w1))
    h2, w2 = layers[-1].hout, layers[-1].wout
    layers.append(ConvLayer(f"{prefix}.layer2.0.conv2", 64, 64, 3, 1, 1, h2, w2))
    layers.append(ConvLayer(f"{prefix}.layer2.0.downsample", 32, 64, 1, 2, 0, h1, w1))

    # layer2 block1: ResidualBlock(64 -> 64, stride=1)
    layers.append(ConvLayer(f"{prefix}.layer2.1.conv1", 64, 64, 3, 1, 1, h2, w2))
    layers.append(ConvLayer(f"{prefix}.layer2.1.conv2", 64, 64, 3, 1, 1, h2, w2))

    layers.append(ConvLayer(f"{prefix}.conv2", 64, output_dim, 1, 1, 0, h2, w2))
    return layers


def estimate_feature_encoder(height: int, width: int, dtype_bytes: int) -> WorkloadRow:
    fnet_layers = build_basic_encoder4_layers(height, width, FNET_DIM, "fnet")
    inet_layers = build_basic_encoder4_layers(height, width, DIM, "inet")
    layers = fnet_layers + inet_layers

    macs = sum(layer.macs for layer in layers)
    bytes_ = sum(layer.bytes(dtype_bytes) for layer in layers)
    out_h = fnet_layers[-1].hout
    out_w = fnet_layers[-1].wout

    return make_row(
        module="Feature encoder",
        operator="Conv",
        shape=f"image 1x3x{height}x{width} -> fmap 128x{out_h}x{out_w}, imap 384x{out_h}x{out_w}",
        macs=macs,
        bytes_=bytes_,
        access_pattern="regular",
        bottleneck="compute",
    )


def estimate_update_block(edges: int, patch_size: int, radius: int, levels: int, dtype_bytes: int) -> WorkloadRow:
    corr_dim = levels * (2 * radius + 1) ** 2 * patch_size * patch_size

    macs = 0
    bytes_ = 0

    # self.corr: corr_dim -> DIM -> DIM -> DIM
    for in_dim, out_dim in ((corr_dim, DIM), (DIM, DIM), (DIM, DIM)):
        macs += linear_macs(edges, in_dim, out_dim)
        bytes_ += linear_bytes(edges, in_dim, out_dim, dtype_bytes)

    # c1 and c2: each DIM -> DIM -> DIM
    for _ in range(4):
        macs += linear_macs(edges, DIM, DIM)
        bytes_ += linear_bytes(edges, DIM, DIM, dtype_bytes)

    # Two SoftAgg blocks. Each has f/g/h linear projections.
    for _ in range(6):
        macs += linear_macs(edges, DIM, DIM)
        bytes_ += linear_bytes(edges, DIM, DIM, dtype_bytes)

    # Two GatedResidual blocks. Each has gate plus two residual linears.
    for _ in range(6):
        macs += linear_macs(edges, DIM, DIM)
        bytes_ += linear_bytes(edges, DIM, DIM, dtype_bytes)

    # d and w heads: DIM -> 2 each.
    for _ in range(2):
        macs += linear_macs(edges, DIM, 2)
        bytes_ += linear_bytes(edges, DIM, 2, dtype_bytes)

    # Index tensors, gather/scatter traffic, and state read/write not covered by
    # the GEMM minimum traffic above.
    bytes_ += edges * 5 * 8
    bytes_ += edges * DIM * dtype_bytes * 8

    return make_row(
        module="Update block",
        operator="GRU / MLP",
        shape=f"E={edges}, DIM={DIM}, corr={corr_dim}",
        macs=macs,
        bytes_=bytes_,
        access_pattern="regular",
        bottleneck="compute / memory",
    )


def estimate_correlation(
    edges: int,
    patch_size: int,
    radius: int,
    levels: int,
    channels: int,
    dtype_bytes: int,
) -> WorkloadRow:
    dot_diameter = 2 * radius + 2
    output_diameter = 2 * radius + 1
    macs = levels * edges * patch_size * patch_size * dot_diameter * dot_diameter * channels

    dot_values = levels * edges * patch_size * patch_size * dot_diameter * dot_diameter
    output_values = levels * edges * patch_size * patch_size * output_diameter * output_diameter
    bytes_ = dot_values * (2 * channels * dtype_bytes)
    bytes_ += output_values * dtype_bytes
    bytes_ += edges * patch_size * patch_size * 2 * 4
    bytes_ += edges * 2 * 8

    return make_row(
        module="Correlation lookup",
        operator="dot + bilinear sample",
        shape=f"E={edges}, levels={levels}, P={patch_size}, R={radius}, C={channels}",
        macs=macs,
        bytes_=bytes_,
        access_pattern="semi-irregular",
        bottleneck="memory",
    )


def estimate_ba_jacobian(
    edges: int,
    iterations: int,
    dtype_bytes: int,
    macs_per_edge: int,
) -> WorkloadRow:
    macs = edges * iterations * macs_per_edge

    input_bytes_per_edge = (2 * 7 + 3 + 2 + 2 + 4) * 4 + 3 * 8
    b_atomic_bytes_per_edge = 2 * 4 * 36 * 2 * dtype_bytes
    e_atomic_bytes_per_edge = 2 * 2 * 6 * 2 * dtype_bytes
    v_atomic_bytes_per_edge = 2 * 2 * 6 * 2 * dtype_bytes
    scalar_atomic_bytes_per_edge = 2 * 3 * 2 * dtype_bytes
    bytes_per_edge = (
        input_bytes_per_edge
        + b_atomic_bytes_per_edge
        + e_atomic_bytes_per_edge
        + v_atomic_bytes_per_edge
        + scalar_atomic_bytes_per_edge
    )
    bytes_ = edges * iterations * bytes_per_edge

    return make_row(
        module="BA Jacobian",
        operator="SE(3) projection jacobian",
        shape=f"E={edges}, rows=2, iters={iterations}",
        macs=macs,
        bytes_=bytes_,
        access_pattern="irregular",
        bottleneck="latency",
    )


def estimate_schur(
    poses: int,
    unique_patches: int,
    iterations: int,
    dtype_bytes: int,
) -> WorkloadRow:
    state_dim = 6 * poses

    eqet = state_dim * unique_patches * state_dim
    equ = state_dim * unique_patches
    etdx = unique_patches * state_dim
    chol = int(state_dim**3 / 3)
    solve = state_dim * state_dim
    macs = iterations * (eqet + equ + etdx + chol + solve)

    # Dense local BA footprint used by fastba.BA(eff_impl=False). This is a
    # lower bound on traffic; sparse/block accumulation can reread these blocks.
    values = 0
    values += state_dim * state_dim  # B
    values += state_dim * unique_patches  # E
    values += unique_patches  # C/Q
    values += state_dim + unique_patches  # v/u
    values += state_dim * state_dim  # S
    values += state_dim  # y/dX
    values += unique_patches  # dZ
    bytes_ = iterations * values * dtype_bytes

    return make_row(
        module="Schur complement",
        operator="block reduction + dense solve",
        shape=f"Nposes={poses}, state={state_dim}, Kpatches={unique_patches}",
        macs=macs,
        bytes_=bytes_,
        access_pattern="sparse block",
        bottleneck="packing / accumulation",
    )


def make_row(
    module: str,
    operator: str,
    shape: str,
    macs: int,
    bytes_: int,
    access_pattern: str,
    bottleneck: str,
) -> WorkloadRow:
    oi = macs / bytes_ if bytes_ else math.inf
    return WorkloadRow(module, operator, shape, macs, bytes_, oi, access_pattern, bottleneck)


def estimate_edges(
    patches_per_frame: int,
    patch_lifetime: int,
    removal_window: int,
    mode: str,
) -> int:
    edges_per_source_patch = 2 * patch_lifetime - 1
    if mode == "new-frame":
        return patches_per_frame * edges_per_source_patch
    if mode == "steady":
        return patches_per_frame * removal_window * edges_per_source_patch
    raise ValueError(f"unknown edge mode: {mode}")


def default_unique_patches(patches_per_frame: int, removal_window: int, edges: int) -> int:
    return min(edges, patches_per_frame * removal_window)


def build_rows(args: argparse.Namespace) -> list[WorkloadRow]:
    cfg = load_simple_yaml(args.config)
    patches_per_frame = int(args.patches_per_frame or cfg.get("PATCHES_PER_FRAME", 80))
    patch_lifetime = int(args.patch_lifetime or cfg.get("PATCH_LIFETIME", 12))
    removal_window = int(args.removal_window or cfg.get("REMOVAL_WINDOW", 20))
    optimization_window = int(args.optimization_window or cfg.get("OPTIMIZATION_WINDOW", 12))
    mixed_precision = bool(cfg.get("MIXED_PRECISION", True))

    nn_dtype_bytes = args.nn_dtype_bytes
    if nn_dtype_bytes is None:
        nn_dtype_bytes = 2 if mixed_precision else 4

    edges = args.edges
    if edges is None:
        edges = estimate_edges(
            patches_per_frame=patches_per_frame,
            patch_lifetime=patch_lifetime,
            removal_window=removal_window,
            mode=args.edge_mode,
        )

    unique_patches = args.unique_patches
    if unique_patches is None:
        unique_patches = default_unique_patches(patches_per_frame, removal_window, edges)

    rows = [
        estimate_feature_encoder(args.height, args.width, nn_dtype_bytes),
        estimate_update_block(edges, args.patch_size, args.corr_radius, args.corr_levels, nn_dtype_bytes),
        estimate_correlation(edges, args.patch_size, args.corr_radius, args.corr_levels, GMAP_DIM, nn_dtype_bytes),
        estimate_ba_jacobian(edges, args.ba_iterations, args.ba_dtype_bytes, args.ba_macs_per_edge),
        estimate_schur(optimization_window, unique_patches, args.ba_iterations, args.ba_dtype_bytes),
    ]
    return rows


def render_markdown(rows: Iterable[WorkloadRow], roofline_threshold: float | None = None) -> str:
    headers = [
        "Module",
        "Operator",
        "Shape",
        "MACs",
        "Bytes",
        "OI",
        "Access pattern",
        "Bottleneck",
    ]
    if roofline_threshold is not None:
        headers.append("Roofline bound")

    display_rows = [row.display(roofline_threshold) for row in rows]
    widths = {
        header: max(len(header), *(len(row[header]) for row in display_rows))
        for header in headers
    }

    def fmt_row(values: dict[str, str]) -> str:
        return "| " + " | ".join(values[header].ljust(widths[header]) for header in headers) + " |"

    lines = [
        fmt_row({header: header for header in headers}),
        "| " + " | ".join("-" * widths[header] for header in headers) + " |",
    ]
    lines.extend(fmt_row(row) for row in display_rows)
    return "\n".join(lines)


def write_csv(rows: Iterable[WorkloadRow], stream: Any, roofline_threshold: float | None = None) -> None:
    fieldnames = [
        "Module",
        "Operator",
        "Shape",
        "MACs",
        "Bytes",
        "OI",
        "Access pattern",
        "Bottleneck",
    ]
    if roofline_threshold is not None:
        fieldnames.append("Roofline bound")

    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row.display(roofline_threshold))


def rows_as_json(rows: Iterable[WorkloadRow], roofline_threshold: float | None = None) -> list[dict[str, Any]]:
    data = []
    for row in rows:
        item = asdict(row)
        if roofline_threshold is not None:
            item["roofline_bound"] = classify_roofline(row, roofline_threshold)
        data.append(item)
    return data


def roofline_from_args(args: argparse.Namespace) -> tuple[float | None, str | None]:
    if args.peak_macs is not None and args.bandwidth is not None:
        return args.peak_macs / args.bandwidth, "MAC/s over Byte/s"
    if args.peak_macs_per_cycle is not None and args.bandwidth_per_cycle is not None:
        return args.peak_macs_per_cycle / args.bandwidth_per_cycle, "MAC/cycle over Byte/cycle"
    return None, None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate DPVO module-level workload and print a roofline-style table."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help=f"DPVO yaml config. Default: {DEFAULT_CONFIG}")
    parser.add_argument("--height", type=int, default=480, help="Input image height before DPVO /4 feature stride.")
    parser.add_argument("--width", type=int, default=640, help="Input image width before DPVO /4 feature stride.")
    parser.add_argument("--patch-size", type=int, default=3, help="DPVO patch size P.")
    parser.add_argument("--corr-radius", type=int, default=3, help="Correlation radius R.")
    parser.add_argument("--corr-levels", type=int, default=2, help="Number of correlation pyramid levels.")
    parser.add_argument("--ba-iterations", type=int, default=2, help="fastba.BA iterations per update call.")
    parser.add_argument(
        "--edge-mode",
        choices=("steady", "new-frame"),
        default="steady",
        help="Estimate active steady-state graph edges or only newly appended edges. Ignored by --edges.",
    )
    parser.add_argument("--edges", type=int, help="Override active factor/edge count E.")
    parser.add_argument("--unique-patches", type=int, help="Override unique patch count K used in BA/Schur.")
    parser.add_argument("--patches-per-frame", type=int, help="Override PATCHES_PER_FRAME from config.")
    parser.add_argument("--patch-lifetime", type=int, help="Override PATCH_LIFETIME from config.")
    parser.add_argument("--removal-window", type=int, help="Override REMOVAL_WINDOW from config.")
    parser.add_argument("--optimization-window", type=int, help="Override OPTIMIZATION_WINDOW from config.")
    parser.add_argument("--nn-dtype-bytes", type=int, choices=(2, 4), help="Bytes per NN activation/weight element.")
    parser.add_argument("--ba-dtype-bytes", type=int, choices=(4, 8), default=4, help="Bytes per BA matrix element.")
    parser.add_argument(
        "--ba-macs-per-edge",
        type=int,
        default=900,
        help="Heuristic MAC-equivalent count for BA residual/Jacobian/Hessian assembly per edge per iteration.",
    )
    parser.add_argument(
        "--peak-macs",
        type=parse_si_number,
        help="Hardware peak compute in MAC/s. Accepts values like 1e12, 1T, or 250G.",
    )
    parser.add_argument(
        "--bandwidth",
        type=parse_si_number,
        help="Hardware memory bandwidth in bytes/s. Accepts values like 1e11, 100G, or 100GB/s.",
    )
    parser.add_argument(
        "--peak-macs-per-cycle",
        type=parse_si_number,
        help="Hardware peak compute in MAC/cycle. Example: 64.",
    )
    parser.add_argument(
        "--bandwidth-per-cycle",
        type=parse_si_number,
        help="Hardware memory bandwidth in bytes/cycle. Example: 8.",
    )
    parser.add_argument("--format", choices=("markdown", "csv", "json"), default="markdown")
    parser.add_argument("--output", type=Path, help="Optional output path. Prints to stdout if omitted.")
    args = parser.parse_args()
    if (args.peak_macs is None) != (args.bandwidth is None):
        parser.error("--peak-macs and --bandwidth must be provided together")
    if (args.peak_macs_per_cycle is None) != (args.bandwidth_per_cycle is None):
        parser.error("--peak-macs-per-cycle and --bandwidth-per-cycle must be provided together")
    if args.peak_macs is not None and args.peak_macs_per_cycle is not None:
        parser.error("use either per-second roofline args or per-cycle roofline args, not both")
    return args


def main() -> int:
    args = parse_args()
    rows = build_rows(args)
    roofline_threshold, roofline_basis = roofline_from_args(args)

    if args.format == "markdown":
        text = render_markdown(rows, roofline_threshold) + "\n"
        if roofline_threshold is not None:
            text += f"\nRoofline OI threshold: {roofline_threshold:.2f} MAC/Byte ({roofline_basis})\n"
    elif args.format == "json":
        data: dict[str, Any] | list[dict[str, Any]]
        if roofline_threshold is None:
            data = rows_as_json(rows)
        else:
            data = {
                "roofline_threshold_mac_per_byte": roofline_threshold,
                "roofline_basis": roofline_basis,
                "rows": rows_as_json(rows, roofline_threshold),
            }
        text = json.dumps(data, indent=2) + "\n"
    else:
        from io import StringIO

        buffer = StringIO()
        write_csv(rows, buffer, roofline_threshold)
        text = buffer.getvalue()

    if args.output:
        args.output.write_text(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
