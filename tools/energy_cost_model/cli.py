#!/usr/bin/env python3
"""Command line interface for the DPVO-on-Gemmini energy model."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.dont_write_bytecode = True

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from model import estimate_energy
    from parameters import (
        DEFAULT_DPVO_CONFIG,
        DEFAULT_GEMMINI_HEADER,
        AlgorithmParams,
        EnergyTable,
        HardwareParams,
        MappingParams,
    )
else:  # pragma: no cover - package execution path
    from .model import estimate_energy
    from .parameters import (
        DEFAULT_DPVO_CONFIG,
        DEFAULT_GEMMINI_HEADER,
        AlgorithmParams,
        EnergyTable,
        HardwareParams,
        MappingParams,
    )


def parse_si_number(text: str) -> float:
    value = text.strip().replace("_", "")
    for suffix in ("/s", "ps", "B", "b", "Hz", "hz"):
        if value.endswith(suffix):
            value = value[: -len(suffix)]
    suffixes = {"K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12, "P": 1e15}
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate per-frame dynamic energy/power for DPVO on Rocket + Gemmini."
    )

    app = parser.add_argument_group("DPVO algorithm parameters P_a")
    app.add_argument("--config", type=Path, default=DEFAULT_DPVO_CONFIG, help=f"DPVO flat YAML config. Default: {DEFAULT_DPVO_CONFIG}")
    app.add_argument("--height", type=int, default=480, help="Input image height before DPVO /4 feature stride.")
    app.add_argument("--width", type=int, default=640, help="Input image width before DPVO /4 feature stride.")
    app.add_argument("--patches-per-frame", type=int, help="Override PATCHES_PER_FRAME.")
    app.add_argument("--removal-window", type=int, help="Override REMOVAL_WINDOW.")
    app.add_argument("--optimization-window", type=int, help="Override OPTIMIZATION_WINDOW.")
    app.add_argument("--patch-lifetime", type=int, help="Override PATCH_LIFETIME.")
    app.add_argument("--patch-size", type=int, default=3, help="DPVO patch size P.")
    app.add_argument("--corr-radius", type=int, default=3, help="Correlation radius R.")
    app.add_argument("--corr-levels", type=int, default=2, help="Number of correlation pyramid levels.")
    app.add_argument("--update-iterations", type=int, default=1, help="Per-frame correlation/update/BA repetitions.")
    app.add_argument("--ba-iterations", type=int, default=2, help="fastba.BA iterations inside each update.")
    app.add_argument("--edge-mode", choices=("steady", "new-frame"), default="steady", help="Active steady-state graph or only newly appended edges.")
    app.add_argument("--edges", type=int, help="Override active factor/edge count E.")
    app.add_argument("--unique-patches", type=int, help="Override unique patch count K for BA/Schur.")
    app.add_argument("--nn-dtype-bytes", type=int, choices=(1, 2, 4, 8), help="Bytes per NN activation/weight element.")
    app.add_argument("--ba-dtype-bytes", type=int, choices=(4, 8), default=4, help="Bytes per BA scalar.")
    app.add_argument("--ba-macs-per-edge", type=int, default=900, help="MAC-equivalent BA assembly work per edge per BA iteration.")
    app.add_argument("--centroid-selection", choices=("RANDOM", "GRADIENT_BIAS"), help="Patch centroid selection strategy.")

    hw = parser.add_argument_group("Hardware parameters H/P_h")
    hw.add_argument(
        "--hardware-profile",
        choices=("generated-header", "default-int8", "fp32-default", "custom"),
        default="generated-header",
        help="Gemmini defaults. generated-header parses software/libgemmini/gemmini_params.h.",
    )
    hw.add_argument("--gemmini-header", type=Path, default=DEFAULT_GEMMINI_HEADER, help="Path to generated gemmini_params.h.")
    hw.add_argument("--gemmini-dim", type=int, help="Override square Gemmini DIM.")
    hw.add_argument("--input-bytes", type=int, choices=(1, 2, 4, 8), help="Override Gemmini input/weight bytes.")
    hw.add_argument("--acc-bytes", type=int, choices=(2, 4, 8), help="Override Gemmini accumulator bytes.")
    hw.add_argument("--sp-capacity-kib", type=float, help="Override Gemmini scratchpad capacity.")
    hw.add_argument("--acc-capacity-kib", type=float, help="Override Gemmini accumulator capacity.")
    hw.add_argument("--sp-banks", type=int, help="Override scratchpad bank count.")
    hw.add_argument("--acc-banks", type=int, help="Override accumulator bank count.")
    hw.add_argument("--dma-buswidth-bits", type=int, help="Override DMA bus width in bits.")
    hw.add_argument("--dma-maxbytes", type=int, help="Override max DMA transaction bytes.")
    hw.add_argument("--frequency", type=parse_si_number, default=1.0e9, help="Clock frequency, e.g. 1G or 800M.")
    hw.add_argument("--dram-bandwidth", type=parse_si_number, help="DRAM bandwidth in bytes/s, e.g. 6.4GB/s.")
    hw.add_argument("--l2-bandwidth", type=parse_si_number, help="L2 bandwidth in bytes/s.")
    hw.add_argument("--l1-bandwidth", type=parse_si_number, help="L1 bandwidth in bytes/s.")
    hw.add_argument("--random-l1-hit-rate", type=float, help="CPU random read L1 hit rate.")
    hw.add_argument("--random-l2-hit-rate", type=float, help="CPU random read L2 hit rate after L1 miss.")
    hw.add_argument("--sequential-l1-hit-rate", type=float, help="CPU sequential read L1 hit rate.")
    hw.add_argument("--sequential-l2-hit-rate", type=float, help="CPU sequential read L2 hit rate after L1 miss.")

    mp = parser.add_argument_group("Mapping choices M_k")
    mp.add_argument("--encoder-mapping", choices=("gemmini", "cpu"), default="gemmini")
    mp.add_argument("--update-mapping", choices=("gemmini", "cpu"), default="gemmini")
    mp.add_argument("--corr-mapping", choices=("cpu", "gemmini"), default="cpu")
    mp.add_argument("--patch-mapping", choices=("cpu",), default="cpu")
    mp.add_argument("--softagg-mapping", choices=("cpu",), default="cpu")
    mp.add_argument("--ba-mapping", choices=("cpu",), default="cpu")
    mp.add_argument("--graph-mapping", choices=("cpu",), default="cpu")
    mp.add_argument("--no-overlap-dma-compute", action="store_true", help="Use sum instead of max for Gemmini compute/DMA/memory timing.")

    out = parser.add_argument_group("Output")
    out.add_argument("--energy-table", type=Path, help="JSON mapping action name to pJ/action. Values override defaults.")
    out.add_argument("--format", choices=("markdown", "json", "csv"), default="markdown")
    out.add_argument("--output", type=Path, help="Write report to path instead of stdout.")
    out.add_argument("--show-actions", action="store_true", help="Include aggregate action counts in markdown/json.")

    return parser.parse_args()


def build_algorithm(args: argparse.Namespace) -> AlgorithmParams:
    return AlgorithmParams.from_config(
        args.config,
        height=args.height,
        width=args.width,
        patches_per_frame=args.patches_per_frame,
        removal_window=args.removal_window,
        optimization_window=args.optimization_window,
        patch_lifetime=args.patch_lifetime,
        patch_size=args.patch_size,
        corr_radius=args.corr_radius,
        corr_levels=args.corr_levels,
        update_iterations=args.update_iterations,
        ba_iterations=args.ba_iterations,
        edge_mode=args.edge_mode,
        edges=args.edges,
        unique_patches=args.unique_patches,
        nn_dtype_bytes=args.nn_dtype_bytes,
        ba_dtype_bytes=args.ba_dtype_bytes,
        ba_macs_per_edge=args.ba_macs_per_edge,
        centroid_selection=args.centroid_selection,
    )


def build_hardware(args: argparse.Namespace) -> HardwareParams:
    hardware = HardwareParams.from_profile(args.hardware_profile, args.gemmini_header)
    overrides = {
        "dim": args.gemmini_dim,
        "input_bytes": args.input_bytes,
        "acc_bytes": args.acc_bytes,
        "sp_capacity_kib": args.sp_capacity_kib,
        "acc_capacity_kib": args.acc_capacity_kib,
        "sp_banks": args.sp_banks,
        "acc_banks": args.acc_banks,
        "dma_buswidth_bits": args.dma_buswidth_bits,
        "dma_maxbytes": args.dma_maxbytes,
        "frequency_hz": args.frequency,
        "random_l1_hit_rate": args.random_l1_hit_rate,
        "random_l2_hit_rate": args.random_l2_hit_rate,
        "sequential_l1_hit_rate": args.sequential_l1_hit_rate,
        "sequential_l2_hit_rate": args.sequential_l2_hit_rate,
    }
    clean = {key: value for key, value in overrides.items() if value is not None}
    hardware = replace(hardware, **clean)
    bw_overrides = {}
    if args.dram_bandwidth is not None:
        bw_overrides["dram_bandwidth_bytes_per_cycle"] = args.dram_bandwidth / hardware.frequency_hz
    if args.l2_bandwidth is not None:
        bw_overrides["l2_bandwidth_bytes_per_cycle"] = args.l2_bandwidth / hardware.frequency_hz
    if args.l1_bandwidth is not None:
        bw_overrides["l1_bandwidth_bytes_per_cycle"] = args.l1_bandwidth / hardware.frequency_hz
    if bw_overrides:
        hardware = replace(hardware, **bw_overrides)
    return hardware


def build_mapping(args: argparse.Namespace) -> MappingParams:
    return MappingParams(
        encoder=args.encoder_mapping,
        update_dense=args.update_mapping,
        correlation=args.corr_mapping,
        patch_extraction=args.patch_mapping,
        soft_aggregation=args.softagg_mapping,
        ba=args.ba_mapping,
        graph_management=args.graph_mapping,
        overlap_dma_compute=not args.no_overlap_dma_compute,
    )


def main() -> int:
    args = parse_args()
    algorithm = build_algorithm(args)
    hardware = build_hardware(args)
    mapping = build_mapping(args)
    energy_table = EnergyTable.defaults(hardware)
    if args.energy_table is not None:
        energy_table = EnergyTable.from_json(args.energy_table, energy_table)

    report = estimate_energy(algorithm, hardware, mapping, energy_table)
    if args.format == "json":
        text = report.to_json(include_actions=args.show_actions)
    elif args.format == "csv":
        text = report.to_csv()
    else:
        text = report.to_markdown(include_actions=args.show_actions)

    if args.output:
        args.output.write_text(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
