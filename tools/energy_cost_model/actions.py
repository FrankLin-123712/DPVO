"""Action-count utilities for the energy model."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

try:
    from .parameters import HardwareParams, ceil_div, round_up
    from .workloads import DenseOp, IrregularWorkload
except ImportError:  # pragma: no cover - script execution fallback
    from parameters import HardwareParams, ceil_div, round_up
    from workloads import DenseOp, IrregularWorkload


@dataclass
class ActionCounts:
    counts: dict[str, float] = field(default_factory=dict)

    def add(self, name: str, count: float) -> None:
        if count <= 0:
            return
        self.counts[name] = self.counts.get(name, 0.0) + float(count)

    def merge(self, other: "ActionCounts") -> None:
        for name, count in other.counts.items():
            self.add(name, count)

    def prefixed(self, prefix: str) -> float:
        return sum(count for name, count in self.counts.items() if name.startswith(prefix))

    def total_bytes(self, prefixes: tuple[str, ...]) -> float:
        return sum(
            count
            for name, count in self.counts.items()
            if name.endswith("_byte") and any(name.startswith(prefix) for prefix in prefixes)
        )


@dataclass(frozen=True)
class DenseEstimate:
    actions: ActionCounts
    useful_macs: int
    executed_macs: int
    utilization: float
    cycles: float
    dram_read_bytes: float
    dram_write_bytes: float
    notes: str


@dataclass(frozen=True)
class CpuEstimate:
    actions: ActionCounts
    useful_macs: int
    executed_macs: int
    cycles: float
    dram_read_bytes: float
    dram_write_bytes: float
    notes: str


def estimate_dense_on_gemmini(
    op: DenseOp,
    hardware: HardwareParams,
    overlap_dma_compute: bool,
) -> DenseEstimate:
    dim = hardware.dim
    padded_m = round_up(op.m, dim)
    padded_n = round_up(op.n, dim)
    padded_k = round_up(op.k, dim)
    m_tiles = ceil_div(op.m, dim)
    n_tiles = ceil_div(op.n, dim)
    k_tiles = ceil_div(op.k, dim)
    tile_products = m_tiles * n_tiles * k_tiles

    useful_macs = op.macs
    executed_macs = padded_m * padded_n * padded_k
    utilization = useful_macs / executed_macs if executed_macs else 1.0

    # A conservative reuse model: if an entire operand fits in half the
    # scratchpad, load it once. Otherwise reload it across the orthogonal tile
    # dimension.
    a_tensor_bytes = op.m * op.k * hardware.input_bytes
    b_tensor_bytes = op.k * op.n * hardware.input_bytes
    c_tensor_bytes = op.m * op.n * hardware.acc_bytes
    half_spad = max(1, hardware.sp_capacity_bytes // 2)
    a_reuse_factor = 1 if a_tensor_bytes <= half_spad else n_tiles
    b_reuse_factor = 1 if b_tensor_bytes <= half_spad else m_tiles

    dram_read_bytes = a_tensor_bytes * a_reuse_factor + b_tensor_bytes * b_reuse_factor
    dram_write_bytes = op.m * op.n * hardware.input_bytes

    # Low-level SRAM/PE traffic. The arrays stream operands every executed MAC;
    # accumulator traffic is per output tile per K tile.
    spad_read_bytes = executed_macs * 2 * hardware.input_bytes
    spad_write_bytes = dram_read_bytes
    acc_bytes_per_output = padded_m * padded_n * hardware.acc_bytes
    acc_read_bytes = acc_bytes_per_output * max(0, k_tiles - 1)
    acc_write_bytes = acc_bytes_per_output * k_tiles
    pe_reg_read_bytes = executed_macs * 2 * hardware.input_bytes
    pe_reg_write_bytes = executed_macs * hardware.acc_bytes / max(1, dim)

    actions = ActionCounts()
    actions.add("gemmini.mac", executed_macs)
    actions.add("dma.read_byte", dram_read_bytes)
    actions.add("dma.write_byte", dram_write_bytes)
    actions.add("dram.sequential_read_byte", dram_read_bytes)
    actions.add("dram.write_byte", dram_write_bytes)
    actions.add("spad.read_byte", spad_read_bytes)
    actions.add("spad.write_byte", spad_write_bytes)
    actions.add("acc.read_byte", acc_read_bytes)
    actions.add("acc.write_byte", acc_write_bytes)
    actions.add("pe.reg_read_byte", pe_reg_read_bytes)
    actions.add("pe.reg_write_byte", pe_reg_write_bytes)
    actions.add("sync.rocc", 1)

    compute_cycles = math.ceil(executed_macs / max(1, hardware.pe_count))
    fill_drain_cycles = tile_products * (2 * dim + k_tiles)
    dma_cycles = (dram_read_bytes + dram_write_bytes) / hardware.dma_bandwidth_bytes_per_cycle
    memory_cycles = (dram_read_bytes + dram_write_bytes) / hardware.dram_bandwidth_bytes_per_cycle
    command_cycles = hardware.rocc_command_cycles * max(1, m_tiles * n_tiles)
    if overlap_dma_compute:
        cycles = max(compute_cycles + fill_drain_cycles, dma_cycles, memory_cycles) + command_cycles
    else:
        cycles = compute_cycles + fill_drain_cycles + dma_cycles + memory_cycles + command_cycles

    notes = (
        f"{op.kind} {op.m}x{op.k} * {op.k}x{op.n}, "
        f"tiles M/N/K={m_tiles}/{n_tiles}/{k_tiles}, util={utilization:.3f}"
    )
    return DenseEstimate(
        actions=actions,
        useful_macs=useful_macs,
        executed_macs=executed_macs,
        utilization=utilization,
        cycles=cycles,
        dram_read_bytes=dram_read_bytes,
        dram_write_bytes=dram_write_bytes,
        notes=notes,
    )


def estimate_dense_on_cpu(op: DenseOp, hardware: HardwareParams) -> CpuEstimate:
    actions = ActionCounts()
    read_bytes = op.m * op.k * op.input_bytes + op.k * op.n * op.weight_bytes
    write_bytes = op.m * op.n * op.output_bytes
    memory_read_hierarchy(actions, read_bytes, "sequential", hardware)
    memory_write_hierarchy(actions, write_bytes, hardware)
    actions.add("cpu.mac", op.macs)
    actions.add("cpu.alu", op.m * op.n)
    actions.add("cpu.branch", max(1, op.m // 16))

    compute_cycles = op.macs / max(1e-9, hardware.cpu_peak_macs_per_cycle)
    l1_cycles = (read_bytes + write_bytes) / hardware.l1_bandwidth_bytes_per_cycle
    l2_bytes = read_bytes * (1 - hardware.sequential_l1_hit_rate) + write_bytes * (1 - hardware.write_l1_hit_rate)
    l2_cycles = l2_bytes / hardware.l2_bandwidth_bytes_per_cycle
    dram_read = read_bytes * (1 - hardware.sequential_l1_hit_rate) * (1 - hardware.sequential_l2_hit_rate)
    dram_write = write_bytes * (1 - hardware.write_l1_hit_rate) * (1 - hardware.write_l2_hit_rate)
    dram_cycles = (dram_read + dram_write) / hardware.dram_bandwidth_bytes_per_cycle
    cycles = max(compute_cycles, l1_cycles, l2_cycles, dram_cycles) + hardware.cpu_sync_cycles

    return CpuEstimate(
        actions=actions,
        useful_macs=op.macs,
        executed_macs=op.macs,
        cycles=cycles,
        dram_read_bytes=dram_read,
        dram_write_bytes=dram_write,
        notes=f"CPU dense fallback for {op.kind} {op.m}x{op.k} * {op.k}x{op.n}",
    )


def estimate_irregular_on_cpu(work: IrregularWorkload, hardware: HardwareParams) -> CpuEstimate:
    actions = ActionCounts()
    memory_read_hierarchy(actions, work.read_bytes, work.access_pattern, hardware)
    memory_write_hierarchy(actions, work.write_bytes, hardware)
    if work.metadata_bytes:
        actions.add("metadata.read_byte", work.metadata_bytes)
        memory_read_hierarchy(actions, work.metadata_bytes, "random", hardware)
    actions.add("cpu.mac", work.macs)
    actions.add("cpu.alu", work.alu_ops)
    actions.add("cpu.atomic", work.atomics)
    actions.add("cpu.branch", work.branches)
    actions.add("sync.rocc", work.syncs)

    compute_cycles = (
        work.macs / max(1e-9, hardware.cpu_peak_macs_per_cycle)
        + work.alu_ops / max(1e-9, hardware.cpu_peak_alu_ops_per_cycle)
        + work.atomics * 4
        + work.branches * 0.25
    )
    l1_cycles = (work.read_bytes + work.write_bytes + work.metadata_bytes) / hardware.l1_bandwidth_bytes_per_cycle
    l2_read, dram_read = lower_memory_bytes(work.read_bytes + work.metadata_bytes, work.access_pattern, hardware)
    l2_write, dram_write = lower_write_bytes(work.write_bytes, hardware)
    l2_cycles = (l2_read + l2_write) / hardware.l2_bandwidth_bytes_per_cycle
    dram_cycles = (dram_read + dram_write) / hardware.dram_bandwidth_bytes_per_cycle
    cycles = max(compute_cycles, l1_cycles, l2_cycles, dram_cycles) + work.syncs * hardware.cpu_sync_cycles

    return CpuEstimate(
        actions=actions,
        useful_macs=work.macs,
        executed_macs=work.macs,
        cycles=cycles,
        dram_read_bytes=dram_read,
        dram_write_bytes=dram_write,
        notes=f"{work.kind}, {work.access_pattern} access",
    )


def memory_read_hierarchy(
    actions: ActionCounts,
    bytes_: float,
    access_pattern: str,
    hardware: HardwareParams,
) -> None:
    if bytes_ <= 0:
        return
    pattern = "sequential" if access_pattern == "sequential" else "random"
    l2_bytes, dram_bytes = lower_memory_bytes(bytes_, pattern, hardware)
    actions.add(f"l1.{pattern}_read_byte", bytes_)
    actions.add(f"l2.{pattern}_read_byte", l2_bytes)
    actions.add(f"dram.{pattern}_read_byte", dram_bytes)


def memory_write_hierarchy(actions: ActionCounts, bytes_: float, hardware: HardwareParams) -> None:
    if bytes_ <= 0:
        return
    l2_bytes, dram_bytes = lower_write_bytes(bytes_, hardware)
    actions.add("l1.write_byte", bytes_)
    actions.add("l2.write_byte", l2_bytes)
    actions.add("dram.write_byte", dram_bytes)


def lower_memory_bytes(bytes_: float, access_pattern: str, hardware: HardwareParams) -> tuple[float, float]:
    if access_pattern == "sequential":
        h1 = hardware.sequential_l1_hit_rate
        h2 = hardware.sequential_l2_hit_rate
    else:
        h1 = hardware.random_l1_hit_rate
        h2 = hardware.random_l2_hit_rate
    l2_bytes = bytes_ * (1 - h1)
    dram_bytes = l2_bytes * (1 - h2)
    return l2_bytes, dram_bytes


def lower_write_bytes(bytes_: float, hardware: HardwareParams) -> tuple[float, float]:
    l2_bytes = bytes_ * (1 - hardware.write_l1_hit_rate)
    dram_bytes = l2_bytes * (1 - hardware.write_l2_hit_rate)
    return l2_bytes, dram_bytes
