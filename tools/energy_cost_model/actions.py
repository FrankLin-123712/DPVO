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


def choose_dense_blocking(
    m_tiles: int,
    n_tiles: int,
    k_tiles: int,
    a_tensor_bytes: int,
    b_tensor_bytes: int,
    operand_tile_bytes: int,
    accumulator_tile_bytes: int,
    hardware: HardwareParams,
) -> tuple[int, int, int, int, int]:
    """Choose a cheap capacity-feasible GEMM block.

    An MxN output block stays in the accumulator while K panels stream
    through the scratchpad.  This small analytical search captures the main
    capacity/reload threshold without simulating individual cycles or banks.
    """
    spad_tiles = hardware.sp_capacity_bytes // operand_tile_bytes
    acc_tiles = hardware.acc_capacity_bytes // accumulator_tile_bytes
    if spad_tiles < 2 or acc_tiles < 1:
        raise ValueError("Gemmini local memories cannot hold the minimum GEMM tile set")

    best: tuple[float, int, int, int, int, int] | None = None
    max_m_block = min(m_tiles, acc_tiles, spad_tiles - 1)
    for m_block in range(1, max_m_block + 1):
        max_n_block = min(
            n_tiles,
            acc_tiles // m_block,
            spad_tiles - m_block,
        )
        for n_block in range(1, max_n_block + 1):
            k_block = min(k_tiles, spad_tiles // (m_block + n_block))
            if k_block < 1:
                continue
            a_reloads = ceil_div(n_tiles, n_block)
            b_reloads = ceil_div(m_tiles, m_block)
            read_bytes = a_tensor_bytes * a_reloads + b_tensor_bytes * b_reloads
            # First minimize off-chip bytes.  Ties prefer fewer panels, a
            # deeper K block, and a larger output block.
            key = (
                read_bytes,
                a_reloads + b_reloads,
                -k_block,
                -(m_block * n_block),
                m_block,
                n_block,
            )
            if best is None or key < best:
                best = key

    if best is None:  # guarded by the minimum-capacity checks above
        raise ValueError("no capacity-feasible Gemmini blocking was found")
    _, _, neg_k_block, _, m_block, n_block = best
    return (
        m_block,
        n_block,
        -neg_k_block,
        ceil_div(n_tiles, n_block),
        ceil_div(m_tiles, m_block),
    )


def estimate_dense_on_gemmini(
    op: DenseOp,
    hardware: HardwareParams,
    overlap_dma_compute: bool,
    dataflow: str = "WS",
) -> DenseEstimate:
    if op.input_bytes != hardware.input_bytes or op.weight_bytes != hardware.input_bytes:
        raise ValueError(
            f"{op.name} uses {op.input_bytes}/{op.weight_bytes}-byte operands but "
            f"{hardware.name} expects {hardware.input_bytes}-byte Gemmini operands. "
            "Model quantization/precision conversion explicitly before offload."
        )
    if dataflow not in {"WS", "OS"}:
        raise ValueError(f"unsupported Gemmini dataflow: {dataflow}")

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

    a_tensor_bytes = op.m * op.k * op.input_bytes
    b_tensor_bytes = op.k * op.n * op.weight_bytes
    output_bytes = op.m * op.n * op.output_bytes

    operand_tile_bytes = dim * dim * hardware.input_bytes
    accumulator_tile_bytes = dim * dim * hardware.acc_bytes
    if 2 * operand_tile_bytes > hardware.sp_capacity_bytes:
        raise ValueError(
            f"scratchpad cannot hold one A/B tile pair for {op.name}: "
            f"need {2 * operand_tile_bytes} B, have {hardware.sp_capacity_bytes} B"
        )
    if accumulator_tile_bytes > hardware.acc_capacity_bytes:
        raise ValueError(
            f"accumulator cannot hold one output tile for {op.name}: "
            f"need {accumulator_tile_bytes} B, have {hardware.acc_capacity_bytes} B"
        )

    # Select an output block that fits both operand panels in SPAD and partial
    # outputs in ACC. This avoids the pessimistic all-or-nothing rule that
    # reloaded a complete A tensor for every N tile as soon as A exceeded half
    # the scratchpad.
    m_block, n_block, k_block, a_reload_factor, b_reload_factor = choose_dense_blocking(
        m_tiles=m_tiles,
        n_tiles=n_tiles,
        k_tiles=k_tiles,
        a_tensor_bytes=a_tensor_bytes,
        b_tensor_bytes=b_tensor_bytes,
        operand_tile_bytes=operand_tile_bytes,
        accumulator_tile_bytes=accumulator_tile_bytes,
        hardware=hardware,
    )

    dram_read_bytes = a_tensor_bytes * a_reload_factor + b_tensor_bytes * b_reload_factor
    dram_write_bytes = output_bytes

    # Count operand injection at tile granularity. Internal PE forwarding/reuse
    # is represented by executed MAC energy and is not charged again as a SPAD
    # read for every MAC.
    a_spad_tiles = tile_products
    b_spad_tiles = (
        n_tiles * k_tiles * ceil_div(m_tiles, m_block)
        if dataflow == "WS"
        else tile_products
    )
    spad_read_bytes = (a_spad_tiles + b_spad_tiles) * operand_tile_bytes
    spad_write_bytes = dram_read_bytes
    acc_bytes_per_output = padded_m * padded_n * hardware.acc_bytes
    if dataflow == "WS":
        acc_read_bytes = acc_bytes_per_output * max(0, k_tiles - 1)
        acc_write_bytes = acc_bytes_per_output * k_tiles
    else:
        acc_read_bytes = 0
        acc_write_bytes = acc_bytes_per_output

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
    actions.add(
        "dma.transaction",
        ceil_div(int(math.ceil(dram_read_bytes + dram_write_bytes)), hardware.dma_maxbytes),
    )
    actions.add("sync.rocc", 1)

    # K tiles for one output tile stream back-to-back. Pay the systolic
    # fill/drain once per M/N output tile, avoiding the previous O(K_tiles^2)
    # term.
    compute_cycles = m_tiles * n_tiles * padded_k
    fill_drain_cycles = m_tiles * n_tiles * max(0, 2 * dim - 2)
    dma_cycles = (dram_read_bytes + dram_write_bytes) / hardware.dma_bandwidth_bytes_per_cycle
    memory_cycles = (dram_read_bytes + dram_write_bytes) / hardware.dram_bandwidth_bytes_per_cycle
    command_cycles = hardware.rocc_command_cycles * max(1, m_tiles * n_tiles)
    if overlap_dma_compute:
        cycles = (
            max(compute_cycles + fill_drain_cycles, dma_cycles, memory_cycles)
            + command_cycles
            + hardware.cpu_sync_cycles
        )
    else:
        cycles = (
            compute_cycles
            + fill_drain_cycles
            + dma_cycles
            + memory_cycles
            + command_cycles
            + hardware.cpu_sync_cycles
        )

    notes = (
        f"{op.kind} {op.m}x{op.k} * {op.k}x{op.n}, "
        f"tiles M/N/K={m_tiles}/{n_tiles}/{k_tiles}, "
        f"block={m_block}/{n_block}/{k_block}, reload A/B={a_reload_factor}/{b_reload_factor}, "
        f"dataflow={dataflow}, util={utilization:.3f}"
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
    actions.add(cpu_mac_action(precision_from_bytes(op.input_bytes)), op.macs)
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
    memory_read_hierarchy(
        actions,
        work.read_bytes,
        work.access_pattern,
        hardware,
        random_line_utilization=work.random_line_utilization,
    )
    memory_write_hierarchy(actions, work.write_bytes, hardware)
    if work.metadata_bytes:
        actions.add("metadata.read_byte", work.metadata_bytes)
        memory_read_hierarchy(
            actions,
            work.metadata_bytes,
            "random",
            hardware,
            random_line_utilization=min(1.0, 8 / hardware.cache_line_bytes),
        )
    actions.add(cpu_mac_action(work.precision), work.macs)
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
    l2_read, dram_read = lower_memory_bytes(
        work.read_bytes,
        work.access_pattern,
        hardware,
        random_line_utilization=work.random_line_utilization,
    )
    metadata_l2, metadata_dram = lower_memory_bytes(
        work.metadata_bytes,
        "random",
        hardware,
        random_line_utilization=min(1.0, 8 / hardware.cache_line_bytes),
    )
    l2_read += metadata_l2
    dram_read += metadata_dram
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
    random_line_utilization: float = 1.0,
) -> None:
    if bytes_ <= 0:
        return
    pattern = "sequential" if access_pattern == "sequential" else "random"
    l2_bytes, dram_bytes = lower_memory_bytes(
        bytes_,
        pattern,
        hardware,
        random_line_utilization=random_line_utilization,
    )
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


def lower_memory_bytes(
    bytes_: float,
    access_pattern: str,
    hardware: HardwareParams,
    random_line_utilization: float = 1.0,
) -> tuple[float, float]:
    if bytes_ <= 0:
        return 0.0, 0.0
    if access_pattern == "sequential":
        h1 = hardware.sequential_l1_hit_rate
        h2 = hardware.sequential_l2_hit_rate
        transferred_bytes = bytes_
    else:
        h1 = hardware.random_l1_hit_rate
        h2 = hardware.random_l2_hit_rate
        if not 0.0 < random_line_utilization <= 1.0:
            raise ValueError("random_line_utilization must be in (0, 1]")
        useful_per_line = hardware.cache_line_bytes * random_line_utilization
        unique_lines = math.ceil(bytes_ / useful_per_line)
        transferred_bytes = unique_lines * hardware.cache_line_bytes
    l2_bytes = transferred_bytes * (1 - h1)
    dram_bytes = l2_bytes * (1 - h2)
    return l2_bytes, dram_bytes


def lower_write_bytes(bytes_: float, hardware: HardwareParams) -> tuple[float, float]:
    l2_bytes = bytes_ * (1 - hardware.write_l1_hit_rate)
    dram_bytes = l2_bytes * (1 - hardware.write_l2_hit_rate)
    return l2_bytes, dram_bytes


def precision_from_bytes(dtype_bytes: int) -> str:
    try:
        return {1: "int8", 2: "fp16", 4: "fp32", 8: "fp64"}[dtype_bytes]
    except KeyError as exc:
        raise ValueError(f"unsupported CPU operand width: {dtype_bytes} bytes") from exc


def cpu_mac_action(precision: str) -> str:
    supported = {"int8", "int16", "fp16", "fp32", "fp64"}
    if precision not in supported:
        raise ValueError(f"unsupported CPU MAC precision: {precision}")
    return f"cpu.mac.{precision}"
