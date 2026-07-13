"""Top-level DPVO-on-Gemmini energy and dynamic-power estimator."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from io import StringIO
from typing import Any, Iterable

try:
    from .actions import (
        ActionCounts,
        estimate_dense_on_cpu,
        estimate_dense_on_gemmini,
        estimate_irregular_on_cpu,
    )
    from .parameters import AlgorithmParams, EnergyTable, HardwareParams, MappingParams, dataclass_dict
    from .workloads import IrregularWorkload, ModuleWorkload, build_dpvo_workloads
except ImportError:  # pragma: no cover - script execution fallback
    from actions import (
        ActionCounts,
        estimate_dense_on_cpu,
        estimate_dense_on_gemmini,
        estimate_irregular_on_cpu,
    )
    from parameters import AlgorithmParams, EnergyTable, HardwareParams, MappingParams, dataclass_dict
    from workloads import IrregularWorkload, ModuleWorkload, build_dpvo_workloads


@dataclass(frozen=True)
class ModuleResult:
    module: str
    mapping: str
    useful_macs: int
    executed_macs: int
    utilization: float
    dram_read_bytes: float
    dram_write_bytes: float
    energy_pj: float
    cycles: float
    seconds: float
    notes: str
    actions: dict[str, float]

    @property
    def dynamic_power_w(self) -> float:
        if self.seconds <= 0:
            return 0.0
        return self.energy_pj * 1e-12 / self.seconds


@dataclass(frozen=True)
class EnergyReport:
    algorithm: AlgorithmParams
    hardware: HardwareParams
    mapping: MappingParams
    energy_table: dict[str, float]
    modules: tuple[ModuleResult, ...]
    target_fps: float | None = None

    @property
    def total_energy_pj(self) -> float:
        return sum(module.energy_pj for module in self.modules)

    @property
    def total_cycles(self) -> float:
        return sum(module.cycles for module in self.modules)

    @property
    def total_seconds(self) -> float:
        return self.total_cycles / self.hardware.frequency_hz

    @property
    def dynamic_power_w(self) -> float:
        if self.total_seconds <= 0:
            return 0.0
        return self.total_energy_pj * 1e-12 / self.total_seconds

    @property
    def frames_per_second(self) -> float:
        if self.total_seconds <= 0:
            return math.inf
        return 1.0 / self.total_seconds

    @property
    def target_dynamic_power_w(self) -> float | None:
        if self.target_fps is None:
            return None
        return self.total_energy_pj * 1e-12 * self.target_fps

    @property
    def target_throughput_feasible(self) -> bool | None:
        if self.target_fps is None:
            return None
        return self.frames_per_second >= self.target_fps

    def aggregate_actions(self) -> dict[str, float]:
        totals: dict[str, float] = {}
        for module in self.modules:
            for action, count in module.actions.items():
                totals[action] = totals.get(action, 0.0) + count
        return dict(sorted(totals.items()))

    def to_dict(self, include_actions: bool = True) -> dict[str, Any]:
        modules = []
        for module in self.modules:
            item = asdict(module)
            if not include_actions:
                item.pop("actions", None)
            item["dynamic_power_w"] = module.dynamic_power_w
            modules.append(item)
        data = {
            "algorithm": dataclass_dict(self.algorithm),
            "hardware": dataclass_dict(self.hardware),
            "mapping": dataclass_dict(self.mapping),
            "summary": {
                "energy_per_frame_pj": self.total_energy_pj,
                "energy_per_frame_j": self.total_energy_pj * 1e-12,
                "cycles_per_frame": self.total_cycles,
                "seconds_per_frame": self.total_seconds,
                "estimated_fps": self.frames_per_second,
                "active_dynamic_power_w": self.dynamic_power_w,
                "dynamic_power_w": self.dynamic_power_w,
                "target_fps": self.target_fps,
                "target_dynamic_power_w": self.target_dynamic_power_w,
                "target_throughput_feasible": self.target_throughput_feasible,
            },
            "modules": modules,
        }
        if include_actions:
            data["aggregate_actions"] = self.aggregate_actions()
            data["energy_table_pj_per_action"] = self.energy_table
        return data

    def to_json(self, include_actions: bool = True) -> str:
        return json.dumps(self.to_dict(include_actions=include_actions), indent=2) + "\n"

    def to_csv(self) -> str:
        buffer = StringIO()
        fields = [
            "module",
            "mapping",
            "useful_macs",
            "executed_macs",
            "utilization",
            "dram_read_bytes",
            "dram_write_bytes",
            "energy_pj",
            "energy_j",
            "cycles",
            "seconds",
            "dynamic_power_w",
            "notes",
        ]
        writer = csv.DictWriter(buffer, fieldnames=fields)
        writer.writeheader()
        for module in self.modules:
            writer.writerow(
                {
                    "module": module.module,
                    "mapping": module.mapping,
                    "useful_macs": module.useful_macs,
                    "executed_macs": module.executed_macs,
                    "utilization": module.utilization,
                    "dram_read_bytes": module.dram_read_bytes,
                    "dram_write_bytes": module.dram_write_bytes,
                    "energy_pj": module.energy_pj,
                    "energy_j": module.energy_pj * 1e-12,
                    "cycles": module.cycles,
                    "seconds": module.seconds,
                    "dynamic_power_w": module.dynamic_power_w,
                    "notes": module.notes,
                }
            )
        return buffer.getvalue()

    def to_markdown(self, include_actions: bool = False) -> str:
        rows = [
            {
                "Module": module.module,
                "Mapping": module.mapping,
                "Useful MACs": human_count(module.useful_macs),
                "Exec MACs": human_count(module.executed_macs),
                "Util.": f"{module.utilization:.2f}",
                "DRAM R": human_bytes(module.dram_read_bytes),
                "DRAM W": human_bytes(module.dram_write_bytes),
                "Energy": human_energy_j(module.energy_pj * 1e-12),
                "Time": human_seconds(module.seconds),
                "Pdyn": f"{module.dynamic_power_w:.3f} W",
            }
            for module in self.modules
        ]
        headers = list(rows[0].keys()) if rows else []
        lines = []
        lines.append(
            f"Energy/frame: {human_energy_j(self.total_energy_pj * 1e-12)}; "
            f"Latency/frame: {human_seconds(self.total_seconds)}; "
            f"Throughput: {self.frames_per_second:.2f} frame/s; "
            f"Active dynamic power: {self.dynamic_power_w:.3f} W"
        )
        if self.target_fps is not None:
            lines.append(
                f"At target {self.target_fps:.2f} frame/s: "
                f"dynamic power={self.target_dynamic_power_w:.3f} W; "
                f"feasible={'yes' if self.target_throughput_feasible else 'no'}"
            )
        lines.append(
            f"Hardware: {self.hardware.name}, DIM={self.hardware.dim}, "
            f"{self.hardware.input_precision} input, {self.hardware.acc_precision} acc, "
            f"SPAD={self.hardware.sp_capacity_kib:.0f} KiB, ACC={self.hardware.acc_capacity_kib:.0f} KiB"
        )
        lines.append("")
        lines.append(render_table(headers, rows))
        if include_actions:
            action_rows = [
                {
                    "Action": action,
                    "Count": human_count(count),
                    "Unit pJ": f"{self.energy_table.get(action, 0.0):.3g}",
                    "Energy": human_energy_j(count * self.energy_table.get(action, 0.0) * 1e-12),
                }
                for action, count in self.aggregate_actions().items()
            ]
            lines.append("")
            lines.append(render_table(["Action", "Count", "Unit pJ", "Energy"], action_rows))
        return "\n".join(lines) + "\n"


def estimate_energy(
    algorithm: AlgorithmParams,
    hardware: HardwareParams,
    mapping: MappingParams | None = None,
    energy_table: EnergyTable | None = None,
    target_fps: float | None = None,
) -> EnergyReport:
    if target_fps is not None and target_fps <= 0:
        raise ValueError(f"target_fps must be positive, got {target_fps}")
    mapping = mapping or MappingParams()
    energy_table = energy_table or EnergyTable.defaults(hardware)
    module_results = [
        estimate_module(workload, hardware, mapping, energy_table)
        for workload in build_dpvo_workloads(algorithm)
    ]
    return EnergyReport(
        algorithm=algorithm,
        hardware=hardware,
        mapping=mapping,
        energy_table=energy_table.as_dict(),
        modules=tuple(module_results),
        target_fps=target_fps,
    )


def estimate_module(
    workload: ModuleWorkload,
    hardware: HardwareParams,
    mapping: MappingParams,
    energy_table: EnergyTable,
) -> ModuleResult:
    module_mapping = mapping_for_module(workload.name, mapping)
    actions = ActionCounts()
    useful_macs = 0
    executed_macs = 0
    utilization_numer = 0.0
    utilization_denom = 0.0
    cycles = 0.0
    dram_read_bytes = 0.0
    dram_write_bytes = 0.0
    notes: list[str] = []
    effective_mappings: set[str] = set()

    if workload.name == "correlation_lookup" and module_mapping == "gemmini":
        raise NotImplementedError(
            "Gemmini correlation mapping is intentionally disabled: each edge/patch row "
            "has a different candidate tensor and is not a conventional shared-B GEMM."
        )

    dense_ops = selected_dense_ops(workload, module_mapping)
    irregular = selected_irregular_workloads(workload, module_mapping)

    for op in dense_ops:
        if module_mapping == "gemmini":
            gemmini_estimate = estimate_dense_on_gemmini(
                op,
                hardware,
                mapping.overlap_dma_compute,
                mapping.dataflow,
            )
            if gemmini_estimate.utilization < mapping.gemmini_min_utilization_for_offload:
                estimate = estimate_dense_on_cpu(op, hardware)
                effective_mappings.add("cpu-auto")
                notes.append(
                    f"{op.name} kept on CPU because Gemmini utilization "
                    f"{gemmini_estimate.utilization:.3f} is below "
                    f"{mapping.gemmini_min_utilization_for_offload:.3f}"
                )
            else:
                estimate = gemmini_estimate
                effective_mappings.add("gemmini")
        elif module_mapping == "cpu":
            estimate = estimate_dense_on_cpu(op, hardware)
            effective_mappings.add("cpu")
        else:
            raise ValueError(f"unsupported mapping {module_mapping} for dense op {op.name}")
        actions.merge(estimate.actions)
        useful_macs += estimate.useful_macs
        executed_macs += estimate.executed_macs
        utilization_numer += estimate.utilization * estimate.executed_macs
        utilization_denom += estimate.executed_macs
        cycles += estimate.cycles
        dram_read_bytes += estimate.dram_read_bytes
        dram_write_bytes += estimate.dram_write_bytes
        notes.append(estimate.notes)

    for work in irregular:
        if module_mapping not in {"cpu", "gemmini"}:
            raise ValueError(f"unsupported mapping {module_mapping} for irregular work {work.name}")
        estimate = estimate_irregular_on_cpu(work, hardware)
        effective_mappings.add("cpu")
        actions.merge(estimate.actions)
        useful_macs += estimate.useful_macs
        executed_macs += estimate.executed_macs
        utilization_numer += estimate.executed_macs
        utilization_denom += estimate.executed_macs
        cycles += estimate.cycles
        dram_read_bytes += estimate.dram_read_bytes
        dram_write_bytes += estimate.dram_write_bytes
        notes.append(estimate.notes)

    if utilization_denom > 0:
        utilization = utilization_numer / utilization_denom
    else:
        utilization = 1.0

    energy_pj = energy_table.energy_pj(actions.counts)
    seconds = cycles / hardware.frequency_hz
    return ModuleResult(
        module=workload.name,
        mapping="+".join(sorted(effective_mappings)) if effective_mappings else module_mapping,
        useful_macs=useful_macs,
        executed_macs=executed_macs,
        utilization=utilization,
        dram_read_bytes=dram_read_bytes,
        dram_write_bytes=dram_write_bytes,
        energy_pj=energy_pj,
        cycles=cycles,
        seconds=seconds,
        notes="; ".join(notes[:3]) if notes else workload.notes,
        actions=dict(sorted(actions.counts.items())),
    )


def selected_dense_ops(workload: ModuleWorkload, module_mapping: str):
    if workload.name == "correlation_lookup":
        return workload.dense_ops if module_mapping == "gemmini" else ()
    return workload.dense_ops


def selected_irregular_workloads(workload: ModuleWorkload, module_mapping: str) -> tuple[IrregularWorkload, ...]:
    if workload.name == "correlation_lookup":
        if module_mapping == "gemmini":
            return tuple(work for work in workload.irregular if work.name == "correlation_gemmini_staging")
        return tuple(work for work in workload.irregular if work.name == "local_correlation_lookup")
    return workload.irregular


def mapping_for_module(module_name: str, mapping: MappingParams) -> str:
    table = {
        "feature_context_encoder": mapping.encoder,
        "feature_pyramid_pooling": "cpu",
        "patch_extraction": mapping.patch_extraction,
        "update_geometry_elementwise": mapping.geometry,
        "correlation_lookup": mapping.correlation,
        "update_dense_linears": mapping.update_dense,
        "factor_heads": mapping.factor_head,
        "soft_aggregation_scatter": mapping.soft_aggregation,
        "bundle_adjustment": mapping.ba,
        "graph_management": mapping.graph_management,
    }
    return table.get(module_name, "cpu")


def render_table(headers: list[str], rows: Iterable[dict[str, str]]) -> str:
    rows = list(rows)
    if not headers:
        return ""
    widths = {
        header: max(len(header), *(len(str(row.get(header, ""))) for row in rows))
        for header in headers
    }

    def fmt(values: dict[str, str]) -> str:
        return "| " + " | ".join(str(values.get(header, "")).ljust(widths[header]) for header in headers) + " |"

    lines = [
        fmt({header: header for header in headers}),
        "| " + " | ".join("-" * widths[header] for header in headers) + " |",
    ]
    lines.extend(fmt(row) for row in rows)
    return "\n".join(lines)


def human_count(value: float) -> str:
    units = ("", "K", "M", "G", "T", "P")
    number = float(value)
    for unit in units:
        if abs(number) < 1000.0 or unit == units[-1]:
            return f"{number:.2f}{unit}" if unit else str(int(number))
        number /= 1000.0
    return str(value)


def human_bytes(value: float) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")
    number = float(value)
    for unit in units:
        if abs(number) < 1024.0 or unit == units[-1]:
            return f"{number:.2f} {unit}" if unit != "B" else f"{int(number)} B"
        number /= 1024.0
    return f"{value} B"


def human_seconds(value: float) -> str:
    abs_value = abs(value)
    if abs_value < 1e-6:
        return f"{value * 1e9:.2f} ns"
    if abs_value < 1e-3:
        return f"{value * 1e6:.2f} us"
    if abs_value < 1:
        return f"{value * 1e3:.2f} ms"
    return f"{value:.3f} s"


def human_energy_j(value: float) -> str:
    abs_value = abs(value)
    if abs_value < 1e-9:
        return f"{value * 1e12:.2f} pJ"
    if abs_value < 1e-6:
        return f"{value * 1e9:.2f} nJ"
    if abs_value < 1e-3:
        return f"{value * 1e6:.2f} uJ"
    if abs_value < 1:
        return f"{value * 1e3:.2f} mJ"
    return f"{value:.3f} J"
