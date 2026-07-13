"""Parameter schemas for the DPVO-on-Gemmini energy model.

The defaults are intentionally explicit and replaceable.  They provide a
first-order analytical model, not calibrated silicon numbers.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
IMPLEMENTATION_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DPVO_CONFIG = REPO_ROOT / "config" / "default.yaml"
DEFAULT_GEMMINI_HEADER = (
    IMPLEMENTATION_ROOT
    / "gemmini"
    / "software"
    / "gemmini-rocc-tests"
    / "include"
    / "gemmini_params.h"
)


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
    """Load the flat DPVO config YAML without requiring PyYAML."""
    data: dict[str, Any] = {}
    if not path.exists():
        return data

    for raw_line in path.read_text().splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key and value:
            data[key] = parse_scalar(value)
    return data


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def round_up(value: int, divisor: int) -> int:
    return ceil_div(value, divisor) * divisor


def dtype_bytes_from_c_type(c_type: str) -> int:
    c_type = c_type.strip()
    if c_type in {"float", "uint32_t", "int32_t"}:
        return 4
    if c_type in {"double", "uint64_t", "int64_t"}:
        return 8
    if c_type in {"uint16_t", "int16_t"}:
        return 2
    if c_type in {"uint8_t", "int8_t", "char", "signed char", "unsigned char"}:
        return 1
    raise ValueError(f"unsupported C type in Gemmini header: {c_type}")


def precision_name(dtype_bytes: int, floating: bool) -> str:
    if floating:
        return {2: "fp16", 4: "fp32", 8: "fp64"}.get(dtype_bytes, f"fp{8 * dtype_bytes}")
    return {1: "int8", 2: "int16", 4: "int32"}.get(dtype_bytes, f"int{8 * dtype_bytes}")


@dataclass(frozen=True)
class AlgorithmParams:
    """DPVO algorithm/design parameters, P_a."""

    height: int = 480
    width: int = 640
    patches_per_frame: int = 96
    removal_window: int = 22
    optimization_window: int = 10
    patch_lifetime: int = 13
    patch_size: int = 3
    corr_radius: int = 3
    corr_levels: int = 2
    update_iterations: int = 1
    ba_iterations: int = 2
    edge_mode: str = "steady"
    edges: int | None = None
    unique_patches: int | None = None
    unique_frame_pairs: int | None = None
    nn_dtype_bytes: int = 2
    ba_dtype_bytes: int = 4
    ba_macs_per_edge: int = 900
    centroid_selection: str = "RANDOM"
    loop_closure: bool = False
    classic_loop_closure: bool = False

    def __post_init__(self) -> None:
        positive = {
            "height": self.height,
            "width": self.width,
            "patches_per_frame": self.patches_per_frame,
            "removal_window": self.removal_window,
            "optimization_window": self.optimization_window,
            "patch_lifetime": self.patch_lifetime,
            "patch_size": self.patch_size,
            "corr_levels": self.corr_levels,
            "update_iterations": self.update_iterations,
            "ba_iterations": self.ba_iterations,
            "ba_macs_per_edge": self.ba_macs_per_edge,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.corr_radius < 0:
            raise ValueError(f"corr_radius must be non-negative, got {self.corr_radius}")
        if self.edge_mode not in {"steady", "new-frame"}:
            raise ValueError(f"unknown edge_mode: {self.edge_mode}")
        for name, value in {
            "edges": self.edges,
            "unique_patches": self.unique_patches,
            "unique_frame_pairs": self.unique_frame_pairs,
        }.items():
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when provided, got {value}")
        for name, value in {
            "unique_patches": self.unique_patches,
            "unique_frame_pairs": self.unique_frame_pairs,
        }.items():
            if value is not None and value > self.active_edges:
                raise ValueError(
                    f"{name} cannot exceed active edge count {self.active_edges}, got {value}"
                )
        if self.nn_dtype_bytes not in {1, 2, 4, 8}:
            raise ValueError(f"unsupported nn_dtype_bytes: {self.nn_dtype_bytes}")
        if self.ba_dtype_bytes not in {4, 8}:
            raise ValueError(f"unsupported ba_dtype_bytes: {self.ba_dtype_bytes}")

    @classmethod
    def from_config(
        cls,
        path: Path = DEFAULT_DPVO_CONFIG,
        **overrides: Any,
    ) -> "AlgorithmParams":
        cfg = load_simple_yaml(path)
        mixed_precision = bool(cfg.get("MIXED_PRECISION", True))
        params = cls(
            patches_per_frame=int(cfg.get("PATCHES_PER_FRAME", cls.patches_per_frame)),
            removal_window=int(cfg.get("REMOVAL_WINDOW", cls.removal_window)),
            optimization_window=int(cfg.get("OPTIMIZATION_WINDOW", cls.optimization_window)),
            patch_lifetime=int(cfg.get("PATCH_LIFETIME", cls.patch_lifetime)),
            nn_dtype_bytes=2 if mixed_precision else 4,
            centroid_selection=str(cfg.get("CENTROID_SEL_STRAT", cls.centroid_selection)),
            loop_closure=bool(cfg.get("LOOP_CLOSURE", cls.loop_closure)),
            classic_loop_closure=bool(cfg.get("CLASSIC_LOOP_CLOSURE", cls.classic_loop_closure)),
        )
        clean_overrides = {key: value for key, value in overrides.items() if value is not None}
        return replace(params, **clean_overrides)

    @property
    def active_edges(self) -> int:
        if self.edges is not None:
            return self.edges
        new_edges = self.new_edges_per_frame
        if self.edge_mode == "new-frame":
            return new_edges
        if self.edge_mode == "steady":
            # DPVO removes expired factors after the current frame update, so
            # the normal update sees ages 0..REMOVAL_WINDOW. Recent patches
            # have not accumulated all future edges, hence r+min(r-1, age).
            retained_edges_per_patch = sum(
                self.patch_lifetime + min(self.patch_lifetime - 1, age)
                for age in range(self.removal_window + 1)
            )
            # If r exceeds the removal horizon, __edges_forw() temporarily
            # reintroduces older source patches for their single edge to the
            # current target. Their earlier factors remain removed.
            reintroduced_edges_per_patch = max(
                0,
                self.patch_lifetime - self.removal_window - 1,
            )
            return self.patches_per_frame * (
                retained_edges_per_patch + reintroduced_edges_per_patch
            )
        raise ValueError(f"unknown edge_mode: {self.edge_mode}")

    @property
    def new_edges_per_frame(self) -> int:
        return self.patches_per_frame * (2 * self.patch_lifetime - 1)

    @property
    def active_source_frames(self) -> int:
        if self.edge_mode == "new-frame":
            # Newly appended forward factors can reintroduce patches older
            # than REMOVAL_WINDOW, up to PATCH_LIFETIME.
            return self.patch_lifetime
        return max(self.removal_window + 1, self.patch_lifetime)

    @property
    def active_unique_patches(self) -> int:
        if self.unique_patches is not None:
            return self.unique_patches
        return min(self.active_edges, self.patches_per_frame * self.active_source_frames)

    @property
    def active_unique_frame_pairs(self) -> int:
        if self.unique_frame_pairs is not None:
            return self.unique_frame_pairs
        # Forward/backward factors are created in blocks of roughly one
        # PATCHES_PER_FRAME-sized source/target frame pair.
        return max(1, ceil_div(self.active_edges, self.patches_per_frame))

    @property
    def feature_height(self) -> int:
        return ceil_div(self.height, 4)

    @property
    def feature_width(self) -> int:
        return ceil_div(self.width, 4)


@dataclass(frozen=True)
class MappingParams:
    """Implementation/mapping choices, M_k."""

    encoder: str = "gemmini"
    update_dense: str = "gemmini"
    correlation: str = "cpu"
    factor_head: str = "gemmini"
    patch_extraction: str = "cpu"
    soft_aggregation: str = "cpu"
    ba: str = "cpu"
    graph_management: str = "cpu"
    geometry: str = "cpu"
    dataflow: str = "WS"
    overlap_dma_compute: bool = True
    gemmini_min_utilization_for_offload: float = 0.10

    def __post_init__(self) -> None:
        for name in (
            "encoder",
            "update_dense",
            "correlation",
            "factor_head",
            "patch_extraction",
            "soft_aggregation",
            "ba",
            "graph_management",
            "geometry",
        ):
            value = getattr(self, name)
            if value not in {"cpu", "gemmini"}:
                raise ValueError(f"unsupported mapping {name}={value}")
        for name in (
            "correlation",
            "patch_extraction",
            "soft_aggregation",
            "ba",
            "graph_management",
            "geometry",
        ):
            if getattr(self, name) != "cpu":
                raise ValueError(
                    f"{name}=gemmini is not implemented by the current action model"
                )
        if self.dataflow not in {"WS", "OS"}:
            raise ValueError(f"dataflow must be WS or OS, got {self.dataflow}")
        if not 0.0 <= self.gemmini_min_utilization_for_offload <= 1.0:
            raise ValueError("gemmini_min_utilization_for_offload must be in [0, 1]")


@dataclass(frozen=True)
class HardwareParams:
    """Rocket + Gemmini hardware parameters, P_h."""

    name: str = "generated-gemmini-header"
    dim: int = 8
    input_bytes: int = 4
    acc_bytes: int = 4
    input_precision: str = "fp32"
    acc_precision: str = "fp32"
    sp_capacity_kib: float = 256.0
    acc_capacity_kib: float = 64.0
    sp_banks: int = 4
    acc_banks: int = 1
    dma_maxbytes: int = 64
    dma_buswidth_bits: int = 128
    frequency_hz: float = 1.0e9
    cpu_peak_macs_per_cycle: float = 4.0
    cpu_peak_alu_ops_per_cycle: float = 4.0
    l1_bandwidth_bytes_per_cycle: float = 32.0
    l2_bandwidth_bytes_per_cycle: float = 16.0
    dram_bandwidth_bytes_per_cycle: float = 8.0
    rocc_command_cycles: int = 80
    cpu_sync_cycles: int = 150
    random_l1_hit_rate: float = 0.35
    random_l2_hit_rate: float = 0.70
    sequential_l1_hit_rate: float = 0.90
    sequential_l2_hit_rate: float = 0.95
    write_l1_hit_rate: float = 0.80
    write_l2_hit_rate: float = 0.90
    cache_line_bytes: int = 64

    def __post_init__(self) -> None:
        positive = {
            "dim": self.dim,
            "input_bytes": self.input_bytes,
            "acc_bytes": self.acc_bytes,
            "sp_capacity_kib": self.sp_capacity_kib,
            "acc_capacity_kib": self.acc_capacity_kib,
            "sp_banks": self.sp_banks,
            "acc_banks": self.acc_banks,
            "dma_maxbytes": self.dma_maxbytes,
            "dma_buswidth_bits": self.dma_buswidth_bits,
            "frequency_hz": self.frequency_hz,
            "cpu_peak_macs_per_cycle": self.cpu_peak_macs_per_cycle,
            "cpu_peak_alu_ops_per_cycle": self.cpu_peak_alu_ops_per_cycle,
            "l1_bandwidth_bytes_per_cycle": self.l1_bandwidth_bytes_per_cycle,
            "l2_bandwidth_bytes_per_cycle": self.l2_bandwidth_bytes_per_cycle,
            "dram_bandwidth_bytes_per_cycle": self.dram_bandwidth_bytes_per_cycle,
            "cache_line_bytes": self.cache_line_bytes,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        precision_bytes = {
            "int8": 1,
            "int16": 2,
            "int32": 4,
            "int64": 8,
            "fp16": 2,
            "bf16": 2,
            "fp32": 4,
            "fp64": 8,
        }
        for prefix, precision, storage_bytes in (
            ("input", self.input_precision, self.input_bytes),
            ("acc", self.acc_precision, self.acc_bytes),
        ):
            expected_bytes = precision_bytes.get(precision)
            if expected_bytes is None:
                raise ValueError(f"unsupported {prefix}_precision: {precision}")
            if expected_bytes != storage_bytes:
                raise ValueError(
                    f"{prefix}_precision={precision} requires {expected_bytes} bytes, "
                    f"got {storage_bytes}"
                )
        for name in (
            "random_l1_hit_rate",
            "random_l2_hit_rate",
            "sequential_l1_hit_rate",
            "sequential_l2_hit_rate",
            "write_l1_hit_rate",
            "write_l2_hit_rate",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")

    @property
    def pe_count(self) -> int:
        return self.dim * self.dim

    @property
    def sp_capacity_bytes(self) -> int:
        return int(self.sp_capacity_kib * 1024)

    @property
    def acc_capacity_bytes(self) -> int:
        return int(self.acc_capacity_kib * 1024)

    @property
    def dma_bandwidth_bytes_per_cycle(self) -> float:
        return max(1.0, self.dma_buswidth_bits / 8)

    @classmethod
    def default_int8(cls) -> "HardwareParams":
        return cls(
            name="GemminiConfigs.defaultConfig",
            dim=16,
            input_bytes=1,
            acc_bytes=4,
            input_precision="int8",
            acc_precision="int32",
            sp_capacity_kib=256.0,
            acc_capacity_kib=64.0,
            sp_banks=4,
            acc_banks=2,
            dma_maxbytes=64,
            dma_buswidth_bits=128,
        )

    @classmethod
    def fp16_default(cls) -> "HardwareParams":
        return cls(
            name="GemminiFPConfigs.FP16DefaultConfig",
            dim=4,
            input_bytes=2,
            acc_bytes=4,
            input_precision="fp16",
            acc_precision="fp32",
            sp_capacity_kib=256.0,
            acc_capacity_kib=64.0,
            sp_banks=4,
            acc_banks=1,
            dma_maxbytes=64,
            dma_buswidth_bits=128,
        )

    @classmethod
    def fp32_default(cls) -> "HardwareParams":
        return cls(
            name="GemminiFPConfigs.FP32DefaultConfig",
            dim=4,
            input_bytes=4,
            acc_bytes=4,
            input_precision="fp32",
            acc_precision="fp32",
            sp_capacity_kib=256.0,
            acc_capacity_kib=64.0,
            sp_banks=4,
            acc_banks=1,
            dma_maxbytes=64,
            dma_buswidth_bits=128,
        )

    @classmethod
    def from_gemmini_header(cls, path: Path = DEFAULT_GEMMINI_HEADER) -> "HardwareParams":
        if not path.exists():
            raise FileNotFoundError(
                f"generated Gemmini header not found: {path}. "
                "Choose an explicit hardware profile or pass --gemmini-header."
            )

        text = path.read_text()

        def macro_int(name: str, default: int) -> int:
            match = re.search(rf"^\s*#define\s+{re.escape(name)}\s+([0-9]+)\b", text, re.MULTILINE)
            return int(match.group(1)) if match else default

        dim = macro_int("DIM", 16)
        sp_banks = macro_int("BANK_NUM", 4)
        bank_rows = macro_int("BANK_ROWS", 0)
        acc_rows = macro_int("ACC_ROWS", 0)
        dma_maxbytes = macro_int("MAX_BYTES", 64)

        elem_match = re.search(r"^\s*typedef\s+(.+?)\s+elem_t\s*;", text, re.MULTILINE)
        acc_match = re.search(r"^\s*typedef\s+(.+?)\s+acc_t\s*;", text, re.MULTILINE)
        elem_type = elem_match.group(1).strip() if elem_match else "int8_t"
        acc_type = acc_match.group(1).strip() if acc_match else "int32_t"
        input_bytes = dtype_bytes_from_c_type(elem_type)
        acc_bytes = dtype_bytes_from_c_type(acc_type)
        input_floating = elem_type in {"float", "double"} or "ELEM_T_IS_FLOAT" in text
        acc_floating = acc_type in {"float", "double"} or "ACC_T_EXP_BITS" in text

        sp_capacity_kib = (
            sp_banks * bank_rows * dim * input_bytes / 1024
            if bank_rows
            else 256.0
        )
        acc_capacity_kib = (
            acc_rows * dim * acc_bytes / 1024
            if acc_rows
            else 64.0
        )

        return cls(
            name=f"generated:{path.name}",
            dim=dim,
            input_bytes=input_bytes,
            acc_bytes=acc_bytes,
            input_precision=precision_name(input_bytes, input_floating),
            acc_precision=precision_name(acc_bytes, acc_floating),
            sp_capacity_kib=sp_capacity_kib,
            acc_capacity_kib=acc_capacity_kib,
            sp_banks=sp_banks,
            acc_banks=1,
            dma_maxbytes=dma_maxbytes,
            dma_buswidth_bits=128,
        )

    @classmethod
    def from_profile(
        cls,
        profile: str,
        header_path: Path = DEFAULT_GEMMINI_HEADER,
    ) -> "HardwareParams":
        if profile == "generated-header":
            return cls.from_gemmini_header(header_path)
        if profile == "default-int8":
            return cls.default_int8()
        if profile == "fp16-default":
            return cls.fp16_default()
        if profile == "fp32-default":
            return cls.fp32_default()
        if profile == "custom":
            return cls()
        raise ValueError(f"unknown hardware profile: {profile}")


@dataclass(frozen=True)
class EnergyTable:
    """Unit dynamic energy values in picojoules per action unit."""

    unit_pj: dict[str, float]

    def __post_init__(self) -> None:
        if not self.unit_pj:
            raise ValueError("energy table must not be empty")
        for name, value in self.unit_pj.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    f"unit energy must be finite and non-negative: {name}={value}"
                )

    @classmethod
    def defaults(cls, hardware: HardwareParams) -> "EnergyTable":
        # Published exact values vary widely by process and SRAM compiler. These
        # defaults preserve the expected ordering: DRAM >> L2/L1/SRAM > compute.
        table = {
            "gemmini.mac": {
                "int8": 0.20,
                "int16": 0.70,
                "int32": 3.20,
                "fp16": 1.40,
                "bf16": 1.60,
                "fp32": 4.60,
                "fp64": 18.00,
            }.get(hardware.input_precision, 1.40),
            "cpu.mac.int8": 0.80,
            "cpu.mac.int16": 1.60,
            "cpu.mac.fp16": 3.00,
            "cpu.mac.fp32": 8.00,
            "cpu.mac.fp64": 32.00,
            "cpu.alu": 1.00,
            "cpu.branch": 0.25,
            "cpu.atomic": 40.00,
            "sync.rocc": 200.00,
            "dma.read_byte": 0.60,
            "dma.write_byte": 0.70,
            "dma.transaction": 20.00,
            "spad.read_byte": 1.00,
            "spad.write_byte": 1.20,
            "acc.read_byte": 1.20,
            "acc.write_byte": 1.50,
            "l1.sequential_read_byte": 1.10,
            "l1.random_read_byte": 1.40,
            "l1.write_byte": 1.50,
            "l2.sequential_read_byte": 5.00,
            "l2.random_read_byte": 7.00,
            "l2.write_byte": 7.50,
            "dram.sequential_read_byte": 80.00,
            "dram.random_read_byte": 140.00,
            "dram.write_byte": 120.00,
            "metadata.read_byte": 2.00,
            "control.op": 1.00,
        }
        return cls(table)

    @classmethod
    def from_json(cls, path: Path, base: "EnergyTable") -> "EnergyTable":
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            raise ValueError("energy table JSON must be an object mapping action names to pJ values")
        merged = dict(base.unit_pj)
        for key, value in data.items():
            key = str(key)
            value = float(value)
            if key == "cpu.mac":
                # Backward-compatible override for older tables. New tables
                # should provide precision-specific CPU MAC actions.
                for cpu_key in tuple(name for name in merged if name.startswith("cpu.mac.")):
                    merged[cpu_key] = value
            else:
                merged[key] = value
        return cls(merged)

    def energy_pj(self, action_counts: dict[str, float]) -> float:
        missing = [name for name in action_counts if name not in self.unit_pj]
        if missing:
            raise KeyError(f"missing unit energy for actions: {', '.join(sorted(missing))}")
        return sum(action_counts[name] * self.unit_pj[name] for name in action_counts)

    def as_dict(self) -> dict[str, float]:
        return dict(self.unit_pj)


def dataclass_dict(value: Any) -> dict[str, Any]:
    return asdict(value)
