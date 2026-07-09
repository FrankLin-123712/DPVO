"""Parameter schemas for the DPVO-on-Gemmini energy model.

The defaults are intentionally explicit and replaceable.  They provide a
first-order analytical model, not calibrated silicon numbers.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DPVO_CONFIG = REPO_ROOT / "config" / "default.yaml"
DEFAULT_GEMMINI_HEADER = (
    WORKSPACE_ROOT
    / "chipyard"
    / "generators"
    / "gemmini"
    / "software"
    / "libgemmini"
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
    nn_dtype_bytes: int = 2
    ba_dtype_bytes: int = 4
    ba_macs_per_edge: int = 900
    centroid_selection: str = "RANDOM"
    loop_closure: bool = False
    classic_loop_closure: bool = False

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
            return self.patches_per_frame * self.removal_window * (2 * self.patch_lifetime - 1)
        raise ValueError(f"unknown edge_mode: {self.edge_mode}")

    @property
    def new_edges_per_frame(self) -> int:
        return self.patches_per_frame * (2 * self.patch_lifetime - 1)

    @property
    def active_unique_patches(self) -> int:
        if self.unique_patches is not None:
            return self.unique_patches
        return min(self.active_edges, self.patches_per_frame * self.removal_window)

    @property
    def feature_height(self) -> int:
        return self.height // 4

    @property
    def feature_width(self) -> int:
        return self.width // 4


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
    dataflow: str = "BOTH"
    overlap_dma_compute: bool = True
    gemmini_min_utilization_for_offload: float = 0.10


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
    def fp32_default(cls) -> "HardwareParams":
        return cls(
            name="GemminiFP32DefaultConfig",
            dim=8,
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
            return cls.default_int8()

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
        if profile == "fp32-default":
            return cls.fp32_default()
        if profile == "custom":
            return cls()
        raise ValueError(f"unknown hardware profile: {profile}")


@dataclass(frozen=True)
class EnergyTable:
    """Unit dynamic energy values in picojoules per action unit."""

    unit_pj: dict[str, float]

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
            "cpu.mac": {
                "int8": 0.80,
                "int16": 1.60,
                "fp16": 3.00,
                "fp32": 8.00,
            }.get(hardware.input_precision, 4.00),
            "cpu.alu": 1.00,
            "cpu.branch": 0.25,
            "cpu.atomic": 40.00,
            "sync.rocc": 200.00,
            "dma.read_byte": 0.60,
            "dma.write_byte": 0.70,
            "spad.read_byte": 1.00,
            "spad.write_byte": 1.20,
            "acc.read_byte": 1.20,
            "acc.write_byte": 1.50,
            "pe.reg_read_byte": 0.20,
            "pe.reg_write_byte": 0.25,
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
            merged[str(key)] = float(value)
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
