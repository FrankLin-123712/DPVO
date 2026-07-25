#!/usr/bin/env python3
"""Run reproducible one-at-a-time DPVO algorithm-parameter sweeps.

The workflow has four outputs under statistic_result/:

1. generated_configs/: one YAML config per candidate;
2. per_module/: exact per-module estimator JSON/CSV for every candidate;
3. sweep_summary.csv and module_summary.csv;
4. sweep_plots/*.svg plus one_at_a_time_sweep_report.md.

ATE evaluation is optional because it requires the full EuRoC image dataset,
CUDA-enabled DPVO dependencies, and the compiled DPVO extensions.  Use
--run-eval when that environment is ready.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path
from typing import Any, Iterable, Sequence

import statistic_dpvo as statistic

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = None


TOOLS_DIR = Path(__file__).resolve().parent
DPVO_ROOT = TOOLS_DIR.parents[1]
DEFAULT_RESULT_DIR = TOOLS_DIR / "statistic_result"

EUROC_SCENES = [
    "MH_01_easy",
    "MH_02_easy",
    "MH_03_medium",
    "MH_04_difficult",
    "MH_05_difficult",
    "V1_01_easy",
    "V1_02_medium",
    "V1_03_difficult",
    "V2_01_easy",
    "V2_02_medium",
    "V2_03_difficult",
]

PARAMETER_ORDER = [
    "PATCHES_PER_FRAME",
    "PATCH_LIFETIME",
    "REMOVAL_WINDOW",
    "OPTIMIZATION_WINDOW",
    "BA_ITERATIONS",
    "IMAGE_SIZE",
]

PARAMETER_LABELS = {
    "PATCHES_PER_FRAME": "PATCHES_PER_FRAME",
    "PATCH_LIFETIME": "PATCH_LIFETIME",
    "REMOVAL_WINDOW": "REMOVAL_WINDOW",
    "OPTIMIZATION_WINDOW": "OPTIMIZATION_WINDOW",
    "BA_ITERATIONS": "BA_ITERATIONS",
    "IMAGE_SIZE": "{H,W}",
}

DEFAULT_HEIGHT = 480
DEFAULT_WIDTH = 640


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    sweep_parameter: str
    sweep_value: int | tuple[int, int]
    sweep_value_label: str
    height: int
    width: int
    config_path: Path
    overrides: dict[str, int]


@dataclass
class StatsResult:
    candidate: Candidate
    pa: statistic.PaConfig
    graph: statistic.GraphStats
    module_rows: list[statistic.LayerRow]
    total: statistic.LayerRow
    average_ate_m: float | None = None
    evaluation_status: str = "not_run"
    evaluation_json: Path | None = None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DPVO_ROOT / "config" / "default.yaml")
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--network", type=Path, default=DPVO_ROOT / "dpvo.pth")
    parser.add_argument("--eurocdir", type=Path, default=DPVO_ROOT / "datasets" / "EUROC")
    parser.add_argument("--active-frames", type=int, default=statistic.DEFAULT_ACTIVE_FRAMES)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--backend-thresh", type=float, default=64.0)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--run-eval", action="store_true")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm/progress output in sweep and ATE subprocesses.",
    )
    parser.add_argument(
        "--reuse-eval",
        action="store_true",
        help="Reuse existing per-candidate evaluation JSON when present.",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        choices=EUROC_SCENES,
        default=EUROC_SCENES,
        help="EuRoC scenes for ATE evaluation. Default: all 11 Table-2 scenes.",
    )
    parser.add_argument(
        "--parameters",
        nargs="+",
        choices=PARAMETER_ORDER,
        default=PARAMETER_ORDER,
        help="Subset of one-at-a-time sweeps to run.",
    )
    parser.add_argument(
        "--sweep-file",
        type=Path,
        help=(
            "Optional JSON mapping parameter names to value lists. IMAGE_SIZE "
            "values must be [height, width] pairs."
        ),
    )
    parser.add_argument("--no-keyframe", action="store_true")
    return parser.parse_args(argv)


def yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    return str(value)


def read_base_values(config_path: Path) -> dict[str, Any]:
    values = statistic.load_simple_yaml(config_path)
    if not values:
        raise ValueError(f"no top-level values found in {config_path}")
    return values


def write_candidate_config(base_config: Path, output: Path, overrides: dict[str, int]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    lines: list[str] = []
    for raw_line in base_config.read_text().splitlines():
        stripped = raw_line.split("#", 1)[0].strip()
        if ":" not in stripped:
            lines.append(raw_line)
            continue
        key = stripped.split(":", 1)[0].strip().upper()
        if key in overrides:
            seen.add(key)
            suffix = ""
            if "#" in raw_line:
                suffix = "  #" + raw_line.split("#", 1)[1]
            lines.append(f"{key}: {yaml_scalar(overrides[key])}{suffix}")
        else:
            lines.append(raw_line)
    for key, value in overrides.items():
        if key not in seen:
            lines.append(f"{key}: {yaml_scalar(value)}")
    output.write_text("\n".join(lines).rstrip() + "\n")


def unique_decreasing(values: Iterable[int]) -> list[int]:
    result: list[int] = []
    for value in values:
        value = int(value)
        if value > 0 and value not in result:
            result.append(value)
    return result


def default_sweeps(base: dict[str, Any]) -> dict[str, list[int | tuple[int, int]]]:
    patches = int(base.get("PATCHES_PER_FRAME", 96))
    lifetime = int(base.get("PATCH_LIFETIME", 13))
    removal = int(base.get("REMOVAL_WINDOW", 22))
    optimization = int(base.get("OPTIMIZATION_WINDOW", 10))
    return {
        "PATCHES_PER_FRAME": unique_decreasing([patches, 80, 64, 48, 32]),
        "PATCH_LIFETIME": unique_decreasing([lifetime, 11, 9, 7, 5]),
        "REMOVAL_WINDOW": unique_decreasing([removal, 18, 14, 10]),
        "OPTIMIZATION_WINDOW": unique_decreasing([optimization, 8, 6, 4]),
        "BA_ITERATIONS": list(range(20, 1, -2)),
        "IMAGE_SIZE": [(480, 640), (384, 512), (320, 416), (240, 320), (192, 256)],
    }


def load_sweeps(base: dict[str, Any], sweep_file: Path | None) -> dict[str, list[int | tuple[int, int]]]:
    if sweep_file is None:
        return default_sweeps(base)
    payload = json.loads(sweep_file.read_text())
    sweeps: dict[str, list[int | tuple[int, int]]] = {}
    for parameter, values in payload.items():
        normalized = parameter.upper()
        if normalized not in PARAMETER_ORDER:
            raise ValueError(f"unsupported sweep parameter {parameter!r}")
        if normalized == "IMAGE_SIZE":
            parsed_pairs: list[tuple[int, int]] = []
            for item in values:
                if not isinstance(item, list | tuple) or len(item) != 2:
                    raise ValueError("IMAGE_SIZE values must be [height, width] pairs")
                parsed_pairs.append((int(item[0]), int(item[1])))
            sweeps[normalized] = parsed_pairs
        else:
            sweeps[normalized] = unique_decreasing(int(item) for item in values)
    return sweeps


def value_label(value: int | tuple[int, int]) -> str:
    if isinstance(value, tuple):
        return f"{value[0]}x{value[1]}"
    return str(value)


def candidate_id(parameter: str, value: int | tuple[int, int]) -> str:
    if isinstance(value, tuple):
        return f"{parameter.lower()}_{value[0]}x{value[1]}"
    return f"{parameter.lower()}_{int(value):03d}"


def generate_candidates(args: argparse.Namespace, sweeps: dict[str, list[int | tuple[int, int]]]) -> list[Candidate]:
    candidates: list[Candidate] = []
    config_dir = args.result_dir / "generated_configs"
    for parameter in PARAMETER_ORDER:
        if parameter not in args.parameters:
            continue
        for value in sweeps.get(parameter, []):
            height, width = DEFAULT_HEIGHT, DEFAULT_WIDTH
            overrides: dict[str, int] = {}
            if parameter == "IMAGE_SIZE":
                assert isinstance(value, tuple)
                height, width = value
            else:
                assert isinstance(value, int)
                overrides[parameter] = value
            cid = candidate_id(parameter, value)
            config_path = config_dir / f"{cid}.yaml"
            write_candidate_config(args.config, config_path, overrides)
            candidates.append(
                Candidate(
                    candidate_id=cid,
                    sweep_parameter=parameter,
                    sweep_value=value,
                    sweep_value_label=value_label(value),
                    height=height,
                    width=width,
                    config_path=config_path,
                    overrides=overrides,
                )
            )
    return candidates


def write_csv(path: Path, rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def row_prefix(candidate: Candidate) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "sweep_parameter": candidate.sweep_parameter,
        "sweep_value": candidate.sweep_value_label,
        "height": candidate.height,
        "width": candidate.width,
    }


def exact_module_row(candidate: Candidate, row: statistic.LayerRow) -> dict[str, Any]:
    return {
        **row_prefix(candidate),
        "module": row.module,
        "fp16_ops": row.fp16_ops,
        "fp32_ops": row.fp32_ops,
        "fp64_ops": row.fp64_ops,
        "int_bool_ops": row.int_bool_ops,
        "total_ops": row.total_ops,
        "mem_read_bytes": row.mem_read_bytes,
        "mem_write_bytes": row.mem_write_bytes,
        "total_memory_bytes": row.total_memory_bytes,
        "operation_intensity_ops_per_byte": row.operation_intensity,
        "access_pattern": row.access_pattern,
    }


def collect_statistics(args: argparse.Namespace, candidate: Candidate) -> StatsResult:
    statistic_args = [
        "--config",
        str(candidate.config_path),
        "--height",
        str(candidate.height),
        "--width",
        str(candidate.width),
        "--active-frames",
        str(args.active_frames),
        "--per-module",
        "--format",
        "json",
    ]
    if args.no_keyframe:
        statistic_args.append("--no-keyframe")
    parsed = statistic.parse_args(statistic_args)
    pa, graph, layer_rows = statistic.build_rows(parsed)
    module_rows = statistic.aggregate_modules(layer_rows)
    total = statistic.sum_rows(layer_rows)

    output_dir = args.result_dir / "per_module"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "candidate": {
            **row_prefix(candidate),
            "config_path": str(candidate.config_path),
            "overrides": candidate.overrides,
        },
        "pa": asdict(pa),
        "workload": {
            "height": candidate.height,
            "width": candidate.width,
            "active_frames": graph.frame_count,
            "keyframe_test_included": not args.no_keyframe,
            "output_granularity": "per-module",
        },
        "derived": {key: value for key, value in asdict(graph).items() if key != "pairs"},
        "rows": [row.raw_dict() for row in module_rows],
        "total": total.raw_dict(),
    }
    (output_dir / f"{candidate.candidate_id}.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n"
    )
    write_csv(
        output_dir / f"{candidate.candidate_id}.csv",
        [exact_module_row(candidate, row) for row in module_rows],
        MODULE_COLUMNS,
    )
    return StatsResult(candidate=candidate, pa=pa, graph=graph, module_rows=module_rows, total=total)


MODULE_COLUMNS = [
    "candidate_id",
    "sweep_parameter",
    "sweep_value",
    "height",
    "width",
    "module",
    "fp16_ops",
    "fp32_ops",
    "fp64_ops",
    "int_bool_ops",
    "total_ops",
    "mem_read_bytes",
    "mem_write_bytes",
    "total_memory_bytes",
    "operation_intensity_ops_per_byte",
    "access_pattern",
]

SUMMARY_COLUMNS = [
    "candidate_id",
    "sweep_parameter",
    "sweep_value",
    "height",
    "width",
    "patches_per_frame",
    "patch_lifetime",
    "removal_window",
    "optimization_window",
    "ba_iterations",
    "edge_count",
    "unique_patches",
    "edge_groups",
    "free_pose_count",
    "total_ops",
    "total_memory_bytes",
    "ate_m",
    "evaluation_status",
    "config_path",
    "evaluation_json",
]

SCENE_COLUMNS = [
    "candidate_id",
    "sweep_parameter",
    "sweep_value",
    "scene",
    "trial_count",
    "median_ate_m",
    "mean_ate_m",
    "trial_ate_m",
]


def preflight_evaluation(args: argparse.Namespace) -> list[str]:
    errors: list[str] = []
    if not args.network.exists():
        errors.append(f"missing network checkpoint: {args.network}")
    missing_scenes = []
    for scene in args.scenes:
        image_dir = args.eurocdir / scene / "mav0" / "cam0" / "data"
        if not any(image_dir.glob("*.png")):
            missing_scenes.append(scene)
    if missing_scenes:
        errors.append(
            "missing EuRoC image data under "
            f"{args.eurocdir}: {', '.join(missing_scenes)}"
        )

    dependency_check = (
        "import cv2, evo, torch; "
        "import dpvo.fastba, dpvo.altcorr; "
        "print('ok')"
    )
    completed = subprocess.run(
        [str(args.python), "-c", dependency_check],
        cwd=DPVO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        first_line = completed.stderr.strip().splitlines()[0] if completed.stderr.strip() else "unknown error"
        errors.append(f"DPVO Python/CUDA dependencies are not ready: {first_line}")
    return errors


def run_evaluation(args: argparse.Namespace, result: StatsResult) -> None:
    output_json = args.result_dir / "euroc_eval" / f"{result.candidate.candidate_id}.json"
    scene_csv = args.result_dir / "euroc_eval" / f"{result.candidate.candidate_id}_scenes.csv"
    if args.reuse_eval and output_json.exists():
        apply_eval_json(result, output_json)
        return

    command = [
        str(args.python),
        str(TOOLS_DIR / "evaluate_euroc_sweep.py"),
        "--network",
        str(args.network),
        "--config",
        str(result.candidate.config_path),
        "--eurocdir",
        str(args.eurocdir),
        "--output",
        str(output_json),
        "--per-scene-csv",
        str(scene_csv),
        "--height",
        str(result.candidate.height),
        "--width",
        str(result.candidate.width),
        "--stride",
        str(args.stride),
        "--trials",
        str(args.trials),
        "--seed",
        str(args.seed),
        "--backend-thresh",
        str(args.backend_thresh),
        "--scenes",
        *args.scenes,
    ]
    if args.no_progress:
        command.append("--no-progress")
    completed = subprocess.run(command, cwd=DPVO_ROOT)
    result.evaluation_json = output_json
    if completed.returncode != 0:
        result.evaluation_status = f"failed:{completed.returncode}"
        return
    apply_eval_json(result, output_json)


def apply_eval_json(result: StatsResult, output_json: Path) -> None:
    payload = json.loads(output_json.read_text())
    result.average_ate_m = float(payload["avg_ate_m"])
    result.evaluation_status = "ok"
    result.evaluation_json = output_json


def summary_row(result: StatsResult) -> dict[str, Any]:
    return {
        **row_prefix(result.candidate),
        "patches_per_frame": result.pa.patches_per_frame,
        "patch_lifetime": result.pa.patch_lifetime,
        "removal_window": result.pa.removal_window,
        "optimization_window": result.pa.optimization_window,
        "ba_iterations": result.pa.ba_iterations,
        "edge_count": result.graph.edge_count,
        "unique_patches": result.graph.unique_patches,
        "edge_groups": result.graph.edge_groups,
        "free_pose_count": result.graph.free_pose_count,
        "total_ops": result.total.total_ops,
        "total_memory_bytes": result.total.total_memory_bytes,
        "ate_m": "" if result.average_ate_m is None else result.average_ate_m,
        "evaluation_status": result.evaluation_status,
        "config_path": str(result.candidate.config_path),
        "evaluation_json": "" if result.evaluation_json is None else str(result.evaluation_json),
    }


def scene_rows(result: StatsResult) -> list[dict[str, Any]]:
    if result.evaluation_json is None or not result.evaluation_json.exists():
        return []
    payload = json.loads(result.evaluation_json.read_text())
    rows = []
    for scene in payload.get("scenes", []):
        rows.append(
            {
                **row_prefix(result.candidate),
                "scene": scene["scene"],
                "trial_count": scene["trial_count"],
                "median_ate_m": scene["median_ate_m"],
                "mean_ate_m": scene["mean_ate_m"],
                "trial_ate_m": json.dumps(scene["trial_ate_m"], allow_nan=False),
            }
        )
    return rows


def finite_values(values: Iterable[float | int | None]) -> list[float]:
    return [float(value) for value in values if value not in ("", None) and math.isfinite(float(value))]


def metric_range(values: Iterable[float | int | None]) -> tuple[float, float]:
    finite = finite_values(values)
    if not finite:
        return 0.0, 1.0
    low, high = min(finite), max(finite)
    if low == high:
        pad = max(abs(low) * 0.05, 1.0)
        return low - pad, high + pad
    pad = (high - low) * 0.08
    return low - pad, high + pad


def y_at(value: float, low: float, high: float, top: float, bottom: float) -> float:
    return bottom - (value - low) * (bottom - top) / (high - low)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def x_at(index: int, count: int, left: float, right: float) -> float:
    if count <= 1:
        return (left + right) / 2
    return left + index * (right - left) / (count - 1)


def svg_polyline(
    values: Sequence[float | int | None],
    low: float,
    high: float,
    left: float,
    right: float,
    top: float,
    bottom: float,
    color: str,
    *,
    y_offset: float = 0.0,
    dash: str = "",
    opacity: float = 1.0,
) -> str:
    points = []
    count = len(values)
    for index, value in enumerate(values):
        if value in ("", None):
            continue
        y = clamp(y_at(float(value), low, high, top, bottom) + y_offset, top, bottom)
        points.append((x_at(index, count, left, right), y))
    if not points:
        return ""
    point_text = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    opacity_attr = f' opacity="{opacity:.2f}"' if opacity < 1.0 else ""
    circles = "\n".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}" stroke="white" stroke-width="1.5"{opacity_attr}/>'
        for x, y in points
    )
    return (
        f'<polyline points="{point_text}" fill="none" stroke="{color}" '
        f'stroke-width="2.6" stroke-linejoin="round"{dash_attr}{opacity_attr}/>\n{circles}'
    )


def svg_pending_line(
    label_count: int,
    left: float,
    right: float,
    y: float,
    color: str,
) -> str:
    if label_count <= 0:
        return ""
    points = " ".join(f"{x_at(index, label_count, left, right):.1f},{y:.1f}" for index in range(label_count))
    return (
        f'<polyline points="{points}" fill="none" stroke="{color}" '
        'stroke-width="2.2" stroke-dasharray="7 6" opacity="0.55"/>'
    )


def si(value: float) -> str:
    units = ("", "K", "M", "G", "T", "P")
    scaled = float(value)
    unit = 0
    while abs(scaled) >= 1000.0 and unit < len(units) - 1:
        scaled /= 1000.0
        unit += 1
    if unit == 0:
        return f"{scaled:.0f}"
    return f"{scaled:.2f}".rstrip("0").rstrip(".") + units[unit]


def ticks(low: float, high: float, count: int = 5) -> list[float]:
    if count <= 1:
        return [low]
    return [low + (high - low) * index / (count - 1) for index in range(count)]


def axis_svg(
    x: float,
    low: float,
    high: float,
    top: float,
    bottom: float,
    color: str,
    side: str,
    formatter: str,
) -> str:
    anchor = "end" if side == "left" else "start"
    tick_x2 = x - 6 if side == "left" else x + 6
    text_x = x - 10 if side == "left" else x + 10
    parts = [f'<line x1="{x}" y1="{top}" x2="{x}" y2="{bottom}" stroke="{color}" stroke-width="1.5"/>']
    for value in ticks(low, high):
        y = y_at(value, low, high, top, bottom)
        label = si(value) if formatter == "si" else f"{value:.3f}"
        parts.append(f'<line x1="{x}" y1="{y:.1f}" x2="{tick_x2}" y2="{y:.1f}" stroke="{color}" />')
        parts.append(
            f'<text x="{text_x}" y="{y + 4:.1f}" text-anchor="{anchor}" '
            f'font-size="12" fill="{color}">{escape(label)}</text>'
        )
    return "\n".join(parts)


def plot_axis_ranges(summary_rows: Sequence[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    return {
        "ops": metric_range(int(row["total_ops"]) for row in summary_rows),
        "mem": metric_range(int(row["total_memory_bytes"]) for row in summary_rows),
        "ate": metric_range(None if row["ate_m"] == "" else float(row["ate_m"]) for row in summary_rows),
    }


def write_svg_plot(
    path: Path,
    parameter: str,
    rows: Sequence[dict[str, Any]],
    axis_ranges: dict[str, tuple[float, float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1080, 620
    left, right = 105.0, 870.0
    top, bottom = 72.0, 505.0
    mem_axis_x = 910.0
    ate_axis_x = 1000.0
    ops_color = "#1b6ca8"
    mem_color = "#b45f06"
    ate_color = "#2e7d32"
    labels = [str(row["sweep_value"]) for row in rows]
    ops = [int(row["total_ops"]) for row in rows]
    mem = [int(row["total_memory_bytes"]) for row in rows]
    ate = [None if row["ate_m"] == "" else float(row["ate_m"]) for row in rows]
    ops_low, ops_high = axis_ranges["ops"]
    mem_low, mem_high = axis_ranges["mem"]
    ate_low, ate_high = axis_ranges["ate"]
    title = f"One-at-a-Time Sweep: {PARAMETER_LABELS[parameter]}"

    grid_lines = []
    for value in ticks(ops_low, ops_high):
        y = y_at(value, ops_low, ops_high, top, bottom)
        grid_lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="#e5e7eb"/>')

    x_labels = []
    for index, label in enumerate(labels):
        x = x_at(index, len(labels), left, right)
        x_labels.append(f'<line x1="{x:.1f}" y1="{bottom}" x2="{x:.1f}" y2="{bottom + 6}" stroke="#6b7280"/>')
        x_labels.append(
            f'<text x="{x:.1f}" y="{bottom + 26}" text-anchor="middle" '
            f'font-size="13" fill="#111827">{escape(label)}</text>'
        )

    ate_note = ""
    if not finite_values(ate):
        ate_note = (
            f'<text x="{(left + right) / 2:.1f}" y="{top + 28}" text-anchor="middle" '
            f'font-size="13" fill="{ate_color}">ATE pending: run with --run-eval</text>\n'
            f'{svg_pending_line(len(labels), left, right, top + 54, ate_color)}'
        )

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width / 2:.1f}" y="34" text-anchor="middle" font-size="22" font-family="Arial, sans-serif" fill="#111827">{escape(title)}</text>
<text x="{width / 2:.1f}" y="57" text-anchor="middle" font-size="12" font-family="Arial, sans-serif" fill="#4b5563">shared y-axis scales across all sweep plots</text>
<g font-family="Arial, sans-serif">
{chr(10).join(grid_lines)}
<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#111827" stroke-width="1.5"/>
{axis_svg(left, ops_low, ops_high, top, bottom, ops_color, "left", "si")}
{axis_svg(mem_axis_x, mem_low, mem_high, top, bottom, mem_color, "right", "si")}
{axis_svg(ate_axis_x, ate_low, ate_high, top, bottom, ate_color, "right", "ate")}
{chr(10).join(x_labels)}
<text x="{(left + right) / 2:.1f}" y="{height - 28}" text-anchor="middle" font-size="14" fill="#111827">{escape(PARAMETER_LABELS[parameter])}</text>
<text x="24" y="{(top + bottom) / 2:.1f}" transform="rotate(-90 24 {(top + bottom) / 2:.1f})" text-anchor="middle" font-size="14" fill="{ops_color}">total ops (count)</text>
<text x="956" y="{(top + bottom) / 2:.1f}" transform="rotate(90 956 {(top + bottom) / 2:.1f})" text-anchor="middle" font-size="14" fill="{mem_color}">total mem (bytes)</text>
<text x="1046" y="{(top + bottom) / 2:.1f}" transform="rotate(90 1046 {(top + bottom) / 2:.1f})" text-anchor="middle" font-size="14" fill="{ate_color}">ATE (m)</text>
<g>
<rect x="{left}" y="48" width="14" height="4" fill="{ops_color}"/><text x="{left + 20}" y="54" font-size="13" fill="#111827">total ops</text>
<rect x="{left + 125}" y="48" width="14" height="4" fill="{mem_color}"/><text x="{left + 145}" y="54" font-size="13" fill="#111827">total mem</text>
<rect x="{left + 250}" y="48" width="14" height="4" fill="{ate_color}"/><text x="{left + 270}" y="54" font-size="13" fill="#111827">ATE</text>
</g>
{svg_polyline(mem, mem_low, mem_high, left, right, top, bottom, mem_color, dash="5 4", opacity=0.82)}
{svg_polyline(ops, ops_low, ops_high, left, right, top, bottom, ops_color)}
{svg_polyline(ate, ate_low, ate_high, left, right, top, bottom, ate_color)}
{ate_note}
</g>
</svg>
'''
    path.write_text(svg)


def write_plots(args: argparse.Namespace, summary_rows: Sequence[dict[str, Any]]) -> None:
    rows_by_parameter: dict[str, list[dict[str, Any]]] = {parameter: [] for parameter in PARAMETER_ORDER}
    for row in summary_rows:
        rows_by_parameter[row["sweep_parameter"]].append(row)
    axis_ranges = plot_axis_ranges(summary_rows)
    for parameter, rows in rows_by_parameter.items():
        if not rows:
            continue
        write_svg_plot(
            args.result_dir / "sweep_plots" / f"{parameter.lower()}_sweep.svg",
            parameter,
            rows,
            axis_ranges,
        )


def percent_reduction(default: float, value: float) -> float:
    if default == 0:
        return 0.0
    return 100.0 * (default - value) / default


def report_table(summary_rows: Sequence[dict[str, Any]]) -> str:
    lines = [
        "| parameter | 最低 candidate | 相對 sweep 起點的 ops 降幅 | 相對 sweep 起點的 mem 降幅 | ATE 狀態 |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for parameter in PARAMETER_ORDER:
        rows = [row for row in summary_rows if row["sweep_parameter"] == parameter]
        if not rows:
            continue
        baseline = rows[0]
        endpoint = rows[-1]
        ops_drop = percent_reduction(float(baseline["total_ops"]), float(endpoint["total_ops"]))
        mem_drop = percent_reduction(float(baseline["total_memory_bytes"]), float(endpoint["total_memory_bytes"]))
        ate_status = "完成" if endpoint["ate_m"] != "" else "待補"
        lines.append(
            f"| `{PARAMETER_LABELS[parameter]}` | `{endpoint['sweep_value']}` | "
            f"{ops_drop:.1f}% | {mem_drop:.1f}% | {ate_status} |"
        )
    return "\n".join(lines)


def write_report(args: argparse.Namespace, summary_rows: Sequence[dict[str, Any]]) -> None:
    any_ate = any(row["ate_m"] != "" for row in summary_rows)
    command = (
        "python3 DPVO/tools/statistic_dpvo/sweep_dpvo.py --run-eval --trials 3"
    )
    if args.parameters != PARAMETER_ORDER:
        command += " --parameters " + " ".join(args.parameters)

    report = f"""# DPVO One-at-a-Time Algorithmic Parameter Sweep 報告

## 範圍

這份報告由 `tools/statistic_dpvo/sweep_dpvo.py` 產生。Sweep 方式是從
`config/default.yaml` 出發，每次只改動一個 tunable parameter。對
`{PARAMETER_LABELS['IMAGE_SIZE']}` 而言，基準點定義為
`{DEFAULT_HEIGHT}x{DEFAULT_WIDTH}`，也就是 `statistic_dpvo.py` 使用的預設輸入尺寸。
`BA_ITERATIONS` 依本次需求改為由 `20` 逐步下降到 default `2`。

EuRoC sequences 依照 DPVO 論文 Table 2，以及本 repository 的
`evaluate_euroc.py`：{", ".join(EUROC_SCENES)}。

## 重現方式

先準備資料與 runtime 環境：

```bash
python3 DPVO/tools/statistic_dpvo/download_euroc.py
conda activate dpvo
```

執行完整 sweep：

```bash
{command}
```

產生的輸出：

- `statistic_result/generated_configs/*.yaml`
- `statistic_result/per_module/*.json`
- `statistic_result/per_module/*.csv`
- `statistic_result/sweep_summary.csv`
- `statistic_result/module_summary.csv`
- `statistic_result/sequence_errors.csv`
- `statistic_result/sweep_plots/*_sweep.svg`

## Sweep 離散點

- `PATCHES_PER_FRAME`: 96, 80, 64, 48, 32
- `PATCH_LIFETIME`: 13, 11, 9, 7, 5
- `REMOVAL_WINDOW`: 22, 18, 14, 10
- `OPTIMIZATION_WINDOW`: 10, 8, 6, 4
- `BA_ITERATIONS`: 20, 18, 16, 14, 12, 10, 8, 6, 4, 2
- `{PARAMETER_LABELS['IMAGE_SIZE']}`: 480x640, 384x512, 320x416, 240x320, 192x256

## 目前結果摘要

{report_table(summary_rows)}

ATE 狀態：{"已完成" if any_ate else "待補。這次執行沒有進行 EuRoC image evaluation；需要在具備完整 EuRoC image dataset 的 DPVO runtime 環境中執行上方命令，才會填入 ATE 欄位。"}

所有 sweep plots 針對同一個 metric 共用同一組 y-axis range：藍線的 total ops 軸
在所有圖一致，橘色虛線的 total mem 軸在所有圖一致，ATE 軸也會在所有圖一致。若尚未
執行 ATE evaluation，綠色虛線只代表 ATE pending；等 `--run-eval` 產生 `ate_m`
後會改畫實際 ATE 曲線。

## 分析

Static estimator 顯示，當各參數逐步下降時，logical workload 符合預期地下降。
`PATCHES_PER_FRAME`、`PATCH_LIFETIME` 與 `REMOVAL_WINDOW` 會直接降低 active
factor count，因此會同時影響 update、correlation 與 BA-heavy modules。
`OPTIMIZATION_WINDOW` 主要縮小 BA 中 free pose 的維度，所以對 front-end
neural-network workload 的影響較小，但仍可能影響 trajectory consistency。
新的 `BA_ITERATIONS` sweep 從 `20` 下降到 `2`；這能量化 solver refinement 次數
對 BA workload 的線性影響，也能在後續 ATE 補齊時判斷 iteration 是否有 accuracy
收益。較小的 `{PARAMETER_LABELS['IMAGE_SIZE']}` 會降低 feature extraction 與
correlation traffic，但也會改變輸入影像訊號，並可能和 patch selection 產生強交互作用。

## 候選 Algorithmic Parameter Sets P_a

ATE 補齊後，建議先驗證下列候選組合：

- `P_a_default`: `PATCHES_PER_FRAME=96`, `PATCH_LIFETIME=13`,
  `REMOVAL_WINDOW=22`, `OPTIMIZATION_WINDOW=10`, `BA_ITERATIONS=2`,
  `H,W=480x640`.
- `P_a_balanced`: `PATCHES_PER_FRAME=64`, `PATCH_LIFETIME=11`,
  `REMOVAL_WINDOW=18`, `OPTIMIZATION_WINDOW=8`, `BA_ITERATIONS=2`,
  `H,W=384x512`.
- `P_a_aggressive`: `PATCHES_PER_FRAME=48`, `PATCH_LIFETIME=9`,
  `REMOVAL_WINDOW=14`, `OPTIMIZATION_WINDOW=6`, `BA_ITERATIONS=2`,
  `H,W=320x416`.

最終選擇規則：保留 EuRoC average ATE 增幅仍在 project tolerance 內的 candidates，
再從這些 survivors 中選擇 total ops / total memory 最低的點。在目前尚未補齊 ATE
前，`P_a_balanced` 是較適合作為第一個 combined candidate 的保守選擇，因為它避開
最容易影響 accuracy 的變更（過低 `BA_ITERATIONS` 與過低 image size），同時仍能降低
factor-graph size。
"""
    (TOOLS_DIR / "one_at_a_time_sweep_report.md").write_text(report)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    try:
        base_values = read_base_values(args.config)
        sweeps = load_sweeps(base_values, args.sweep_file)
        candidates = generate_candidates(args, sweeps)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    results: list[StatsResult] = []
    module_summary_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        try:
            result = collect_statistics(args, candidate)
        except Exception as exc:
            print(f"error: failed to collect statistics for {candidate.candidate_id}: {exc}", file=sys.stderr)
            return 2
        results.append(result)
        module_summary_rows.extend(exact_module_row(candidate, row) for row in result.module_rows)
        print(
            f"stats {candidate.candidate_id}: "
            f"ops={result.total.total_ops} mem={result.total.total_memory_bytes}"
        )

    eval_preflight_errors: list[str] = []
    if args.run_eval:
        eval_preflight_errors = preflight_evaluation(args)
        if eval_preflight_errors:
            for error in eval_preflight_errors:
                print(f"evaluation preflight: {error}", file=sys.stderr)
            for result in results:
                result.evaluation_status = "preflight_failed"
        else:
            eval_results: Iterable[StatsResult] = results
            if not args.no_progress and tqdm is not None:
                eval_results = tqdm(
                    results,
                    desc="ATE candidates",
                    unit="candidate",
                    dynamic_ncols=True,
                    leave=True,
                )
            for result in eval_results:
                run_evaluation(args, result)
                print(f"eval {result.candidate.candidate_id}: {result.evaluation_status}")
    elif args.reuse_eval:
        for result in results:
            output_json = args.result_dir / "euroc_eval" / f"{result.candidate.candidate_id}.json"
            if output_json.exists():
                apply_eval_json(result, output_json)

    summary_rows = [summary_row(result) for result in results]
    all_scene_rows = [row for result in results for row in scene_rows(result)]
    write_csv(args.result_dir / "sweep_summary.csv", summary_rows, SUMMARY_COLUMNS)
    write_csv(args.result_dir / "module_summary.csv", module_summary_rows, MODULE_COLUMNS)
    write_csv(args.result_dir / "sequence_errors.csv", all_scene_rows, SCENE_COLUMNS)
    write_plots(args, summary_rows)
    write_report(args, summary_rows)
    print(f"wrote {args.result_dir / 'sweep_summary.csv'}")
    print(f"wrote {TOOLS_DIR / 'one_at_a_time_sweep_report.md'}")
    return 2 if eval_preflight_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
