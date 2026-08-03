#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
TOOLS_DIR = SCRIPT_DIR.parent
REPO_ROOT = TOOLS_DIR.parent
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
sys.path.insert(0, str(REPO_ROOT))


@dataclass
class TrackerConfig:
    BUFFER_SIZE: int
    CENTROID_SEL_STRAT: str
    PATCHES_PER_FRAME: int
    REMOVAL_WINDOW: int
    OPTIMIZATION_WINDOW: int
    PATCH_LIFETIME: int
    KEYFRAME_INDEX: int
    KEYFRAME_THRESH: float
    MOTION_MODEL: str
    MOTION_DAMPING: float
    BA_ITERATIONS: int
    MIXED_PRECISION: bool
    LOOP_CLOSURE: bool
    BACKEND_THRESH: float
    MAX_EDGE_AGE: int
    GLOBAL_OPT_FREQ: int
    CLASSIC_LOOP_CLOSURE: bool
    LOOP_CLOSE_WINDOW_SIZE: int
    LOOP_RETR_THRESH: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a compact end-to-end DPVO test package. The script writes "
            "preprocessed downsampled frames plus calibration as input data, then "
            "runs the Python DPVO tracker and stores golden outputs."
        )
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=REPO_ROOT / "dpvo.pth",
        help="Path to the DPVO checkpoint.",
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=REPO_ROOT / "subset_0493",
        help="Directory containing source image frames.",
    )
    parser.add_argument(
        "--calib",
        type=Path,
        default=REPO_ROOT / "calib" / "iphone.txt",
        help="Calibration text file for the source sequence.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "testdata" / "dpvo_python_small",
        help="Output directory for the compact input sequence and golden data.",
    )
    parser.add_argument(
        "--frame-start",
        type=int,
        default=1,
        help="1-based first frame index from the source image directory.",
    )
    parser.add_argument(
        "--frame-count",
        type=int,
        default=12,
        help="Number of frames to include. Must be at least 8 for DPVO initialization.",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Stride between selected source frames.",
    )
    parser.add_argument(
        "--max-long-edge",
        type=int,
        default=256,
        help="Preserve aspect ratio and cap the longer image edge at this size before 16-pixel alignment.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=0,
        help="Optional explicit output width. When set with --height, overrides --max-long-edge.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=0,
        help="Optional explicit output height. When set with --width, overrides --max-long-edge.",
    )
    parser.add_argument(
        "--patches-per-frame",
        type=int,
        default=32,
        help="Tracker PATCHES_PER_FRAME for the generated test run.",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=64,
        help="Tracker BUFFER_SIZE.",
    )
    parser.add_argument(
        "--removal-window",
        type=int,
        default=12,
        help="Tracker REMOVAL_WINDOW.",
    )
    parser.add_argument(
        "--optimization-window",
        type=int,
        default=8,
        help="Tracker OPTIMIZATION_WINDOW.",
    )
    parser.add_argument(
        "--patch-lifetime",
        type=int,
        default=10,
        help="Tracker PATCH_LIFETIME.",
    )
    parser.add_argument(
        "--keyframe-index",
        type=int,
        default=4,
        help="Tracker KEYFRAME_INDEX.",
    )
    parser.add_argument(
        "--keyframe-thresh",
        type=float,
        default=12.5,
        help="Tracker KEYFRAME_THRESH.",
    )
    parser.add_argument(
        "--motion-model",
        type=str,
        default="DAMPED_LINEAR",
        help="Tracker MOTION_MODEL.",
    )
    parser.add_argument(
        "--motion-damping",
        type=float,
        default=0.5,
        help="Tracker MOTION_DAMPING.",
    )
    parser.add_argument(
        "--centroid-sel-strat",
        choices=["RANDOM", "GRADIENT_BIAS"],
        default="RANDOM",
        help="Tracker centroid selection strategy. Seed the run for determinism.",
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Enable DPVO mixed precision. Disabled by default for more stable goldens.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Seed used for patch selection and any other stochastic tracker behavior.",
    )
    parser.add_argument(
        "--dump-state",
        action="store_true",
        help="Also save final DPVO patch-graph state tensors for deeper parity debugging.",
    )
    parser.add_argument(
        "--dump-update-parity-cases",
        action="store_true",
        help=(
            "Also save one update_block parity case per recorded tracker update "
            "under <output-root>/update_parity/update_XXX."
        ),
    )
    parser.add_argument(
        "--ba-debug-update-index",
        type=int,
        default=0,
        help="Update-trace index whose BA C/Q/w/dZ internals should be dumped.",
    )
    parser.add_argument(
        "--ba-debug-patches",
        type=str,
        default="",
        help=(
            "Comma-separated absolute patch indices to dump for BA debug. "
            "Empty means all patches; set to 'none' to disable."
        ),
    )
    return parser.parse_args()


def parse_ba_debug_patches(value: str) -> list[int] | None:
    value = value.strip()
    if value.lower() in {"none", "off", "false", "disabled"}:
        return None
    if not value:
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def collect_image_paths(image_dir: Path, frame_start: int, frame_count: int, frame_step: int) -> list[Path]:
    if frame_start <= 0:
        raise ValueError("--frame-start must be positive")
    if frame_count < 8:
        raise ValueError("--frame-count must be at least 8 so DPVO can initialize")
    if frame_step <= 0:
        raise ValueError("--frame-step must be positive")

    image_paths = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
    start = frame_start - 1
    stop = start + frame_count * frame_step
    selected = image_paths[start:stop:frame_step]
    if len(selected) != frame_count:
        raise ValueError(
            f"Requested {frame_count} frames from {image_dir} starting at {frame_start} with step {frame_step}, "
            f"but only found {len(selected)} usable frames"
        )
    return selected


def load_calibration(calib_path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    calib = np.loadtxt(calib_path, delimiter=" ", dtype=np.float64)
    calib = np.atleast_1d(calib).reshape(-1)
    if calib.size < 4:
        raise ValueError(f"Expected at least 4 calibration values in {calib_path}")
    intrinsics = calib[:4].astype(np.float64, copy=False)
    distortion = calib[4:].astype(np.float64, copy=False) if calib.size > 4 else None
    return intrinsics, distortion


def align_down_to_multiple(value: int, multiple: int) -> int:
    aligned = value - (value % multiple)
    return max(multiple, aligned)


def compute_output_shape(
    src_width: int,
    src_height: int,
    width_override: int,
    height_override: int,
    max_long_edge: int,
) -> tuple[int, int]:
    if bool(width_override) != bool(height_override):
        raise ValueError("--width and --height must be provided together")

    if width_override and height_override:
        width = width_override
        height = height_override
    else:
        if max_long_edge <= 0:
            raise ValueError("--max-long-edge must be positive")
        scale = min(1.0, float(max_long_edge) / float(max(src_width, src_height)))
        width = max(1, int(round(src_width * scale)))
        height = max(1, int(round(src_height * scale)))

    width = align_down_to_multiple(width, 16)
    height = align_down_to_multiple(height, 16)
    if width > src_width or height > src_height:
        raise ValueError(
            f"Requested output size {width}x{height} exceeds the source size {src_width}x{src_height}. "
            "This script is intended to generate downsampled test inputs."
        )
    return width, height


def build_tracker_config(args: argparse.Namespace) -> TrackerConfig:
    return TrackerConfig(
        BUFFER_SIZE=max(args.buffer_size, args.frame_count + 8),
        CENTROID_SEL_STRAT=args.centroid_sel_strat,
        PATCHES_PER_FRAME=args.patches_per_frame,
        REMOVAL_WINDOW=args.removal_window,
        OPTIMIZATION_WINDOW=args.optimization_window,
        PATCH_LIFETIME=args.patch_lifetime,
        KEYFRAME_INDEX=args.keyframe_index,
        KEYFRAME_THRESH=args.keyframe_thresh,
        MOTION_MODEL=args.motion_model,
        MOTION_DAMPING=args.motion_damping,
        BA_ITERATIONS=2,
        MIXED_PRECISION=args.mixed_precision,
        LOOP_CLOSURE=False,
        BACKEND_THRESH=64.0,
        MAX_EDGE_AGE=1000,
        GLOBAL_OPT_FREQ=15,
        CLASSIC_LOOP_CLOSURE=False,
        LOOP_CLOSE_WINDOW_SIZE=3,
        LOOP_RETR_THRESH=0.04,
    )


def prepare_output_root(output_root: Path) -> tuple[Path, Path]:
    image_dir = output_root / "images"
    golden_dir = output_root / "golden"
    centers_dir = output_root / "centers"
    bootstrap_depth_dir = output_root / "bootstrap_depths"
    for path in (image_dir, golden_dir, centers_dir, bootstrap_depth_dir):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
    for path in (
        output_root / "calib.txt",
        output_root / "metadata.json",
        output_root / "centers_manifest.txt",
        output_root / "bootstrap_depth_manifest.txt",
    ):
        if path.exists():
            path.unlink()
    return image_dir, golden_dir


def preprocess_frames(
    image_paths: list[Path],
    source_intrinsics: np.ndarray,
    distortion: np.ndarray | None,
    output_root: Path,
    width_override: int,
    height_override: int,
    max_long_edge: int,
) -> tuple[list[np.ndarray], list[str], np.ndarray, int, int, int, int]:
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("OpenCV is required to generate resized DPVO input frames") from exc

    first = cv2.imread(str(image_paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Failed to read image: {image_paths[0]}")

    src_height, src_width = first.shape[:2]
    dst_width, dst_height = compute_output_shape(
        src_width=src_width,
        src_height=src_height,
        width_override=width_override,
        height_override=height_override,
        max_long_edge=max_long_edge,
    )

    fx, fy, cx, cy = map(float, source_intrinsics.tolist())
    K = np.eye(3, dtype=np.float64)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy

    sx = dst_width / float(src_width)
    sy = dst_height / float(src_height)
    output_intrinsics = np.array([fx * sx, fy * sy, cx * sx, cy * sy], dtype=np.float32)

    frames: list[np.ndarray] = []
    frame_names: list[str] = []
    interpolation = cv2.INTER_AREA if dst_width < src_width or dst_height < src_height else cv2.INTER_LINEAR

    for frame_index, image_path in enumerate(image_paths):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        if distortion is not None and distortion.size > 0:
            image = cv2.undistort(image, K, distortion)

        if image.shape[1] != dst_width or image.shape[0] != dst_height:
            image = cv2.resize(image, (dst_width, dst_height), interpolation=interpolation)

        image = np.ascontiguousarray(image)
        frame_name = f"{frame_index:06d}.png"
        if not cv2.imwrite(str(output_root / frame_name), image):
            raise RuntimeError(f"Failed to write resized image: {output_root / frame_name}")
        frames.append(image)
        frame_names.append(frame_name)

    return frames, frame_names, output_intrinsics, src_width, src_height, dst_width, dst_height


def write_adjusted_calibration(calib_path: Path, intrinsics: np.ndarray) -> None:
    calib_path.write_text(
        " ".join(f"{value:.8f}" for value in intrinsics.tolist()) + "\n",
        encoding="utf-8",
    )


def write_manifest(output_dir: Path, tensors: OrderedDict[str, np.ndarray]) -> None:
    lines = [
        "# dpvo_python_testdata_v1",
        "# filename is inferred as <tensor_name>.bin",
    ]
    for name, array in tensors.items():
        shape = " ".join(str(dim) for dim in array.shape)
        lines.append(f"{name} {array.dtype.name} {shape}".rstrip())
    (output_dir / "manifest.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_case(output_dir: Path, tensors: OrderedDict[str, np.ndarray]) -> None:
    write_manifest(output_dir, tensors)
    for name, array in tensors.items():
        np.ascontiguousarray(array).tofile(output_dir / f"{name}.bin")


def write_update_parity_cases(output_root: Path, cases: list[OrderedDict[str, np.ndarray]]) -> Path:
    case_root = output_root / "update_parity"
    if case_root.exists():
        shutil.rmtree(case_root)
    case_root.mkdir(parents=True, exist_ok=True)

    for index, tensors in enumerate(cases):
        case_dir = case_root / f"update_{index:03d}"
        case_dir.mkdir(parents=True, exist_ok=True)
        write_case(case_dir, tensors)
    return case_root


def write_vector_manifest(
    output_root: Path,
    subdir_name: str,
    manifest_name: str,
    file_prefix: str,
    vectors_by_frame: list[np.ndarray],
) -> Path:
    vector_dir = output_root / subdir_name
    vector_dir.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    for frame_index, values in enumerate(vectors_by_frame):
        if values.ndim != 1:
            raise ValueError(
                f"Expected {manifest_name} values for frame {frame_index} to have rank 1, got {tuple(values.shape)}"
            )

        relative_path = Path(subdir_name) / f"{file_prefix}_{frame_index:06d}.bin"
        np.ascontiguousarray(values.astype(np.float32, copy=False)).tofile(output_root / relative_path)
        lines.append(f"{frame_index} {relative_path.as_posix()} float32 {values.shape[0]}")

    manifest_path = output_root / manifest_name
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest_path


def write_centers_manifest(output_root: Path, centers_by_frame: list[np.ndarray]) -> Path:
    centers_dir = output_root / "centers"
    centers_dir.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    for frame_index, centers in enumerate(centers_by_frame):
        if centers.ndim != 2 or centers.shape[1] != 2:
            raise ValueError(
                f"Expected centers for frame {frame_index} to have shape [M,2], got {tuple(centers.shape)}"
            )

        relative_path = Path("centers") / f"centers_{frame_index:06d}.bin"
        np.ascontiguousarray(centers.astype(np.float32, copy=False)).tofile(output_root / relative_path)
        lines.append(
            f"{frame_index} {relative_path.as_posix()} float32 {centers.shape[0]} {centers.shape[1]}"
        )

    manifest_path = output_root / "centers_manifest.txt"
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest_path


def write_tum_trajectory(path: Path, poses: np.ndarray, tstamps: np.ndarray) -> None:
    lines = []
    for tstamp, pose in zip(tstamps.tolist(), poses.tolist()):
        x, y, z, qx, qy, qz, qw = pose
        lines.append(
            f"{float(tstamp):.9f} "
            f"{x:.9f} {y:.9f} {z:.9f} "
            f"{qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def configure_determinism(seed: int) -> None:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("PyTorch is required to run the Python DPVO tracker") from exc

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def compute_ba_debug_trace(
    poses: Any,
    patches: Any,
    intrinsics: Any,
    target: Any,
    weight: Any,
    ii: Any,
    jj: Any,
    kk: Any,
    t0: int,
    t1: int,
    iterations: int,
    patch_filter: list[int],
) -> dict[str, np.ndarray]:
    import torch

    from dpvo import projective_ops as pops
    from dpvo.lietorch import SE3

    poses_debug = poses.detach().clone().float()
    patches_debug = patches.detach().clone().float()
    intrinsics_debug = intrinsics.detach().clone().float()
    target_flat = target.detach().reshape(-1, 2).float()
    weight_flat = weight.detach().reshape(-1, 2).float()
    ii_flat = ii.detach().reshape(-1).long()
    jj_flat = jj.detach().reshape(-1).long()
    kk_flat = kk.detach().reshape(-1).long()
    t0 = int(t0)
    t1 = int(t1)
    iterations = int(iterations)

    device = poses_debug.device
    fx, fy, cx, cy = intrinsics_debug.reshape(-1, 4)[0]
    del fx, fy
    bound_min_x = torch.tensor(-64.0, device=device)
    bound_min_y = torch.tensor(-64.0, device=device)
    bound_max_x = 2.0 * cx + 64.0
    bound_max_y = 2.0 * cy + 64.0

    debug_iterations: list[np.ndarray] = []
    debug_patch_indices: list[np.ndarray] = []
    debug_c: list[np.ndarray] = []
    debug_q: list[np.ndarray] = []
    debug_w: list[np.ndarray] = []
    debug_dz: list[np.ndarray] = []
    debug_coords_center: list[np.ndarray] = []
    debug_residual: list[np.ndarray] = []
    debug_jz: list[np.ndarray] = []
    debug_weight: list[np.ndarray] = []
    debug_c_contrib: list[np.ndarray] = []
    debug_w_contrib: list[np.ndarray] = []

    for iteration in range(iterations):
        kx, ku = torch.unique(kk_flat, sorted=True, return_inverse=True)
        pose_count = max(t1 - t0, 0)
        point_count = int(kx.numel())
        pose_dim = pose_count * 6

        coords, valid, (Ji, Jj, Jz) = pops.transform(
            SE3(poses_debug), patches_debug, intrinsics_debug, ii_flat, jj_flat, kk_flat, jacobian=True
        )
        patch_size = coords.shape[2]
        coords_center = coords[:, :, patch_size // 2, patch_size // 2, :].reshape(-1, 2)
        residual = target_flat - coords_center
        residual_norm = torch.linalg.norm(residual, dim=-1)
        valid_flat = valid.reshape(-1) > 0.5
        in_bounds = (
            (residual_norm < 128.0)
            & valid_flat
            & (coords_center[:, 0] > bound_min_x)
            & (coords_center[:, 1] > bound_min_y)
            & (coords_center[:, 0] < bound_max_x)
            & (coords_center[:, 1] < bound_max_y)
        )

        Ji = Ji.reshape(-1, 2, 6).float()
        Jj = Jj.reshape(-1, 2, 6).float()
        Jz = Jz.reshape(-1, 2).float()

        B = torch.zeros((pose_dim, pose_dim), device=device, dtype=torch.float32)
        E = torch.zeros((pose_dim, point_count), device=device, dtype=torch.float32)
        C = torch.zeros((point_count,), device=device, dtype=torch.float32)
        v = torch.zeros((pose_dim,), device=device, dtype=torch.float32)
        w_vec = torch.zeros((point_count,), device=device, dtype=torch.float32)
        c_contrib = torch.zeros((int(ii_flat.numel()), 2), device=device, dtype=torch.float32)
        w_contrib = torch.zeros_like(c_contrib)

        for edge in range(int(ii_flat.numel())):
            if not bool(in_bounds[edge].item()):
                continue
            point_index = int(ku[edge].item())
            i_frame = int(ii_flat[edge].item()) - t0
            j_frame = int(jj_flat[edge].item()) - t0
            for obs in range(2):
                obs_weight = weight_flat[edge, obs]
                obs_residual = residual[edge, obs]
                obs_jz = Jz[edge, obs]
                ji = Ji[edge, obs]
                jj_row = Jj[edge, obs]
                if i_frame >= 0:
                    i_slice = slice(i_frame * 6, (i_frame + 1) * 6)
                    B[i_slice, i_slice] += obs_weight * torch.outer(ji, ji)
                    E[i_slice, point_index] += obs_weight * ji * obs_jz
                    v[i_slice] += obs_weight * ji * obs_residual
                if j_frame >= 0:
                    j_slice = slice(j_frame * 6, (j_frame + 1) * 6)
                    B[j_slice, j_slice] += obs_weight * torch.outer(jj_row, jj_row)
                    E[j_slice, point_index] += obs_weight * jj_row * obs_jz
                    v[j_slice] += obs_weight * jj_row * obs_residual
                if i_frame >= 0 and j_frame >= 0:
                    i_slice = slice(i_frame * 6, (i_frame + 1) * 6)
                    j_slice = slice(j_frame * 6, (j_frame + 1) * 6)
                    B[i_slice, j_slice] += obs_weight * torch.outer(ji, jj_row)
                    B[j_slice, i_slice] += obs_weight * torch.outer(jj_row, ji)
                obs_c_contrib = obs_weight * obs_jz * obs_jz
                obs_w_contrib = obs_weight * obs_jz * obs_residual
                c_contrib[edge, obs] = obs_c_contrib
                w_contrib[edge, obs] = obs_w_contrib
                C[point_index] += obs_c_contrib
                w_vec[point_index] += obs_w_contrib

        Q = 1.0 / (C + 1e-4)
        if pose_dim > 0:
            EQ = E * Q.reshape(1, -1)
            S = B - torch.matmul(EQ, E.transpose(0, 1))
            y_vec = v - torch.matmul(EQ, w_vec.reshape(-1, 1)).reshape(-1)
            diagonal = torch.arange(pose_dim, device=device)
            S[diagonal, diagonal] = S[diagonal, diagonal] + 1.0 + 1e-4 * S[diagonal, diagonal]
            chol, info = torch.linalg.cholesky_ex(S)
            if bool(torch.any(info != 0).item()):
                raise RuntimeError(f"BA debug cholesky failed at iteration {iteration}: info={info.detach().cpu().tolist()}")
            dX = torch.cholesky_solve(y_vec.reshape(-1, 1), chol).reshape(-1)
            dZ = Q * (w_vec - torch.matmul(E.transpose(0, 1), dX.reshape(-1, 1)).reshape(-1))
        else:
            dX = torch.zeros((0,), device=device, dtype=torch.float32)
            dZ = Q * w_vec

        selected_patch_indices = (
            torch.tensor(patch_filter, device=device, dtype=torch.long)
            if patch_filter
            else kx
        )
        selected_c = torch.zeros((int(selected_patch_indices.numel()),), device=device, dtype=torch.float32)
        selected_q = torch.zeros_like(selected_c)
        selected_w = torch.zeros_like(selected_c)
        selected_dz = torch.zeros_like(selected_c)
        for selected_index, patch_index_value in enumerate(selected_patch_indices.tolist()):
            patch_index = int(patch_index_value)
            matches = torch.nonzero(kx == int(patch_index), as_tuple=False).reshape(-1)
            if int(matches.numel()) == 0:
                continue
            point_index = int(matches[0].item())
            selected_c[selected_index] = C[point_index]
            selected_q[selected_index] = Q[point_index]
            selected_w[selected_index] = w_vec[point_index]
            selected_dz[selected_index] = dZ[point_index]

        debug_iterations.append(np.array(iteration, dtype=np.int64))
        debug_patch_indices.append(selected_patch_indices.detach().cpu().numpy().astype(np.int64, copy=True))
        debug_c.append(selected_c.detach().cpu().numpy().astype(np.float32, copy=True))
        debug_q.append(selected_q.detach().cpu().numpy().astype(np.float32, copy=True))
        debug_w.append(selected_w.detach().cpu().numpy().astype(np.float32, copy=True))
        debug_dz.append(selected_dz.detach().cpu().numpy().astype(np.float32, copy=True))
        debug_coords_center.append(coords_center.detach().cpu().numpy().astype(np.float32, copy=True))
        debug_residual.append(residual.detach().cpu().numpy().astype(np.float32, copy=True))
        debug_jz.append(Jz.detach().cpu().numpy().astype(np.float32, copy=True))
        debug_weight.append(weight_flat.detach().cpu().numpy().astype(np.float32, copy=True))
        debug_c_contrib.append(c_contrib.detach().cpu().numpy().astype(np.float32, copy=True))
        debug_w_contrib.append(w_contrib.detach().cpu().numpy().astype(np.float32, copy=True))

        patches_flat = patches_debug.reshape(-1, patches_debug.shape[-3], patches_debug.shape[-2], patches_debug.shape[-1])
        for point_index, patch_index in enumerate(kx.tolist()):
            depth = patches_flat[int(patch_index), 2, 0, 0] + dZ[point_index]
            depth = torch.where(depth > 20.0, torch.ones_like(depth), depth)
            depth = torch.clamp(depth, min=1e-4)
            patches_flat[int(patch_index), 2, :, :] = depth

        if pose_dim > 0:
            dx_full = torch.zeros((*poses_debug.shape[:-1], 6), device=device, dtype=torch.float32)
            dx_full.reshape(-1, 6)[t0:t1, :] = dX.reshape(pose_count, 6)
            poses_debug = SE3(poses_debug).retr(dx_full).data.float()

    return {
        "ba_debug_iterations": np.stack(debug_iterations, axis=0).astype(np.int64, copy=False),
        "ba_debug_patch_indices": np.stack(debug_patch_indices, axis=0).astype(np.int64, copy=False),
        "ba_debug_c": np.stack(debug_c, axis=0).astype(np.float32, copy=False),
        "ba_debug_q": np.stack(debug_q, axis=0).astype(np.float32, copy=False),
        "ba_debug_w": np.stack(debug_w, axis=0).astype(np.float32, copy=False),
        "ba_debug_dz": np.stack(debug_dz, axis=0).astype(np.float32, copy=False),
        "ba_debug_coords_center": np.stack(debug_coords_center, axis=0).astype(np.float32, copy=False),
        "ba_debug_residual": np.stack(debug_residual, axis=0).astype(np.float32, copy=False),
        "ba_debug_jz": np.stack(debug_jz, axis=0).astype(np.float32, copy=False),
        "ba_debug_weight": np.stack(debug_weight, axis=0).astype(np.float32, copy=False),
        "ba_debug_c_contrib": np.stack(debug_c_contrib, axis=0).astype(np.float32, copy=False),
        "ba_debug_w_contrib": np.stack(debug_w_contrib, axis=0).astype(np.float32, copy=False),
    }


def run_tracker(
    weights: Path,
    frames: list[np.ndarray],
    intrinsics: np.ndarray,
    tracker_cfg: TrackerConfig,
    ba_debug_update_index: int,
    ba_debug_patches: list[int] | None,
) -> tuple[
    Any,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[np.ndarray],
    list[np.ndarray],
    list[dict[str, np.ndarray]],
    list[OrderedDict[str, np.ndarray]],
]:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("PyTorch is required to run the Python DPVO tracker") from exc

    import dpvo.dpvo as dpvo_module
    from dpvo.dpvo import DPVO

    if not torch.cuda.is_available():
        raise RuntimeError("DPVO testdata generation requires CUDA because the Python tracker runs on GPU")

    height, width = frames[0].shape[:2]
    slam = DPVO(tracker_cfg, str(weights), ht=height, wd=width, viz=False)
    centers_by_frame: list[np.ndarray] = []
    bootstrap_depths_by_frame: list[np.ndarray | None] = [None] * len(frames)
    update_trace: list[dict[str, np.ndarray]] = []
    update_parity_cases: list[OrderedDict[str, np.ndarray]] = []
    ba_debug_by_update_index: dict[int, dict[str, np.ndarray]] = {}
    ba_call_index = 0

    original_patchify_forward = slam.network.patchify.forward
    original_rand_like = torch.rand_like
    original_update = slam.update
    original_network_update_forward = slam.network.update.forward
    original_fastba_ba = dpvo_module.fastba.BA
    current_frame_index: int | None = None
    recording_tracker_update = False

    def recording_patchify_forward(images, patches_per_image=80, disps=None, centroid_sel_strat="RANDOM", return_color=False):
        outputs = original_patchify_forward(
            images,
            patches_per_image=patches_per_image,
            disps=disps,
            centroid_sel_strat=centroid_sel_strat,
            return_color=return_color,
        )

        patches = outputs[3]
        centers = torch.stack([patches[0, :, 0, 1, 1], patches[0, :, 1, 1, 1]], dim=-1)
        centers_by_frame.append(centers.detach().cpu().numpy().astype(np.float32, copy=False))
        return outputs

    def recording_rand_like(tensor, *args, **kwargs):
        result = original_rand_like(tensor, *args, **kwargs)
        nonlocal current_frame_index

        def looks_like_bootstrap_depth_draw(shape: Any) -> bool:
            if len(shape) < 2:
                return False
            if shape[0] != 1 or shape[1] != tracker_cfg.PATCHES_PER_FRAME:
                return False
            return all(dim == 1 for dim in shape[2:])

        if current_frame_index is not None and looks_like_bootstrap_depth_draw(result.shape):
            bootstrap_depths_by_frame[current_frame_index] = (
                result.detach().cpu().reshape(-1).numpy().astype(np.float32, copy=False)
            )
        return result

    def recording_fastba_ba(
        poses,
        patches,
        intrinsics,
        target,
        weight,
        lmbda,
        ii,
        jj,
        kk,
        t0,
        t1,
        M,
        iterations,
        eff_impl=False,
    ) -> Any:
        nonlocal ba_call_index
        current_update_index = ba_call_index
        ba_call_index += 1
        if ba_debug_patches is not None and current_update_index == ba_debug_update_index:
            try:
                ba_debug_by_update_index[current_update_index] = compute_ba_debug_trace(
                    poses=poses,
                    patches=patches,
                    intrinsics=intrinsics,
                    target=target,
                    weight=weight,
                    ii=ii,
                    jj=jj,
                    kk=kk,
                    t0=t0,
                    t1=t1,
                    iterations=iterations,
                    patch_filter=ba_debug_patches,
                )
            except Exception as exc:
                print(f"Warning BA debug capture failed for update {current_update_index}: {exc}")
        return original_fastba_ba(
            poses,
            patches,
            intrinsics,
            target,
            weight,
            lmbda,
            ii,
            jj,
            kk,
            t0,
            t1,
            M=M,
            iterations=iterations,
            eff_impl=eff_impl,
        )

    def recording_network_update_forward(net, inp, corr, flow, ii, jj, kk) -> Any:
        result = original_network_update_forward(net, inp, corr, flow, ii, jj, kk)
        if recording_tracker_update:
            net_out, (delta, weight, _) = result
            update_parity_cases.append(
                OrderedDict(
                    [
                        ("net", net.detach().cpu().numpy().astype(np.float32, copy=True)),
                        ("ctx", inp.detach().cpu().numpy().astype(np.float32, copy=True)),
                        ("corr", corr.detach().cpu().numpy().astype(np.float32, copy=True)),
                        ("ii", ii.detach().cpu().numpy().astype(np.int64, copy=True)),
                        ("jj", jj.detach().cpu().numpy().astype(np.int64, copy=True)),
                        ("kk", kk.detach().cpu().numpy().astype(np.int64, copy=True)),
                        ("golden_net", net_out.detach().cpu().numpy().astype(np.float32, copy=True)),
                        ("golden_delta", delta.detach().cpu().numpy().astype(np.float32, copy=True)),
                        ("golden_weight", weight.detach().cpu().numpy().astype(np.float32, copy=True)),
                    ]
                )
            )
        return result

    def recording_update(*args: Any, **kwargs: Any) -> Any:
        nonlocal recording_tracker_update
        recording_tracker_update = True
        try:
            result = original_update(*args, **kwargs)
        finally:
            recording_tracker_update = False
        if hasattr(slam.pg, "target") and hasattr(slam.pg, "weight"):
            target = slam.pg.target.detach().cpu().numpy().astype(np.float32, copy=False)
            weight = slam.pg.weight.detach().cpu().numpy().astype(np.float32, copy=False)
            ii = slam.pg.ii.detach().cpu().numpy().astype(np.int64, copy=False)
            jj = slam.pg.jj.detach().cpu().numpy().astype(np.int64, copy=False)
            kk = slam.pg.kk.detach().cpu().numpy().astype(np.int64, copy=False)
            poses = slam.pg.poses_[: slam.n].detach().cpu().numpy().astype(np.float32, copy=False)
            patch_depths = (
                slam.pg.patches_[: slam.n, :, 2]
                .reshape(-1, slam.P, slam.P)[: slam.m]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
            if target.ndim == 3 and weight.ndim == 3 and target.shape[1] == ii.shape[0]:
                entry = {
                    "ii": ii.copy(),
                    "jj": jj.copy(),
                    "kk": kk.copy(),
                    "target": target.copy(),
                    "weight": weight.copy(),
                    "post_ba_poses": poses.copy(),
                    "post_ba_patch_depths": patch_depths.copy(),
                }
                entry.update(ba_debug_by_update_index.get(len(update_trace), {}))
                update_trace.append(entry)
        return result

    slam.network.patchify.forward = recording_patchify_forward
    torch.rand_like = recording_rand_like
    slam.update = recording_update
    slam.network.update.forward = recording_network_update_forward
    dpvo_module.fastba.BA = recording_fastba_ba

    try:
        with torch.no_grad():
            for tstamp, frame in enumerate(frames):
                current_frame_index = tstamp
                image_tensor = torch.from_numpy(frame).permute(2, 0, 1).cuda(non_blocking=False)
                intrinsics_tensor = torch.from_numpy(intrinsics.copy()).cuda(non_blocking=False)
                slam(tstamp, image_tensor, intrinsics_tensor)
            current_frame_index = None

            poses, tstamps = slam.terminate()
    finally:
        slam.network.patchify.forward = original_patchify_forward
        torch.rand_like = original_rand_like
        slam.update = original_update
        slam.network.update.forward = original_network_update_forward
        dpvo_module.fastba.BA = original_fastba_ba

    points = slam.pg.points_.detach().cpu().numpy()[: slam.m].astype(np.float32, copy=False)
    colors = slam.pg.colors_.view(-1, 3).detach().cpu().numpy()[: slam.m].astype(np.uint8, copy=False)
    if len(centers_by_frame) != len(frames):
        raise RuntimeError(
            f"Recorded {len(centers_by_frame)} center sets for {len(frames)} frames; patchify capture is incomplete"
        )
    if any(depths is None for depths in bootstrap_depths_by_frame):
        missing = [idx for idx, depths in enumerate(bootstrap_depths_by_frame) if depths is None]
        raise RuntimeError(f"Failed to record bootstrap depth draws for frames: {missing}")
    return (
        slam,
        poses.astype(np.float32, copy=False),
        tstamps,
        points,
        colors,
        centers_by_frame,
        [depths for depths in bootstrap_depths_by_frame if depths is not None],
        update_trace,
        update_parity_cases,
    )


def dump_tracker_state(slam: Any, tracker_cfg: TrackerConfig) -> OrderedDict[str, np.ndarray]:
    def scalar_i64(value: int) -> np.ndarray:
        return np.array([value], dtype=np.int64)

    arrays: OrderedDict[str, np.ndarray] = OrderedDict()
    arrays["state_n"] = scalar_i64(int(slam.n))
    arrays["state_m"] = scalar_i64(int(slam.m))
    arrays["state_dim"] = scalar_i64(int(slam.DIM))
    arrays["state_patch_size"] = scalar_i64(int(slam.P))
    arrays["state_buffer_size"] = scalar_i64(int(tracker_cfg.BUFFER_SIZE))
    arrays["state_tstamps"] = slam.pg.tstamps_[: slam.n].astype(np.int64, copy=False)
    arrays["state_patch_frames"] = slam.ix[: slam.m].detach().cpu().numpy().astype(np.int64, copy=False)
    arrays["state_poses"] = slam.pg.poses_[: slam.n].detach().cpu().numpy().astype(np.float32, copy=False)
    arrays["state_patches"] = slam.pg.patches_[: slam.n].detach().cpu().numpy().astype(np.float32, copy=False)
    arrays["state_intrinsics"] = slam.pg.intrinsics_[: slam.n].detach().cpu().numpy().astype(np.float32, copy=False)
    arrays["state_colors"] = slam.pg.colors_[: slam.n].detach().cpu().numpy().astype(np.uint8, copy=False)
    arrays["state_points"] = slam.pg.points_[: slam.m].detach().cpu().numpy().astype(np.float32, copy=False)
    arrays["state_net"] = slam.pg.net.detach().cpu().numpy().astype(np.float32, copy=False)
    arrays["state_target"] = slam.pg.target.detach().cpu().numpy().astype(np.float32, copy=False)
    arrays["state_weight"] = slam.pg.weight.detach().cpu().numpy().astype(np.float32, copy=False)
    arrays["state_ii"] = slam.pg.ii.detach().cpu().numpy().astype(np.int64, copy=False)
    arrays["state_jj"] = slam.pg.jj.detach().cpu().numpy().astype(np.int64, copy=False)
    arrays["state_kk"] = slam.pg.kk.detach().cpu().numpy().astype(np.int64, copy=False)
    arrays["state_ii_inac"] = slam.pg.ii_inac.detach().cpu().numpy().astype(np.int64, copy=False)
    arrays["state_jj_inac"] = slam.pg.jj_inac.detach().cpu().numpy().astype(np.int64, copy=False)
    arrays["state_kk_inac"] = slam.pg.kk_inac.detach().cpu().numpy().astype(np.int64, copy=False)
    arrays["state_weight_inac"] = slam.pg.weight_inac.detach().cpu().numpy()
    arrays["state_target_inac"] = slam.pg.target_inac.detach().cpu().numpy()

    if isinstance(getattr(slam.pg, "delta", None), dict) and slam.pg.delta:
        delta_items = []
        for t, (t0, dP) in slam.pg.delta.items():
            pose = dP.data.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
            delta_items.append((int(t), int(t0), pose))
        delta_items.sort(key=lambda item: item[0])
        arrays["state_delta_tstamp"] = np.array([t for t, _, _ in delta_items], dtype=np.int64)
        arrays["state_delta_parent"] = np.array([t0 for _, t0, _ in delta_items], dtype=np.int64)
        arrays["state_delta_pose"] = np.stack([pose for _, _, pose in delta_items], axis=0).astype(np.float32, copy=False)
    else:
        arrays["state_delta_tstamp"] = np.zeros((0,), dtype=np.int64)
        arrays["state_delta_parent"] = np.zeros((0,), dtype=np.int64)
        arrays["state_delta_pose"] = np.zeros((0, 7), dtype=np.float32)

    return arrays


def append_ba_debug_trace(
    arrays: OrderedDict[str, np.ndarray],
    update_trace: list[dict[str, np.ndarray]],
) -> None:
    counts = np.array(
        [entry.get("ba_debug_iterations", np.zeros((0,), dtype=np.int64)).shape[0] for entry in update_trace],
        dtype=np.int64,
    )
    offsets = np.zeros((len(update_trace) + 1,), dtype=np.int64)
    if counts.size:
        offsets[1:] = np.cumsum(counts, dtype=np.int64)
    total_rows = int(offsets[-1])
    patch_width = 0
    edge_width = 0
    for entry in update_trace:
        patch_indices = entry.get("ba_debug_patch_indices")
        if patch_indices is not None and patch_indices.ndim == 2 and patch_indices.shape[1] > 0:
            patch_width = int(patch_indices.shape[1])
            break
    for entry in update_trace:
        edge_values = entry.get("ba_debug_coords_center")
        if edge_values is not None and edge_values.ndim == 3 and edge_values.shape[1] > 0:
            edge_width = int(edge_values.shape[1])
            break

    arrays["update_trace_ba_debug_counts"] = counts
    arrays["update_trace_ba_debug_offsets"] = offsets
    if total_rows == 0:
        arrays["update_trace_ba_debug_iterations"] = np.zeros((0,), dtype=np.int64)
        arrays["update_trace_ba_debug_patch_indices"] = np.zeros((0, patch_width), dtype=np.int64)
        arrays["update_trace_ba_debug_c"] = np.zeros((0, patch_width), dtype=np.float32)
        arrays["update_trace_ba_debug_q"] = np.zeros((0, patch_width), dtype=np.float32)
        arrays["update_trace_ba_debug_w"] = np.zeros((0, patch_width), dtype=np.float32)
        arrays["update_trace_ba_debug_dz"] = np.zeros((0, patch_width), dtype=np.float32)
        arrays["update_trace_ba_debug_coords_center"] = np.zeros((0, edge_width, 2), dtype=np.float32)
        arrays["update_trace_ba_debug_residual"] = np.zeros((0, edge_width, 2), dtype=np.float32)
        arrays["update_trace_ba_debug_jz"] = np.zeros((0, edge_width, 2), dtype=np.float32)
        arrays["update_trace_ba_debug_weight"] = np.zeros((0, edge_width, 2), dtype=np.float32)
        arrays["update_trace_ba_debug_c_contrib"] = np.zeros((0, edge_width, 2), dtype=np.float32)
        arrays["update_trace_ba_debug_w_contrib"] = np.zeros((0, edge_width, 2), dtype=np.float32)
        return

    arrays["update_trace_ba_debug_iterations"] = np.concatenate(
        [entry.get("ba_debug_iterations", np.zeros((0,), dtype=np.int64)) for entry in update_trace],
        axis=0,
    ).astype(np.int64, copy=False)
    arrays["update_trace_ba_debug_patch_indices"] = np.concatenate(
        [
            entry.get("ba_debug_patch_indices", np.zeros((0, patch_width), dtype=np.int64))
            for entry in update_trace
        ],
        axis=0,
    ).astype(np.int64, copy=False)
    for source_name, tensor_name in [
        ("ba_debug_c", "update_trace_ba_debug_c"),
        ("ba_debug_q", "update_trace_ba_debug_q"),
        ("ba_debug_w", "update_trace_ba_debug_w"),
        ("ba_debug_dz", "update_trace_ba_debug_dz"),
    ]:
        arrays[tensor_name] = np.concatenate(
            [
                entry.get(source_name, np.zeros((0, patch_width), dtype=np.float32))
                for entry in update_trace
            ],
            axis=0,
        ).astype(np.float32, copy=False)
    for source_name, tensor_name in [
        ("ba_debug_coords_center", "update_trace_ba_debug_coords_center"),
        ("ba_debug_residual", "update_trace_ba_debug_residual"),
        ("ba_debug_jz", "update_trace_ba_debug_jz"),
        ("ba_debug_weight", "update_trace_ba_debug_weight"),
        ("ba_debug_c_contrib", "update_trace_ba_debug_c_contrib"),
        ("ba_debug_w_contrib", "update_trace_ba_debug_w_contrib"),
    ]:
        arrays[tensor_name] = np.concatenate(
            [
                entry.get(source_name, np.zeros((0, edge_width, 2), dtype=np.float32))
                for entry in update_trace
            ],
            axis=0,
        ).astype(np.float32, copy=False)


def dump_update_trace(update_trace: list[dict[str, np.ndarray]]) -> OrderedDict[str, np.ndarray]:
    arrays: OrderedDict[str, np.ndarray] = OrderedDict()
    edge_counts = np.array([entry["ii"].shape[0] for entry in update_trace], dtype=np.int64)
    edge_offsets = np.zeros((len(update_trace) + 1,), dtype=np.int64)
    if edge_counts.size:
        edge_offsets[1:] = np.cumsum(edge_counts, dtype=np.int64)
    total_edges = int(edge_offsets[-1])

    arrays["update_trace_count"] = np.array([len(update_trace)], dtype=np.int64)
    arrays["update_trace_edge_counts"] = edge_counts
    arrays["update_trace_edge_offsets"] = edge_offsets

    if total_edges == 0:
        arrays["update_trace_ii"] = np.zeros((0,), dtype=np.int64)
        arrays["update_trace_jj"] = np.zeros((0,), dtype=np.int64)
        arrays["update_trace_kk"] = np.zeros((0,), dtype=np.int64)
        arrays["update_trace_target"] = np.zeros((1, 0, 2), dtype=np.float32)
        arrays["update_trace_weight"] = np.zeros((1, 0, 2), dtype=np.float32)
        arrays["update_trace_post_ba_pose_counts"] = np.zeros((0,), dtype=np.int64)
        arrays["update_trace_post_ba_pose_offsets"] = np.zeros((1,), dtype=np.int64)
        arrays["update_trace_post_ba_patch_counts"] = np.zeros((0,), dtype=np.int64)
        arrays["update_trace_post_ba_patch_offsets"] = np.zeros((1,), dtype=np.int64)
        arrays["update_trace_post_ba_poses"] = np.zeros((0, 7), dtype=np.float32)
        arrays["update_trace_post_ba_patch_depths"] = np.zeros((0, 3, 3), dtype=np.float32)
        append_ba_debug_trace(arrays, update_trace)
        return arrays

    arrays["update_trace_ii"] = np.concatenate([entry["ii"] for entry in update_trace]).astype(np.int64, copy=False)
    arrays["update_trace_jj"] = np.concatenate([entry["jj"] for entry in update_trace]).astype(np.int64, copy=False)
    arrays["update_trace_kk"] = np.concatenate([entry["kk"] for entry in update_trace]).astype(np.int64, copy=False)
    arrays["update_trace_target"] = np.concatenate(
        [entry["target"] for entry in update_trace], axis=1
    ).astype(np.float32, copy=False)
    arrays["update_trace_weight"] = np.concatenate(
        [entry["weight"] for entry in update_trace], axis=1
    ).astype(np.float32, copy=False)
    pose_counts = np.array([entry["post_ba_poses"].shape[0] for entry in update_trace], dtype=np.int64)
    pose_offsets = np.zeros((len(update_trace) + 1,), dtype=np.int64)
    pose_offsets[1:] = np.cumsum(pose_counts, dtype=np.int64)
    patch_counts = np.array([entry["post_ba_patch_depths"].shape[0] for entry in update_trace], dtype=np.int64)
    patch_offsets = np.zeros((len(update_trace) + 1,), dtype=np.int64)
    patch_offsets[1:] = np.cumsum(patch_counts, dtype=np.int64)
    arrays["update_trace_post_ba_pose_counts"] = pose_counts
    arrays["update_trace_post_ba_pose_offsets"] = pose_offsets
    arrays["update_trace_post_ba_patch_counts"] = patch_counts
    arrays["update_trace_post_ba_patch_offsets"] = patch_offsets
    arrays["update_trace_post_ba_poses"] = np.concatenate(
        [entry["post_ba_poses"] for entry in update_trace], axis=0
    ).astype(np.float32, copy=False)
    arrays["update_trace_post_ba_patch_depths"] = np.concatenate(
        [entry["post_ba_patch_depths"] for entry in update_trace], axis=0
    ).astype(np.float32, copy=False)
    append_ba_debug_trace(arrays, update_trace)
    return arrays


def write_metadata(
    output_root: Path,
    args: argparse.Namespace,
    tracker_cfg: TrackerConfig,
    source_paths: list[Path],
    generated_names: list[str],
    src_width: int,
    src_height: int,
    dst_width: int,
    dst_height: int,
    intrinsics: np.ndarray,
    distortion: np.ndarray | None,
    centers_manifest_path: Path,
    bootstrap_depth_manifest_path: Path,
) -> None:
    metadata = {
        "format": "dpvo_python_testdata_v1",
        "weights": str(args.weights),
        "source_images_dir": str(args.images),
        "source_calib": str(args.calib),
        "frame_start": args.frame_start,
        "frame_count": args.frame_count,
        "frame_step": args.frame_step,
        "seed": args.seed,
        "images_are_preprocessed_bgr": True,
        "undistortion_applied": bool(distortion is not None and distortion.size > 0),
        "images_are_aligned_to_multiple_of_16": True,
        "source_frame_paths": [str(path) for path in source_paths],
        "generated_frame_files": generated_names,
        "source_distortion_coefficients": [] if distortion is None else [float(value) for value in distortion.tolist()],
        "centers_manifest": str(centers_manifest_path),
        "bootstrap_depth_manifest": str(bootstrap_depth_manifest_path),
        "resize": {
            "source_width": src_width,
            "source_height": src_height,
            "output_width": dst_width,
            "output_height": dst_height,
            "scale_x": dst_width / float(src_width),
            "scale_y": dst_height / float(src_height),
        },
        "input_intrinsics": [float(value) for value in intrinsics.tolist()],
        "tracker_config": asdict(tracker_cfg),
        "dump_state": bool(args.dump_state),
        "ba_debug_update_index": int(args.ba_debug_update_index),
        "ba_debug_patches": parse_ba_debug_patches(args.ba_debug_patches),
    }
    (output_root / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    ba_debug_patches = parse_ba_debug_patches(args.ba_debug_patches)
    np.random.seed(args.seed)
    configure_determinism(args.seed)

    image_paths = collect_image_paths(args.images, args.frame_start, args.frame_count, args.frame_step)
    source_intrinsics, distortion = load_calibration(args.calib)
    tracker_cfg = build_tracker_config(args)

    image_dir, golden_dir = prepare_output_root(args.output_root)
    frames, frame_names, intrinsics, src_width, src_height, dst_width, dst_height = preprocess_frames(
        image_paths=image_paths,
        source_intrinsics=source_intrinsics,
        distortion=distortion,
        output_root=image_dir,
        width_override=args.width,
        height_override=args.height,
        max_long_edge=args.max_long_edge,
    )

    (
        slam,
        poses,
        tstamps,
        points,
        colors,
        centers_by_frame,
        bootstrap_depths_by_frame,
        update_trace,
        update_parity_cases,
    ) = run_tracker(
        args.weights,
        frames,
        intrinsics,
        tracker_cfg,
        args.ba_debug_update_index,
        ba_debug_patches,
    )
    active_tstamps = slam.pg.tstamps_[: slam.n].astype(np.int64, copy=False)
    active_patch_frames = slam.ix[: slam.m].detach().cpu().numpy().astype(np.int64, copy=False)
    centers_manifest_path = write_centers_manifest(args.output_root, centers_by_frame)
    bootstrap_depth_manifest_path = write_vector_manifest(
        output_root=args.output_root,
        subdir_name="bootstrap_depths",
        manifest_name="bootstrap_depth_manifest.txt",
        file_prefix="bootstrap_depths",
        vectors_by_frame=bootstrap_depths_by_frame,
    )
    write_adjusted_calibration(args.output_root / "calib.txt", intrinsics)
    write_metadata(
        output_root=args.output_root,
        args=args,
        tracker_cfg=tracker_cfg,
        source_paths=image_paths,
        generated_names=frame_names,
        src_width=src_width,
        src_height=src_height,
        dst_width=dst_width,
        dst_height=dst_height,
        intrinsics=intrinsics,
        distortion=distortion,
        centers_manifest_path=centers_manifest_path,
        bootstrap_depth_manifest_path=bootstrap_depth_manifest_path,
    )

    tensors: OrderedDict[str, np.ndarray] = OrderedDict(
        [
            ("input_tstamps", np.arange(len(frames), dtype=np.int64)),
            ("input_intrinsics", intrinsics.astype(np.float32, copy=False)),
            ("golden_poses", poses.astype(np.float32, copy=False)),
            ("golden_tstamps", tstamps.astype(np.float64, copy=False)),
            ("golden_active_tstamps", active_tstamps),
            ("golden_active_patch_frames", active_patch_frames),
            ("golden_points", points.astype(np.float32, copy=False)),
            ("golden_colors", colors.astype(np.uint8, copy=False)),
            ("golden_pose_count", np.array([poses.shape[0]], dtype=np.int64)),
            ("golden_point_count", np.array([points.shape[0]], dtype=np.int64)),
        ]
    )

    if args.dump_state:
        tensors.update(dump_tracker_state(slam, tracker_cfg))
        tensors.update(dump_update_trace(update_trace))

    write_case(golden_dir, tensors)
    if args.dump_update_parity_cases:
        update_parity_root = write_update_parity_cases(args.output_root, update_parity_cases)
    write_tum_trajectory(golden_dir / "golden_trajectory_tum.txt", poses, tstamps)

    print(f"Generated compact DPVO input sequence under {args.output_root / 'images'}")
    print(f"Wrote adjusted calibration to {args.output_root / 'calib.txt'}")
    print(f"Wrote centers manifest to {centers_manifest_path}")
    print(f"Wrote bootstrap depth manifest to {bootstrap_depth_manifest_path}")
    print(f"Wrote golden tensors under {golden_dir}")
    print(
        "frames=%d size=%dx%d patches_per_frame=%d mixed_precision=%s point_count=%d"
        % (
            len(frames),
            dst_width,
            dst_height,
            tracker_cfg.PATCHES_PER_FRAME,
            str(tracker_cfg.MIXED_PRECISION).lower(),
            int(points.shape[0]),
        )
    )
    if args.dump_state:
        print(f"Included final DPVO patch-graph state tensors and {len(update_trace)} update trace snapshots")
    if args.dump_update_parity_cases:
        print(f"Wrote {len(update_parity_cases)} update parity cases under {update_parity_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
