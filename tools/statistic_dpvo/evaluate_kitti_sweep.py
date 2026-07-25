#!/usr/bin/env python3
"""Evaluate one DPVO configuration on KITTI odometry.

This mirrors evaluate_euroc_sweep.py's command-line style: one invocation runs
one config over selected KITTI sequences and writes machine-readable JSON/CSV.

Resolution modes:
  * native: use the KITTI image's original size, cropped down to multiples of 16;
  * low: scale each original KITTI image by the same factors as 480x640 -> 320x416,
    then round down to multiples of 16;
  * explicit --height/--width: resize every frame to that exact 16-aligned size.

Metrics follow the KITTI odometry protocol: relative pose error over all possible
subsequences of length 100, 200, ..., 800 meters. Translation is reported in
percent, rotation in deg/m and deg/100m.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
from pathlib import Path
from typing import Iterable, Iterator, Sequence

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = None


DPVO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DPVO_ROOT))

KITTI_SCENES = [f"{index:02d}" for index in range(11)]
DEFAULT_SEGMENT_LENGTHS = [100, 200, 300, 400, 500, 600, 700, 800]
LOW_HEIGHT_SCALE = 320.0 / 480.0
LOW_WIDTH_SCALE = 416.0 / 640.0
ALIGNMENT = 16


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", type=Path, default=DPVO_ROOT / "dpvo.pth")
    parser.add_argument("--config", type=Path, default=DPVO_ROOT / "config" / "default.yaml")
    parser.add_argument("--kittidir", type=Path, default=DPVO_ROOT / "datasets" / "KITTI")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-sequence-csv", type=Path)
    parser.add_argument("--image-folder", default="image_2", help="KITTI camera image folder. Default: image_2.")
    parser.add_argument(
        "--resolution",
        choices=("native", "low"),
        default="native",
        help=(
            "native keeps the original KITTI image size, aligned down to 16; "
            "low applies the 480x640 -> 320x416 scale ratio, then aligns down to 16."
        ),
    )
    parser.add_argument("--height", type=int, default=0, help="Explicit resized height. Must be used with --width.")
    parser.add_argument("--width", type=int, default=0, help="Explicit resized width. Must be used with --height.")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--backend-thresh", type=float, default=32.0)
    parser.add_argument("--segment-lengths", nargs="+", type=float, default=DEFAULT_SEGMENT_LENGTHS)
    parser.add_argument("--scenes", nargs="+", choices=KITTI_SCENES, default=KITTI_SCENES)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--save-trajectory", action="store_true")
    parser.add_argument("--align", action="store_true", help="Align estimated trajectory to GT before KITTI RPE.")
    parser.add_argument(
        "--correct-scale",
        action="store_true",
        help="Correct global scale before KITTI RPE. Useful for monocular scale comparison.",
    )
    parser.add_argument("--opts", nargs="+", default=[])
    parser.add_argument("--no-progress", action="store_true", help="Disable frame progress output.")
    return parser.parse_args(argv)


def require_runtime_dependencies() -> None:
    missing: list[str] = []
    for module in ("cv2", "evo", "numpy", "torch"):
        try:
            __import__(module)
        except ModuleNotFoundError:
            missing.append(module)
    if missing:
        raise ModuleNotFoundError("missing DPVO evaluation dependency/dependencies: " + ", ".join(missing))


def validate_args(args: argparse.Namespace) -> None:
    if (args.height == 0) != (args.width == 0):
        raise ValueError("--height and --width must be provided together")
    if args.height < 0 or args.width < 0:
        raise ValueError("--height and --width cannot be negative")
    if args.height and (args.height % ALIGNMENT or args.width % ALIGNMENT):
        raise ValueError("explicit --height and --width must both be divisible by 16")
    if args.stride <= 0:
        raise ValueError("--stride must be positive")
    if args.trials <= 0:
        raise ValueError("--trials must be positive")
    if not args.network.exists():
        raise FileNotFoundError(f"DPVO network checkpoint does not exist: {args.network}")
    if not args.config.exists():
        raise FileNotFoundError(f"DPVO config does not exist: {args.config}")
    if not args.kittidir.exists():
        raise FileNotFoundError(f"KITTI directory does not exist: {args.kittidir}")
    for scene in args.scenes:
        sequence_dir = args.kittidir / "dataset" / "sequences" / scene
        image_dir = sequence_dir / args.image_folder
        if not (sequence_dir / "calib.txt").exists():
            raise FileNotFoundError(f"KITTI calibration file does not exist: {sequence_dir / 'calib.txt'}")
        if not any(image_dir.glob("*.png")):
            raise FileNotFoundError(f"no KITTI images found under {image_dir}")
        pose_file = args.kittidir / "dataset" / "poses" / f"{scene}.txt"
        if not pose_file.exists():
            raise FileNotFoundError(f"KITTI pose file does not exist: {pose_file}")


def release_cuda_cache() -> None:
    gc.collect()
    try:
        import torch
    except ModuleNotFoundError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def read_calib_file(filepath: Path) -> dict[str, "np.ndarray"]:
    import numpy as np

    data: dict[str, np.ndarray] = {}
    with filepath.open("r") as handle:
        for line in handle:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            try:
                data[key] = np.array([float(item) for item in value.split()], dtype=np.float64)
            except ValueError:
                continue
    return data


def projection_key_for_image_folder(image_folder: str) -> str:
    suffix = image_folder.rsplit("_", 1)[-1]
    if suffix.isdigit():
        return f"P{int(suffix)}"
    return "P2"


def align_down(value: float, multiple: int = ALIGNMENT) -> int:
    aligned = int(value) // multiple * multiple
    return max(multiple, aligned)


def target_size(args: argparse.Namespace, src_height: int, src_width: int) -> tuple[int, int, str]:
    if args.height and args.width:
        return args.height, args.width, "explicit"
    if args.resolution == "low":
        return (
            align_down(src_height * LOW_HEIGHT_SCALE),
            align_down(src_width * LOW_WIDTH_SCALE),
            "low",
        )
    return align_down(src_height), align_down(src_width), "native"


def iter_images(
    args: argparse.Namespace,
    scene: str,
    progress_label: str,
    show_progress: bool,
) -> Iterator[tuple[int, "np.ndarray", "np.ndarray", int, int]]:
    import cv2
    import numpy as np

    sequence_dir = args.kittidir / "dataset" / "sequences" / scene
    image_dir = sequence_dir / args.image_folder
    image_paths = sorted(image_dir.glob("*.png"))[:: args.stride]
    if not image_paths:
        raise FileNotFoundError(f"no KITTI images found under {image_dir}")

    calib = read_calib_file(sequence_dir / "calib.txt")
    projection_key = projection_key_for_image_folder(args.image_folder)
    projection = calib.get(projection_key)
    if projection is None:
        projection = calib.get("P2")
    if projection is None:
        projection = calib.get("P0")
    if projection is None:
        raise KeyError(f"missing {projection_key}/P2/P0 calibration in {sequence_dir / 'calib.txt'}")
    intrinsics = projection[[0, 5, 2, 6]].astype(np.float64, copy=True)

    paths: Iterable[Path] = image_paths
    if show_progress and tqdm is not None:
        paths = tqdm(image_paths, desc=progress_label, unit="frame", dynamic_ncols=True, leave=True)
    elif show_progress:
        print(f"{progress_label}: {len(image_paths)} frame(s)")

    for t, image_path in enumerate(paths):
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"failed to read image: {image_path}")
        src_height, src_width = image.shape[:2]
        height, width, _ = target_size(args, src_height, src_width)
        scaled_intrinsics = intrinsics.copy()
        if (src_height, src_width) != (height, width):
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            scaled_intrinsics[0] *= width / src_width
            scaled_intrinsics[2] *= width / src_width
            scaled_intrinsics[1] *= height / src_height
            scaled_intrinsics[3] *= height / src_height
        else:
            image = image[:height, :width]
        if show_progress and tqdm is None and (t + 1) % 100 == 0:
            print(f"{progress_label}: {t + 1}/{len(image_paths)} frame(s)")
        yield t, image, scaled_intrinsics.astype("float32"), height, width

    if show_progress and tqdm is None:
        print(f"{progress_label}: done")


def run_sequence(
    args: argparse.Namespace,
    cfg: object,
    network: Path,
    scene: str,
    progress_label: str,
    show_progress: bool,
) -> tuple["np.ndarray", "np.ndarray", list[tuple[int, int]]]:
    import numpy as np
    import torch

    from dpvo.dpvo import DPVO

    slam = None
    timestamps: list[int] = []
    sizes: list[tuple[int, int]] = []
    with torch.no_grad():
        for t, image, intrinsics, height, width in iter_images(args, scene, progress_label, show_progress):
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).cuda()
            intrinsics_tensor = torch.from_numpy(intrinsics).cuda()
            if slam is None:
                slam = DPVO(cfg, str(network), ht=height, wd=width, viz=False)
            slam(t, image_tensor, intrinsics_tensor)
            timestamps.append(t * args.stride)
            sizes.append((height, width))
        if slam is None:
            raise RuntimeError(f"no frames processed for KITTI sequence {scene}")
        poses, _ = slam.terminate()
    return poses, np.array(timestamps, dtype=np.float64), sizes


def trajectory_to_matrices(traj: "PoseTrajectory3D") -> list["np.ndarray"]:
    import numpy as np

    matrices: list[np.ndarray] = []
    for position, quat_wxyz in zip(traj.positions_xyz, traj.orientations_quat_wxyz):
        quat = np.asarray(quat_wxyz, dtype=np.float64)
        quat = quat / np.linalg.norm(quat)
        w, x, y, z = quat
        rotation = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation
        transform[:3, 3] = position
        matrices.append(transform)
    return matrices


def trajectory_distances(poses: Sequence["np.ndarray"]) -> list[float]:
    import numpy as np

    distances = [0.0]
    for index in range(1, len(poses)):
        delta = poses[index][:3, 3] - poses[index - 1][:3, 3]
        distances.append(distances[-1] + float(np.linalg.norm(delta)))
    return distances


def rotation_error(transform: "np.ndarray") -> float:
    trace_value = float(transform[0, 0] + transform[1, 1] + transform[2, 2])
    cosine = max(-1.0, min(1.0, (trace_value - 1.0) / 2.0))
    return math.acos(cosine)


def translation_error(transform: "np.ndarray") -> float:
    import numpy as np

    return float(np.linalg.norm(transform[:3, 3]))


def last_frame_from_segment_length(distances: Sequence[float], first_frame: int, length: float) -> int | None:
    target = distances[first_frame] + length
    for index in range(first_frame, len(distances)):
        if distances[index] > target:
            return index
    return None


def kitti_relative_errors(
    poses_ref: Sequence["np.ndarray"],
    poses_est: Sequence["np.ndarray"],
    segment_lengths: Sequence[float],
) -> list[dict[str, float]]:
    import numpy as np

    distances = trajectory_distances(poses_ref)
    errors: list[dict[str, float]] = []
    step_size = 10
    pose_count = min(len(poses_ref), len(poses_est))
    for first_frame in range(0, pose_count, step_size):
        for length in segment_lengths:
            last_frame = last_frame_from_segment_length(distances, first_frame, length)
            if last_frame is None or last_frame >= pose_count:
                continue

            pose_delta_ref = np.linalg.inv(poses_ref[first_frame]) @ poses_ref[last_frame]
            pose_delta_est = np.linalg.inv(poses_est[first_frame]) @ poses_est[last_frame]
            error_transform = np.linalg.inv(pose_delta_est) @ pose_delta_ref
            errors.append(
                {
                    "first_frame": float(first_frame),
                    "last_frame": float(last_frame),
                    "length_m": float(length),
                    "translation_error_fraction": translation_error(error_transform) / float(length),
                    "rotation_error_rad_per_m": rotation_error(error_transform) / float(length),
                }
            )
    return errors


def evaluate_scene(args: argparse.Namespace, scene: str, trial: int) -> tuple[float, float, int, list[tuple[int, int]]]:
    import numpy as np
    import torch
    from evo.core import sync
    from evo.core.trajectory import PoseTrajectory3D
    from evo.tools import file_interface

    from dpvo.config import cfg as base_cfg

    cfg = base_cfg.clone()
    cfg.merge_from_file(str(args.config))
    cfg.BACKEND_THRESH = args.backend_thresh
    cfg.merge_from_list(args.opts)
    torch.manual_seed(args.seed + trial)

    try:
        traj_est, timestamps, sizes = run_sequence(
            args,
            cfg,
            args.network,
            scene,
            f"{args.config.stem} {scene} trial {trial + 1}/{args.trials}",
            not args.no_progress,
        )
        poses_ref = file_interface.read_kitti_poses_file(str(args.kittidir / "dataset" / "poses" / f"{scene}.txt"))
        traj_est_evo = PoseTrajectory3D(
            positions_xyz=traj_est[:, :3],
            orientations_quat_wxyz=traj_est[:, [6, 3, 4, 5]],
            timestamps=timestamps[: len(traj_est)],
        )
        traj_ref = PoseTrajectory3D(
            positions_xyz=poses_ref.positions_xyz,
            orientations_quat_wxyz=poses_ref.orientations_quat_wxyz,
            timestamps=np.arange(poses_ref.num_poses, dtype=np.float64),
        )
        traj_ref, traj_est_evo = sync.associate_trajectories(traj_ref, traj_est_evo)
        if args.align or args.correct_scale:
            traj_est_evo.align(
                traj_ref,
                correct_scale=args.correct_scale,
                correct_only_scale=args.correct_scale and not args.align,
            )

        errors = kitti_relative_errors(
            trajectory_to_matrices(traj_ref),
            trajectory_to_matrices(traj_est_evo),
            args.segment_lengths,
        )
        if not errors:
            raise RuntimeError(f"no valid KITTI error segments for sequence {scene}")

        translation_error_percent = 100.0 * float(
            np.mean([item["translation_error_fraction"] for item in errors])
        )
        rotation_error_deg_per_m = (180.0 / math.pi) * float(
            np.mean([item["rotation_error_rad_per_m"] for item in errors])
        )

        if args.plot:
            from dpvo.plot_utils import plot_trajectory

            output_dir = args.output.parent / "trajectory_plots"
            output_dir.mkdir(parents=True, exist_ok=True)
            plot_trajectory(
                traj_est_evo,
                traj_ref,
                f"KITTI {scene} Trial #{trial + 1}",
                str(output_dir / f"kitti_{args.config.stem}_{scene}_trial{trial + 1:02d}.pdf"),
                align=args.align,
                correct_scale=args.correct_scale,
            )

        if args.save_trajectory:
            output_dir = args.output.parent / "saved_trajectories"
            output_dir.mkdir(parents=True, exist_ok=True)
            file_interface.write_tum_trajectory_file(
                str(output_dir / f"KITTI_{args.config.stem}_{scene}_trial{trial + 1:02d}.txt"),
                traj_est_evo,
            )

        return translation_error_percent, rotation_error_deg_per_m, len(errors), sizes
    finally:
        release_cuda_cache()


def write_scene_csv(path: Path, scene_rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "scene",
        "trial_count",
        "median_translation_error_percent",
        "mean_translation_error_percent",
        "median_rotation_error_deg_per_m",
        "mean_rotation_error_deg_per_m",
        "median_rotation_error_deg_per_100m",
        "mean_rotation_error_deg_per_100m",
        "trial_translation_error_percent",
        "trial_rotation_error_deg_per_m",
        "num_segments",
        "image_sizes",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(scene_rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        require_runtime_dependencies()
        validate_args(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    import numpy as np

    scene_rows: list[dict[str, object]] = []
    for scene_index, scene in enumerate(args.scenes, start=1):
        trial_translation: list[float] = []
        trial_rotation: list[float] = []
        trial_segments: list[int] = []
        seen_sizes: set[tuple[int, int]] = set()
        for trial in range(args.trials):
            print(
                f"candidate={args.config.stem} scene={scene_index}/{len(args.scenes)} {scene} "
                f"trial={trial + 1}/{args.trials}",
                flush=True,
            )
            t_err, r_err, segment_count, sizes = evaluate_scene(args, scene, trial)
            trial_translation.append(t_err)
            trial_rotation.append(r_err)
            trial_segments.append(segment_count)
            seen_sizes.update(sizes)
            print(
                f"{scene} trial {trial + 1}: "
                f"t_err={t_err:.6f}% r_err={r_err:.8f} deg/m "
                f"segments={segment_count}",
                flush=True,
            )

        median_rotation = float(np.median(trial_rotation))
        mean_rotation = float(np.mean(trial_rotation))
        scene_rows.append(
            {
                "scene": scene,
                "trial_count": args.trials,
                "median_translation_error_percent": float(np.median(trial_translation)),
                "mean_translation_error_percent": float(np.mean(trial_translation)),
                "median_rotation_error_deg_per_m": median_rotation,
                "mean_rotation_error_deg_per_m": mean_rotation,
                "median_rotation_error_deg_per_100m": 100.0 * median_rotation,
                "mean_rotation_error_deg_per_100m": 100.0 * mean_rotation,
                "trial_translation_error_percent": trial_translation,
                "trial_rotation_error_deg_per_m": trial_rotation,
                "num_segments": int(np.median(trial_segments)),
                "image_sizes": sorted([f"{height}x{width}" for height, width in seen_sizes]),
            }
        )

    avg_t = float(np.mean([row["median_translation_error_percent"] for row in scene_rows]))
    avg_r = float(np.mean([row["median_rotation_error_deg_per_m"] for row in scene_rows]))
    payload = {
        "config": str(args.config),
        "network": str(args.network),
        "kittidir": str(args.kittidir),
        "image_folder": args.image_folder,
        "resolution": args.resolution,
        "height": args.height if args.height else None,
        "width": args.width if args.width else None,
        "low_height_scale": LOW_HEIGHT_SCALE,
        "low_width_scale": LOW_WIDTH_SCALE,
        "stride": args.stride,
        "trials": args.trials,
        "segment_lengths": args.segment_lengths,
        "align": args.align,
        "correct_scale": args.correct_scale,
        "scenes": scene_rows,
        "avg_translation_error_percent": avg_t,
        "avg_rotation_error_deg_per_m": avg_r,
        "avg_rotation_error_deg_per_100m": 100.0 * avg_r,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    if args.per_sequence_csv:
        write_scene_csv(args.per_sequence_csv, scene_rows)
    print(f"AVG translation: {avg_t:.6f}%")
    print(f"AVG rotation: {avg_r:.8f} deg/m ({100.0 * avg_r:.6f} deg/100m)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
