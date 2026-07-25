#!/usr/bin/env python3
"""Evaluate one DPVO algorithm-parameter candidate on EuRoC.

This is a machine-readable variant of evaluate_euroc.py for the
statistic_dpvo sweep flow.  It keeps the same DPVO class and ATE metric, but
adds optional fixed H,W resizing so image-size candidates can be evaluated
without editing dpvo/stream.py.
"""

from __future__ import annotations

import argparse
import csv
import gc
import glob
import json
import os
import sys
from pathlib import Path
from typing import Iterable, Iterator, Sequence

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = None


DPVO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DPVO_ROOT))

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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", type=Path, default=DPVO_ROOT / "dpvo.pth")
    parser.add_argument("--config", type=Path, default=DPVO_ROOT / "config" / "default.yaml")
    parser.add_argument("--eurocdir", type=Path, default=DPVO_ROOT / "datasets" / "EUROC")
    parser.add_argument("--groundtruth-dir", type=Path, default=DPVO_ROOT / "datasets" / "euroc_groundtruth")
    parser.add_argument("--calib", type=Path, default=DPVO_ROOT / "calib" / "euroc.txt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-scene-csv", type=Path)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--backend-thresh", type=float, default=64.0)
    parser.add_argument("--scenes", nargs="+", choices=EUROC_SCENES, default=EUROC_SCENES)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--save-trajectory", action="store_true")
    parser.add_argument("--opts", nargs="+", default=[])
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm/progress output during frame processing.",
    )
    return parser.parse_args(argv)


def require_runtime_dependencies() -> None:
    missing: list[str] = []
    for module in ("cv2", "evo", "numpy", "torch"):
        try:
            __import__(module)
        except ModuleNotFoundError:
            missing.append(module)
    if missing:
        raise ModuleNotFoundError(
            "missing DPVO evaluation dependency/dependencies: " + ", ".join(missing)
        )


def validate_args(args: argparse.Namespace) -> None:
    if args.height <= 0 or args.width <= 0:
        raise ValueError("--height and --width must be positive")
    if args.height % 16 or args.width % 16:
        raise ValueError("--height and --width must both be divisible by 16")
    if args.trials <= 0:
        raise ValueError("--trials must be positive")
    if not args.network.exists():
        raise FileNotFoundError(f"DPVO network checkpoint does not exist: {args.network}")
    if not args.config.exists():
        raise FileNotFoundError(f"DPVO config does not exist: {args.config}")
    if not args.calib.exists():
        raise FileNotFoundError(f"EuRoC calibration file does not exist: {args.calib}")


def release_cuda_cache() -> None:
    gc.collect()
    try:
        import torch
    except ModuleNotFoundError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_calibration(calib_path: Path) -> tuple["np.ndarray", "np.ndarray | None"]:
    import numpy as np

    calib = np.loadtxt(calib_path, delimiter=" ", dtype=np.float64)
    calib = np.atleast_1d(calib).reshape(-1)
    if calib.size < 4:
        raise ValueError(f"expected at least four calibration values in {calib_path}")
    intrinsics = calib[:4].astype(np.float64, copy=True)
    distortion = calib[4:].astype(np.float64, copy=True) if calib.size > 4 else None
    return intrinsics, distortion


def iter_images(
    image_dir: Path,
    calib_path: Path,
    stride: int,
    height: int,
    width: int,
    progress_label: str,
    show_progress: bool,
) -> Iterator[tuple[int, "np.ndarray", "np.ndarray", float]]:
    import cv2
    import numpy as np

    if not image_dir.exists():
        raise FileNotFoundError(f"EuRoC image directory does not exist: {image_dir}")
    image_paths = sorted(
        path
        for pattern in ("*.png", "*.jpg", "*.jpeg")
        for path in image_dir.glob(pattern)
    )[::stride]
    if not image_paths:
        raise FileNotFoundError(f"no images found in {image_dir}")

    intrinsics, distortion = load_calibration(calib_path)
    fx, fy, cx, cy = intrinsics
    K = np.eye(3, dtype=np.float64)
    K[0, 0] = fx
    K[0, 2] = cx
    K[1, 1] = fy
    K[1, 2] = cy

    paths: Iterable[Path] = image_paths
    if show_progress and tqdm is not None:
        paths = tqdm(
            image_paths,
            desc=progress_label,
            unit="frame",
            dynamic_ncols=True,
            leave=True,
        )
    elif show_progress:
        print(f"{progress_label}: {len(image_paths)} frame(s)")

    for t, image_path in enumerate(paths):
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"failed to read image: {image_path}")
        if distortion is not None and distortion.size:
            image = cv2.undistort(image, K, distortion)

        src_height, src_width = image.shape[:2]
        scaled_intrinsics = intrinsics.copy()
        if (src_height, src_width) != (height, width):
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            scaled_intrinsics[0] *= width / src_width
            scaled_intrinsics[2] *= width / src_width
            scaled_intrinsics[1] *= height / src_height
            scaled_intrinsics[3] *= height / src_height

        timestamp = float(image_path.stem)
        if show_progress and tqdm is None and (t + 1) % 100 == 0:
            print(f"{progress_label}: {t + 1}/{len(image_paths)} frame(s)")
        yield t, image, scaled_intrinsics.astype("float32"), timestamp
    if show_progress and tqdm is None:
        print(f"{progress_label}: done")


def run_sequence(
    cfg: object,
    network: Path,
    image_dir: Path,
    calib_path: Path,
    stride: int,
    height: int,
    width: int,
    progress_label: str,
    show_progress: bool,
) -> tuple["np.ndarray", "np.ndarray"]:
    import torch

    from dpvo.dpvo import DPVO

    slam = None
    timestamps: list[float] = []
    with torch.no_grad():
        for t, image, intrinsics, timestamp in iter_images(
            image_dir,
            calib_path,
            stride,
            height,
            width,
            progress_label,
            show_progress,
        ):
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).cuda()
            intrinsics_tensor = torch.from_numpy(intrinsics).cuda()
            if slam is None:
                slam = DPVO(cfg, str(network), ht=image.shape[0], wd=image.shape[1], viz=False)
            slam(t, image_tensor, intrinsics_tensor)
            timestamps.append(timestamp)
        if slam is None:
            raise RuntimeError(f"no frames processed from {image_dir}")
        poses, _ = slam.terminate()
    return poses, __import__("numpy").array(timestamps, dtype="float64")


def evaluate_scene(args: argparse.Namespace, scene: str, trial: int) -> float:
    import numpy as np
    import torch
    import evo.main_ape as main_ape
    from evo.core import sync
    from evo.core.metrics import PoseRelation
    from evo.core.trajectory import PoseTrajectory3D
    from evo.tools import file_interface

    from dpvo.config import cfg

    cfg.merge_from_file(str(args.config))
    cfg.BACKEND_THRESH = args.backend_thresh
    cfg.merge_from_list(args.opts)
    torch.manual_seed(args.seed + trial)

    image_dir = args.eurocdir / scene / "mav0" / "cam0" / "data"
    groundtruth = args.groundtruth_dir / f"{scene}.txt"
    if not groundtruth.exists():
        raise FileNotFoundError(f"ground-truth trajectory does not exist: {groundtruth}")

    try:
        traj_est, timestamps = run_sequence(
            cfg,
            args.network,
            image_dir,
            args.calib,
            args.stride,
            args.height,
            args.width,
            f"{args.config.stem} {scene} trial {trial + 1}/{args.trials}",
            not args.no_progress,
        )
        traj_est_evo = PoseTrajectory3D(
            positions_xyz=traj_est[:, :3],
            orientations_quat_wxyz=traj_est[:, [6, 3, 4, 5]],
            timestamps=timestamps[: len(traj_est)],
        )
        traj_ref = file_interface.read_tum_trajectory_file(str(groundtruth))
        traj_ref, traj_est_evo = sync.associate_trajectories(traj_ref, traj_est_evo)
        result = main_ape.ape(
            traj_ref,
            traj_est_evo,
            est_name="traj",
            pose_relation=PoseRelation.translation_part,
            align=True,
            correct_scale=True,
        )
        ate_score = float(result.stats["rmse"])

        if args.plot:
            from dpvo.plot_utils import plot_trajectory

            output_dir = args.output.parent / "trajectory_plots"
            output_dir.mkdir(parents=True, exist_ok=True)
            plot_trajectory(
                traj_est_evo,
                traj_ref,
                f"EuRoC {scene} Trial #{trial + 1} (ATE: {ate_score:.03f})",
                str(output_dir / f"Euroc_{scene}_Trial{trial + 1:02d}.pdf"),
                align=True,
                correct_scale=True,
            )

        if args.save_trajectory:
            output_dir = args.output.parent / "saved_trajectories"
            output_dir.mkdir(parents=True, exist_ok=True)
            file_interface.write_tum_trajectory_file(
                str(output_dir / f"Euroc_{scene}_Trial{trial + 1:02d}.txt"),
                traj_est_evo,
            )
        return ate_score
    finally:
        release_cuda_cache()


def write_scene_csv(path: Path, scene_rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["scene", "trial_count", "median_ate_m", "mean_ate_m", "trial_ate_m"]
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
        trial_scores = []
        for trial in range(args.trials):
            print(
                f"candidate={args.config.stem} "
                f"scene={scene_index}/{len(args.scenes)} {scene} "
                f"trial={trial + 1}/{args.trials}",
                flush=True,
            )
            ate_score = evaluate_scene(args, scene, trial)
            trial_scores.append(ate_score)
            print(f"{scene} trial {trial + 1}: {ate_score:.6f}")
        scene_rows.append(
            {
                "scene": scene,
                "trial_count": args.trials,
                "median_ate_m": float(np.median(trial_scores)),
                "mean_ate_m": float(np.mean(trial_scores)),
                "trial_ate_m": trial_scores,
            }
        )

    avg_ate = float(np.mean([row["median_ate_m"] for row in scene_rows]))
    payload = {
        "config": str(args.config),
        "network": str(args.network),
        "eurocdir": str(args.eurocdir),
        "height": args.height,
        "width": args.width,
        "stride": args.stride,
        "trials": args.trials,
        "scenes": scene_rows,
        "avg_ate_m": avg_ate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    if args.per_scene_csv:
        write_scene_csv(args.per_scene_csv, scene_rows)
    print(f"AVG: {avg_ate:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
