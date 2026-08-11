#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
TOOLS_DIR = SCRIPT_DIR.parent
REPO_ROOT = TOOLS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from generate_dpvo_python_testdata import (  # noqa: E402
    build_tracker_config,
    collect_image_paths,
    configure_determinism,
    load_calibration,
    preprocess_frames,
    write_case,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Python DPVO and export standalone bundle-adjustment benchmark "
            "cases by intercepting dpvo.fastba.BA calls."
        )
    )
    parser.add_argument("--weights", type=Path, default=REPO_ROOT / "dpvo.pth")
    parser.add_argument(
        "--images",
        type=Path,
        default=REPO_ROOT / "datasets" / "EUROC" / "MH_01_easy" / "mav0" / "cam0" / "data",
    )
    parser.add_argument("--calib", type=Path, default=REPO_ROOT / "calib" / "euroc.txt")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "testdata" / "ba_benchmark_euroc_mh01",
    )
    parser.add_argument("--frame-start", type=int, default=1)
    parser.add_argument("--frame-count", type=int, default=32)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--max-long-edge", type=int, default=256)
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--height", type=int, default=0)
    parser.add_argument("--patches-per-frame", type=int, default=32)
    parser.add_argument("--buffer-size", type=int, default=64)
    parser.add_argument("--removal-window", type=int, default=12)
    parser.add_argument("--optimization-window", type=int, default=8)
    parser.add_argument("--patch-lifetime", type=int, default=10)
    parser.add_argument("--keyframe-index", type=int, default=4)
    parser.add_argument("--keyframe-thresh", type=float, default=12.5)
    parser.add_argument("--motion-model", type=str, default="DAMPED_LINEAR")
    parser.add_argument("--motion-damping", type=float, default=0.5)
    parser.add_argument("--centroid-sel-strat", choices=["RANDOM", "GRADIENT_BIAS"], default="RANDOM")
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument(
        "--case-index",
        type=int,
        default=-1,
        help="Eligible BA call to export. -1 exports the last eligible call.",
    )
    parser.add_argument("--all-cases", action="store_true", help="Export every eligible BA call.")
    parser.add_argument(
        "--capture-global",
        action="store_true",
        help="Also export global BA calls. Local BA calls are exported by default.",
    )
    parser.add_argument(
        "--include-terminate-updates",
        action="store_true",
        help="Also run slam.terminate(), which performs extra final BA updates.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def tensor_to_numpy(tensor: Any, dtype: np.dtype) -> np.ndarray:
    return tensor.detach().cpu().contiguous().numpy().astype(dtype, copy=False)


def scalar_i64(value: int) -> np.ndarray:
    return np.array([int(value)], dtype=np.int64)


def scalar_f32(value: float) -> np.ndarray:
    return np.array([float(value)], dtype=np.float32)


def make_ba_case_tensors(
    *,
    poses_before: Any,
    patches_before: Any,
    intrinsics_before: Any,
    target: Any,
    weight: Any,
    ii: Any,
    jj: Any,
    kk: Any,
    t0: int,
    t1: int,
    patches_per_frame: int,
    iterations: int,
    poses_after: Any,
    patches_after: Any,
    ba_call_index: int,
    is_global: bool,
) -> OrderedDict[str, np.ndarray]:
    import torch

    ii_flat = ii.detach().reshape(-1).long()
    jj_flat = jj.detach().reshape(-1).long()
    kk_flat = kk.detach().reshape(-1).long()
    max_patch = int(kk_flat.max().item()) if int(kk_flat.numel()) > 0 else -1
    patch_limit = max(int(t1) * int(patches_per_frame), max_patch + 1)

    poses_before_flat = poses_before.detach().reshape(-1, 7)[: int(t1)].clone().float()
    poses_after_flat = poses_after.detach().reshape(-1, 7)[: int(t1)].clone().float()
    intrinsics_flat = intrinsics_before.detach().reshape(-1, 4)[: int(t1)].clone().float()
    patches_before_flat = patches_before.detach().reshape(
        -1,
        int(patches_before.shape[-3]),
        int(patches_before.shape[-2]),
        int(patches_before.shape[-1]),
    )[:patch_limit].clone().float()
    patches_after_flat = patches_after.detach().reshape(
        -1,
        int(patches_after.shape[-3]),
        int(patches_after.shape[-2]),
        int(patches_after.shape[-1]),
    )[:patch_limit].clone().float()

    if patches_before_flat.shape[1:] != (3, 3, 3):
        raise ValueError(
            "dpvo_runner BA expects patches [P,3,3,3], "
            f"got {tuple(patches_before_flat.shape)}"
        )
    if int(ii_flat.numel()) > 0:
        max_frame = max(int(ii_flat.max().item()), int(jj_flat.max().item()))
        if max_frame >= int(t1):
            raise ValueError(f"BA edge references frame {max_frame}, but t1={t1}")
        if max_patch >= int(patches_before_flat.shape[0]):
            raise ValueError(
                f"BA edge references patch {max_patch}, "
                f"but exported patches={patches_before_flat.shape[0]}"
            )

    cx = float(intrinsics_flat[0, 2].item())
    cy = float(intrinsics_flat[0, 3].item())
    bounds = np.array([-64.0, -64.0, 2.0 * cx + 64.0, 2.0 * cy + 64.0], dtype=np.float32)

    del torch
    return OrderedDict(
        [
            ("poses", tensor_to_numpy(poses_before_flat, np.float32)),
            ("intrinsics", tensor_to_numpy(intrinsics_flat, np.float32)),
            ("patches", tensor_to_numpy(patches_before_flat, np.float32)),
            ("ii", tensor_to_numpy(ii_flat, np.int64)),
            ("jj", tensor_to_numpy(jj_flat, np.int64)),
            ("kk", tensor_to_numpy(kk_flat, np.int64)),
            ("target", tensor_to_numpy(target.detach().reshape(1, -1, 2).float(), np.float32)),
            ("weight", tensor_to_numpy(weight.detach().reshape(1, -1, 2).float(), np.float32)),
            ("bounds", bounds),
            ("fixed_pose_count", scalar_i64(t0)),
            ("iterations", scalar_i64(iterations)),
            ("ep", scalar_f32(1.0)),
            ("patches_per_frame", scalar_i64(patches_per_frame)),
            ("ba_t1", scalar_i64(t1)),
            ("ba_call_index", scalar_i64(ba_call_index)),
            ("is_global", scalar_i64(1 if is_global else 0)),
            ("golden_poses", tensor_to_numpy(poses_after_flat, np.float32)),
            ("golden_patches", tensor_to_numpy(patches_after_flat, np.float32)),
        ]
    )


def prepare_output_root(output_root: Path, overwrite: bool) -> Path:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"{output_root} already exists; pass --overwrite to replace it")
        shutil.rmtree(output_root)
    image_dir = output_root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    return image_dir


def main() -> int:
    args = parse_args()
    configure_determinism(args.seed)

    import torch
    import dpvo.dpvo as dpvo_module
    from dpvo.dpvo import DPVO

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required because Python DPVO and dpvo.fastba run on GPU")
    torch.cuda.set_device(args.cuda_device)

    image_paths = collect_image_paths(args.images, args.frame_start, args.frame_count, args.frame_step)
    source_intrinsics, distortion = load_calibration(args.calib)
    tracker_cfg = build_tracker_config(args)
    image_dir = prepare_output_root(args.output_root, args.overwrite)
    frames, frame_names, intrinsics, src_width, src_height, dst_width, dst_height = preprocess_frames(
        image_paths=image_paths,
        source_intrinsics=source_intrinsics,
        distortion=distortion,
        output_root=image_dir,
        width_override=args.width,
        height_override=args.height,
        max_long_edge=args.max_long_edge,
    )

    slam = DPVO(tracker_cfg, str(args.weights), ht=dst_height, wd=dst_width, viz=False)
    original_fastba_ba = dpvo_module.fastba.BA
    captured_cases: list[tuple[int, OrderedDict[str, np.ndarray]]] = []
    eligible_ba_index = 0

    def recording_fastba_ba(
        poses,
        patches,
        intrinsics_tensor,
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
        nonlocal eligible_ba_index
        is_global = bool(eff_impl)
        eligible = args.capture_global or not is_global
        current_index = eligible_ba_index if eligible else -1
        if eligible:
            eligible_ba_index += 1

        poses_before = poses.detach().clone()
        patches_before = patches.detach().clone()
        intrinsics_before = intrinsics_tensor.detach().clone()
        target_before = target.detach().clone()
        weight_before = weight.detach().clone()
        ii_before = ii.detach().clone()
        jj_before = jj.detach().clone()
        kk_before = kk.detach().clone()

        result = original_fastba_ba(
            poses,
            patches,
            intrinsics_tensor,
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
        torch.cuda.synchronize()

        if eligible:
            tensors = make_ba_case_tensors(
                poses_before=poses_before,
                patches_before=patches_before,
                intrinsics_before=intrinsics_before,
                target=target_before,
                weight=weight_before,
                ii=ii_before,
                jj=jj_before,
                kk=kk_before,
                t0=int(t0),
                t1=int(t1),
                patches_per_frame=int(M),
                iterations=int(iterations),
                poses_after=poses,
                patches_after=patches,
                ba_call_index=current_index,
                is_global=is_global,
            )
            if args.all_cases:
                captured_cases.append((current_index, tensors))
            elif args.case_index >= 0:
                if current_index == args.case_index:
                    captured_cases.append((current_index, tensors))
            else:
                captured_cases[:] = [(current_index, tensors)]

        return result

    dpvo_module.fastba.BA = recording_fastba_ba
    try:
        with torch.no_grad():
            for tstamp, frame in enumerate(frames):
                image_tensor = torch.from_numpy(frame).permute(2, 0, 1).cuda(non_blocking=False)
                intrinsics_tensor = torch.from_numpy(intrinsics.copy()).cuda(non_blocking=False)
                slam(tstamp, image_tensor, intrinsics_tensor)
            if args.include_terminate_updates:
                slam.terminate()
    finally:
        dpvo_module.fastba.BA = original_fastba_ba

    if not captured_cases:
        raise RuntimeError(
            "No BA cases were captured. Try a larger --frame-count, --all-cases, "
            "or --include-terminate-updates."
        )

    case_records: list[dict[str, Any]] = []
    for output_index, (ba_index, tensors) in enumerate(captured_cases):
        case_dir = args.output_root / f"case_{output_index:03d}"
        case_dir.mkdir(parents=True, exist_ok=True)
        write_case(case_dir, tensors)
        case_records.append(
            {
                "case_dir": str(case_dir),
                "eligible_ba_index": int(ba_index),
                "edges": int(tensors["ii"].shape[0]),
                "poses": int(tensors["poses"].shape[0]),
                "patches": int(tensors["patches"].shape[0]),
                "fixed_pose_count": int(tensors["fixed_pose_count"][0]),
                "iterations": int(tensors["iterations"][0]),
                "is_global": bool(int(tensors["is_global"][0])),
            }
        )

    metadata = {
        "format": "dpvo_runner_ba_benchmark_v1",
        "weights": str(args.weights),
        "source_images_dir": str(args.images),
        "source_calib": str(args.calib),
        "frame_start": int(args.frame_start),
        "frame_count": int(args.frame_count),
        "frame_step": int(args.frame_step),
        "seed": int(args.seed),
        "cuda_device": int(args.cuda_device),
        "generated_frame_files": frame_names,
        "source_frame_paths": [str(path) for path in image_paths],
        "resize": {
            "source_width": int(src_width),
            "source_height": int(src_height),
            "output_width": int(dst_width),
            "output_height": int(dst_height),
        },
        "input_intrinsics": [float(value) for value in intrinsics.tolist()],
        "tracker_config": asdict(tracker_cfg),
        "case_index": int(args.case_index),
        "all_cases": bool(args.all_cases),
        "capture_global": bool(args.capture_global),
        "include_terminate_updates": bool(args.include_terminate_updates),
        "captured_case_count": len(case_records),
        "cases": case_records,
    }
    (args.output_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_root / "cases.txt").write_text(
        "\n".join(record["case_dir"] for record in case_records) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {len(case_records)} BA benchmark case(s) under {args.output_root}")
    for record in case_records:
        print(
            "case={case_dir} ba_index={eligible_ba_index} edges={edges} "
            "poses={poses} patches={patches} fixed={fixed_pose_count} "
            "iterations={iterations} global={is_global}".format(**record)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
