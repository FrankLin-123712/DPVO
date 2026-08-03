#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
TOOLS_DIR = SCRIPT_DIR.parent
REPO_ROOT = TOOLS_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from dpvo_runner_parity_common import build_sequence_data, tensor_to_numpy, write_case  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate compact DPVO runner parity cases for patchify, "
            "correlation, update, and bundle adjustment."
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
        help="Calibration text file used to scale intrinsics.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "testdata" / "dpvo_runner_parity_small",
        help="Root directory for generated component parity cases.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=144,
        help="Resized image height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=256,
        help="Resized image width.",
    )
    parser.add_argument(
        "--frame-start",
        type=int,
        default=1,
        help="1-based first frame index from the image directory.",
    )
    parser.add_argument(
        "--frame-count",
        type=int,
        default=4,
        help="Number of consecutive frames to use.",
    )
    parser.add_argument(
        "--patches-per-frame",
        type=int,
        default=8,
        help="Number of deterministic patches per frame.",
    )
    return parser.parse_args()


def collect_image_paths(image_dir: Path, frame_start: int, frame_count: int) -> list[Path]:
    image_paths = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if frame_start <= 0:
        raise ValueError("--frame-start must be positive")
    start = frame_start - 1
    end = start + frame_count
    if end > len(image_paths):
        raise ValueError(
            f"Requested frames [{frame_start}, {frame_start + frame_count - 1}] "
            f"but only found {len(image_paths)} images in {image_dir}"
        )
    return image_paths[start:end]


def main() -> int:
    args = parse_args()
    image_paths = collect_image_paths(args.images, args.frame_start, args.frame_count)
    sequence = build_sequence_data(
        weights=args.weights,
        image_paths=image_paths,
        calib_path=args.calib,
        width=args.width,
        height=args.height,
        patches_per_frame=args.patches_per_frame,
    )

    patchify_case = OrderedDict(
        [
            ("image", tensor_to_numpy(sequence.frames[0].image.unsqueeze(0).unsqueeze(0), np.float32)),
            ("centers", tensor_to_numpy(sequence.frames[0].centers, np.float32)),
            ("golden_fmap", tensor_to_numpy(sequence.frames[0].fmap, np.float32)),
            ("golden_imap", tensor_to_numpy(sequence.frames[0].patch_imap, np.float32)),
            ("golden_gmap", tensor_to_numpy(sequence.frames[0].gmap, np.float32)),
            ("golden_patches", tensor_to_numpy(sequence.frames[0].patches, np.float32)),
            ("golden_colors", tensor_to_numpy(sequence.frames[0].colors, np.float32)),
        ]
    )
    write_case(args.output_root / "patchify_small", patchify_case)

    correlation_case = OrderedDict(
        [
            ("poses", tensor_to_numpy(sequence.poses, np.float32)),
            ("intrinsics", tensor_to_numpy(sequence.intrinsics, np.float32)),
            ("patches", tensor_to_numpy(sequence.patches, np.float32)),
            ("gmap", tensor_to_numpy(sequence.gmap, np.float32)),
            ("fmap1", tensor_to_numpy(sequence.fmap1, np.float32)),
            ("fmap2", tensor_to_numpy(sequence.fmap2, np.float32)),
            ("ii", tensor_to_numpy(sequence.ii, np.int64)),
            ("jj", tensor_to_numpy(sequence.jj, np.int64)),
            ("kk", tensor_to_numpy(sequence.kk, np.int64)),
            ("golden_coords", tensor_to_numpy(sequence.coords, np.float32)),
            ("golden_corr", tensor_to_numpy(sequence.corr, np.float32)),
        ]
    )
    write_case(args.output_root / "correlation_small", correlation_case)

    update_case = OrderedDict(
        [
            ("net", tensor_to_numpy(sequence.net, np.float32)),
            ("ctx", tensor_to_numpy(sequence.ctx, np.float32)),
            ("corr", tensor_to_numpy(sequence.corr, np.float32)),
            ("ii", tensor_to_numpy(sequence.ii, np.int64)),
            ("jj", tensor_to_numpy(sequence.jj, np.int64)),
            ("kk", tensor_to_numpy(sequence.kk, np.int64)),
            ("golden_net", tensor_to_numpy(sequence.update_net, np.float32)),
            ("golden_delta", tensor_to_numpy(sequence.update_delta, np.float32)),
            ("golden_weight", tensor_to_numpy(sequence.update_weight, np.float32)),
        ]
    )
    write_case(args.output_root / "update_small", update_case)

    bundle_adjustment_case = OrderedDict(
        [
            ("poses", tensor_to_numpy(sequence.poses, np.float32)),
            ("intrinsics", tensor_to_numpy(sequence.intrinsics, np.float32)),
            ("patches", tensor_to_numpy(sequence.patches, np.float32)),
            ("ii", tensor_to_numpy(sequence.ii, np.int64)),
            ("jj", tensor_to_numpy(sequence.jj, np.int64)),
            ("kk", tensor_to_numpy(sequence.kk, np.int64)),
            (
                "target",
                tensor_to_numpy(
                    sequence.coords[:, :, 1, 1].unsqueeze(0) + sequence.update_delta,
                    np.float32,
                ),
            ),
            ("weight", tensor_to_numpy(sequence.update_weight, np.float32)),
            ("bounds", tensor_to_numpy(sequence.bounds, np.float32)),
            ("fixed_pose_count", tensor_to_numpy(sequence.fixed_pose_count, np.int64)),
            ("golden_poses", tensor_to_numpy(sequence.ba_poses, np.float32)),
            ("golden_patches", tensor_to_numpy(sequence.ba_patches, np.float32)),
        ]
    )
    write_case(args.output_root / "bundle_adjustment_small", bundle_adjustment_case)

    print(f"Generated parity cases under {args.output_root}")
    print(
        "frames=%d patches_per_frame=%d edges=%d size=%dx%d"
        % (
            len(sequence.frames),
            args.patches_per_frame,
            int(sequence.kk.numel()),
            args.width,
            args.height,
        )
    )
    for label, path in (
        ("patchify", args.output_root / "patchify_small"),
        ("correlation", args.output_root / "correlation_small"),
        ("update", args.output_root / "update_small"),
        ("bundle_adjustment", args.output_root / "bundle_adjustment_small"),
    ):
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
