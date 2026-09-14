#!/usr/bin/env python3
"""Capture actual DPVO patchify calls for standalone C++/FireSim replay."""
from __future__ import annotations

import argparse
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from generate_ba_benchmark_testdata import apply_config_yaml  # noqa: E402
from generate_dpvo_python_testdata import (  # noqa: E402
    build_tracker_config, collect_image_paths, configure_determinism,
    load_calibration, preprocess_frames, write_adjusted_calibration,
    write_case,
)

FORMAT = "dpvo_patchify_replay_v1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=REPO_ROOT / "dpvo.pth")
    parser.add_argument("--images", type=Path, default=REPO_ROOT / "datasets/EUROC/MH_01_easy/mav0/cam0/data")
    parser.add_argument("--calib", type=Path, default=REPO_ROOT / "calib/euroc.txt")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "testdata/patchify_replay_euroc_mh01_first16_p16")
    parser.add_argument("--config-yaml", type=Path, help="Tracker settings; FP32 is always enforced after loading YAML.")
    parser.add_argument("--frame-start", type=int, default=1, help="1-based source image index.")
    parser.add_argument("--frame-count", type=int, default=16, help="At least 8, as required by the shared sequence loader.")
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--max-long-edge", type=int, default=752)
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--height", type=int, default=0)
    parser.add_argument("--undistort", dest="no_undistort", action="store_false", help="Opt in to distortion correction; disabled by default.")
    parser.add_argument("--no-undistort", dest="no_undistort", action="store_true")
    parser.set_defaults(no_undistort=True)
    parser.add_argument("--patches-per-frame", type=int, default=16)
    parser.add_argument("--buffer-size", type=int, default=64)
    parser.add_argument("--removal-window", type=int, default=16)
    parser.add_argument("--optimization-window", type=int, default=7)
    parser.add_argument("--patch-lifetime", type=int, default=11)
    parser.add_argument("--keyframe-index", type=int, default=4)
    parser.add_argument("--keyframe-thresh", type=float, default=15.0)
    parser.add_argument("--motion-model", default="DAMPED_LINEAR")
    parser.add_argument("--motion-damping", type=float, default=0.5)
    parser.add_argument("--centroid-sel-strat", choices=["RANDOM", "GRADIENT_BIAS"], default="RANDOM")
    parser.add_argument("--ba-iterations", type=int, default=2)
    parser.add_argument("--no-mixed-precision", action="store_true", help="Accepted for explicit commands; all captures already use FP32.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--skip-terminate-updates", action="store_true", help="Omit final updates; included by default.")
    args = parser.parse_args(argv)
    args.mixed_precision = False
    apply_config_yaml(args)
    if args.mixed_precision:
        print("Overriding YAML MIXED_PRECISION=True: patchify replay captures use FP32.")
    args.mixed_precision = False
    for name in ("frame_start", "frame_count", "frame_step", "patches_per_frame", "buffer_size",
                 "removal_window", "optimization_window", "patch_lifetime", "ba_iterations"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.frame_count < 8:
        parser.error("--frame-count must be at least 8")
    return args


def snapshot(tensor: Any, dtype: str) -> np.ndarray:
    """Make owned host storage before the tracker can overwrite a cache slot."""
    array = tensor.detach().cpu().contiguous().numpy()
    if array.dtype != np.dtype(dtype):
        raise ValueError(f"Expected {dtype} capture input, got {array.dtype}; do not cast FP16 captures to FP32")
    result = np.array(array, dtype=np.dtype(dtype).newbyteorder("<"), order="C", copy=True)
    if np.issubdtype(result.dtype, np.floating) and not np.isfinite(result).all():
        raise ValueError("Non-finite values in patchify capture")
    return result


class ReplayWriter:
    def __init__(self, root: Path):
        self.root = root
        self.records: list[dict[str, Any]] = []

    def write(self, arrays: OrderedDict[str, np.ndarray], metadata: dict[str, Any]) -> None:
        name = f"case_{len(self.records):04d}"
        case_dir = self.root / name
        case_dir.mkdir()
        write_case(case_dir, arrays)
        record = dict(metadata, case_dir=name,
                      tensor_shapes={k: list(v.shape) for k, v in arrays.items()})
        (case_dir / "metadata.json").write_text(json.dumps(record, indent=2) + "\n")
        with (self.root / "cases.txt").open("a", encoding="utf-8") as output:
            output.write(name + "\n")
        self.records.append(record)
        print(f"Captured {name}: frame={record['input_frame_index']} patches={record['patches_per_frame']}", flush=True)


class PatchifyRecorder:
    """Observe the real call once, copying outputs before tracker state updates."""
    def __init__(self, tracker: Any, writer: ReplayWriter):
        self.tracker, self.writer = tracker, writer
        self.input_frame_index = -1
        self.source_image: str | None = None
        self.raw: dict[str, np.ndarray] = {}
        self.recording = False

    def start_frame(self, index: int, image: Path) -> None:
        self.input_frame_index, self.source_image = index, str(image)

    @contextmanager
    def installed(self):
        patchifier = self.tracker.network.patchify
        original = patchifier.forward
        had_forward = "forward" in patchifier.__dict__
        saved = patchifier.__dict__.get("forward")
        handles = []

        def hook(name):
            def record(module, inputs, output):
                if self.recording:
                    if name in self.raw:
                        raise ValueError(f"Encoder {name} called more than once")
                    self.raw[name] = snapshot(output, "float32")
            return record

        def forward(images, patches_per_image=80, disps=None,
                    centroid_sel_strat="RANDOM", return_color=False):
            if disps is not None or not return_color or patchifier.patch_size != 3:
                raise ValueError("Replay v1 requires disps=None, return_color=True, patch size 3")
            image = snapshot(images, "float32")
            if image.ndim != 5 or image.shape[:3] != (1, 1, 3):
                raise ValueError("Replay v1 requires images [1,1,3,H,W]")
            h, w = image.shape[-2:]
            if h % 4 or w % 4:
                raise ValueError("Image height/width must be divisible by 4")
            self.raw.clear()
            self.recording = True
            try:
                outputs = original(images, patches_per_image=patches_per_image, disps=disps,
                                   centroid_sel_strat=centroid_sel_strat, return_color=return_color)
            finally:
                self.recording = False
            fmap, gmap, imap, patches, index, colors = outputs
            # Own every buffer now: DPVO subsequently overwrites patches[...,2,:,:].
            patch_array = snapshot(patches, "float32")[0].copy()
            centers = np.stack((patch_array[:, 0, 1, 1], patch_array[:, 1, 1, 1]), axis=-1)
            arrays = OrderedDict(
                image=image, centers=centers,
                raw_fmap=self.raw["raw_fmap"], raw_imap=self.raw["raw_imap"],
                golden_fmap=snapshot(fmap, "float32"),
                golden_imap=snapshot(imap, "float32")[0].copy(),
                golden_gmap=snapshot(gmap, "float32")[0].copy(),
                golden_patches=patch_array,
                golden_colors=snapshot(colors, "float32")[0].copy(),
                golden_index=snapshot(index, "int64"),
            )
            m = int(patches_per_image)
            expected = dict(image=(1,1,3,h,w), centers=(m,2),
                            raw_fmap=(1,1,128,h//4,w//4), raw_imap=(1,1,384,h//4,w//4),
                            golden_fmap=(1,1,128,h//4,w//4), golden_imap=(m,384,1,1),
                            golden_gmap=(m,128,3,3), golden_patches=(m,3,3,3),
                            golden_colors=(m,3), golden_index=(m,))
            for name, shape in expected.items():
                if arrays[name].shape != shape:
                    raise ValueError(f"{name}: expected {shape}, got {arrays[name].shape}")
            if not np.array_equal(arrays["golden_fmap"], arrays["raw_fmap"] * np.float32(0.25)):
                raise ValueError("Unexpected fmap scaling")
            if np.any(arrays["golden_index"] != 0) or not np.all(patch_array[:, 2] == 1):
                raise ValueError("Expected single-frame indices and unmodified unit disparity")
            self.writer.write(arrays, dict(
                format=FORMAT, call_index=len(self.writer.records),
                input_frame_index=self.input_frame_index, source_image=self.source_image,
                tracker_n=int(self.tracker.n), tracker_counter=int(self.tracker.counter),
                dtype="float32", byte_order="little", layout="BNCHW",
                height=h, width=w, patches_per_frame=m, patch_size=3,
                centroid_sel_strat=centroid_sel_strat, disps="unit", return_color=True,
                feature_scale=0.25, normalization="2 * (BGR / 255.0) - 0.5",
            ))
            self.raw.clear()
            return outputs

        try:
            handles.append(patchifier.fnet.register_forward_hook(hook("raw_fmap")))
            handles.append(patchifier.inet.register_forward_hook(hook("raw_imap")))
            patchifier.forward = forward
            yield self
        finally:
            if had_forward:
                patchifier.forward = saved
            else:
                patchifier.__dict__.pop("forward", None)
            for handle in handles:
                handle.remove()
            self.raw.clear()
            self.recording = False


def main() -> int:
    args = parse_args()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to generate patchify replay data; use a GPU machine.")
    torch.cuda.set_device(args.cuda_device)
    configure_determinism(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    from dpvo.dpvo import DPVO

    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)
    image_paths = collect_image_paths(args.images, args.frame_start, args.frame_count, args.frame_step)
    source_intrinsics, distortion = load_calibration(args.calib)
    if args.no_undistort:
        distortion = None
    tracker_cfg = build_tracker_config(args)
    tracker_cfg.BA_ITERATIONS = args.ba_iterations
    root = args.output_root.resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {root}; choose a new --output-root")
    root.mkdir(parents=True, exist_ok=True)
    metadata = dict(format=FORMAT, status="incomplete", tracker_config=asdict(tracker_cfg))
    metadata_path = root / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    writer = ReplayWriter(root)
    try:
        image_dir = root / "images"
        image_dir.mkdir()
        frames, names, intrinsics, sw, sh, w, h = preprocess_frames(
            image_paths, source_intrinsics, distortion, image_dir,
            args.width, args.height, args.max_long_edge)
        write_adjusted_calibration(root / "calib.txt", intrinsics)
        metadata.update(
            weights=str(args.weights.resolve()),
            weights_sha256=hashlib.sha256(args.weights.read_bytes()).hexdigest(),
            source_calib=str(args.calib.resolve()), seed=args.seed,
            config_yaml=str(args.config_yaml.resolve()) if args.config_yaml else None,
            cuda_device=args.cuda_device, torch_version=torch.__version__,
            frame_start=args.frame_start, frame_count=args.frame_count, frame_step=args.frame_step,
            source_frame_paths=[str(p.resolve()) for p in image_paths], generated_frame_files=names,
            resize=dict(source_width=sw, source_height=sh, output_width=w, output_height=h),
            undistortion_applied=bool(distortion is not None and distortion.size),
            input_intrinsics=intrinsics.tolist(), include_terminate_updates=not args.skip_terminate_updates,
            dtype="float32", byte_order="little", tf32=False)
        slam = DPVO(tracker_cfg, str(args.weights), ht=h, wd=w, viz=False)
        recorder = PatchifyRecorder(slam, writer)
        with torch.no_grad(), recorder.installed():
            for index, frame in enumerate(frames):
                recorder.start_frame(index, image_paths[index].resolve())
                image_tensor = torch.from_numpy(frame).permute(2, 0, 1).cuda()
                intrinsics_tensor = torch.from_numpy(intrinsics.copy()).cuda()
                slam(index, image_tensor, intrinsics_tensor)
            if not args.skip_terminate_updates:
                slam.terminate()
        if len(writer.records) != len(frames):
            raise RuntimeError("Expected one patchify call per input frame")
        metadata["status"] = "complete"
    except BaseException as exc:
        metadata.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        metadata.update(captured_case_count=len(writer.records), cases=writer.records)
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote {len(writer.records)} patchify replay cases under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
