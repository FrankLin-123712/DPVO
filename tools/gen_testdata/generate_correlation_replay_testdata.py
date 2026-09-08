#!/usr/bin/env python3
"""Capture actual DPVO.corr calls for standalone C++/FireSim replay."""
from __future__ import annotations

import argparse
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
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
    write_case, write_manifest,
)

FORMAT = "dpvo_correlation_replay_v1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=REPO_ROOT / "dpvo.pth")
    parser.add_argument("--images", type=Path, default=REPO_ROOT / "datasets/EUROC/MH_01_easy/mav0/cam0/data")
    parser.add_argument("--calib", type=Path, default=REPO_ROOT / "calib/euroc.txt")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "testdata/correlation_replay_euroc_mh01_first16_p16")
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
        print("Overriding YAML MIXED_PRECISION=True: correlation replay captures use FP32.")
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
        raise ValueError("Non-finite values in correlation capture")
    return result


def capture_inputs(tracker: Any, coords: Any, indicies: Any = None) -> OrderedDict[str, np.ndarray]:
    # DPVO.corr calls its patch indices 'ii'; they are actually kk, not source frame ii.
    kk_tensor, jj_tensor = indicies if indicies is not None else (tracker.pg.kk, tracker.pg.jj)
    kk, jj = snapshot(kk_tensor, "int64"), snapshot(jj_tensor, "int64")
    coord_array = snapshot(coords, "float32")
    if kk.ndim != 1 or jj.shape != kk.shape or coord_array.shape != (1, kk.size, 2, 3, 3):
        raise ValueError("Expected kk/jj [E] and coords [1,E,2,3,3]")
    if (kk < 0).any() or (jj < 0).any():
        raise ValueError("Negative correlation indices")
    m, pmem, mem = int(tracker.M), int(tracker.pmem), int(tracker.mem)
    if min(m, pmem, mem) <= 0:
        raise ValueError("Invalid patch count or cache capacity")
    if len(tracker.pyramid) != 2:
        raise ValueError("Expected exactly two correlation pyramid levels")
    gmap_slot_ids = np.unique((kk // m) % pmem)
    fmap_slot_ids = np.unique(jj % mem)
    arrays = OrderedDict(
        kk=kk, jj=jj, coords=coord_array[0].copy(),
        patches_per_frame=np.array([m], dtype="<i8"),
        patch_memory_size=np.array([pmem], dtype="<i8"),
        frame_memory_size=np.array([mem], dtype="<i8"),
        gmap_slot_ids=gmap_slot_ids, fmap_slot_ids=fmap_slot_ids,
    )
    for slot in gmap_slot_ids:
        value = snapshot(tracker.gmap_[int(slot)], "float32")
        if value.shape != (m, 128, 3, 3):
            raise ValueError(f"Unexpected gmap slot shape: {value.shape}")
        arrays[f"gmap_slot_{slot:03d}"] = value
    for slot in fmap_slot_ids:
        for level, cache in enumerate(tracker.pyramid, 1):
            value = snapshot(cache[0, int(slot)], "float32")
            if value.ndim != 3 or value.shape[0] != 128 or min(value.shape[1:]) <= 0:
                raise ValueError(f"Unexpected fmap slot shape: {value.shape}")
            arrays[f"fmap{level}_slot_{slot:03d}"] = value
    return arrays


def is_feature(name: str) -> bool:
    return (name.startswith(("gmap_slot_", "fmap1_slot_", "fmap2_slot_"))
            and name.rsplit("_", 1)[-1].isdigit())


class ReplayWriter:
    """Write one case at a time; deduplicate immutable feature versions on disk."""

    def __init__(self, root: Path):
        self.root = root
        self.feature_root = root / "features"
        self.feature_root.mkdir()
        self.records: list[dict[str, Any]] = []

    def write(self, arrays: OrderedDict[str, np.ndarray], metadata: dict[str, Any]) -> None:
        case_name = f"case_{len(self.records):04d}"
        case_dir = self.root / case_name
        case_dir.mkdir()
        write_case(case_dir, OrderedDict((k, v) for k, v in arrays.items() if not is_feature(k)))
        feature_files = {}
        for name, array in arrays.items():
            if not is_feature(name):
                continue
            digest = hashlib.sha256()
            digest.update(f"{array.dtype.str}:{array.shape}:".encode())
            digest.update(memoryview(array).cast("B"))
            feature_path = self.feature_root / f"{digest.hexdigest()}.bin"
            if not feature_path.exists():
                array.tofile(feature_path)
            try:
                os.link(feature_path, case_dir / f"{name}.bin")
            except OSError:
                shutil.copyfile(feature_path, case_dir / f"{name}.bin")
            feature_files[name] = feature_path.relative_to(self.root).as_posix()
        write_manifest(case_dir, arrays)
        record = dict(metadata, case_dir=case_name, edges=int(arrays["kk"].size),
                      feature_files=feature_files,
                      tensor_shapes={k: list(v.shape) for k, v in arrays.items()})
        (case_dir / "metadata.json").write_text(json.dumps(record, indent=2) + "\n")
        self.records.append(record)
        # Only completely written cases enter the index, including in an interrupted run.
        with (self.root / "cases.txt").open("a", encoding="utf-8") as output:
            output.write(case_name + "\n")
        print(f"Captured {case_name}: stage={record['stage']} frame={record['input_frame_index']} edges={record['edges']}", flush=True)


class CorrelationRecorder:
    def __init__(self, tracker: Any, writer: ReplayWriter):
        self.tracker, self.writer = tracker, writer
        self.input_frame_index = -1
        self.source_image: str | None = None
        self.initialized_at_frame_start = False
        self.terminating = False
        self.stage = "unknown"
        self.iteration: int | None = None
        self.next_iteration = 0

    def start_frame(self, index: int, image: Path) -> None:
        self.input_frame_index, self.source_image = index, str(image)
        self.initialized_at_frame_start = bool(self.tracker.is_initialized)
        self.next_iteration = 0

    @contextmanager
    def installed(self):
        tracker = self.tracker
        names = ("corr", "update", "motion_probe")
        originals = {name: getattr(tracker, name) for name in names}
        # Restore the exact instance namespace, allowing repeated use and exceptions.
        saved = {name: tracker.__dict__[name] for name in names if name in tracker.__dict__}

        def recording_corr(coords, indicies=None):
            arrays = capture_inputs(tracker, coords, indicies)
            metadata = {
                "format": FORMAT, "call_index": len(self.writer.records),
                "input_frame_index": self.input_frame_index, "source_image": self.source_image,
                "tracker_n": int(tracker.n), "tracker_counter": int(tracker.counter),
                "stage": self.stage, "iteration": self.iteration,
                "is_initialized": bool(tracker.is_initialized), "dtype": "float32",
                "radius": 3, "pyramid_coordinate_divisors": [1, 4],
            }
            result = originals["corr"](coords, indicies=indicies)
            arrays["golden_corr"] = snapshot(result, "float32")
            if arrays["golden_corr"].shape != (1, arrays["kk"].size, 882):
                raise ValueError("Expected golden_corr [1,E,882]")
            self.writer.write(arrays, metadata)
            return result

        def in_stage(name, fn, *args, **kwargs):
            previous = self.stage, self.iteration
            self.stage = name
            self.iteration = self.next_iteration if name != "motion_probe" else None
            if name != "motion_probe":
                self.next_iteration += 1
            try:
                return fn(*args, **kwargs)
            finally:
                self.stage, self.iteration = previous

        def recording_update(*args, **kwargs):
            stage = "terminate" if self.terminating else (
                "update" if self.initialized_at_frame_start else "initialization")
            return in_stage(stage, originals["update"], *args, **kwargs)

        def recording_probe(*args, **kwargs):
            return in_stage("motion_probe", originals["motion_probe"], *args, **kwargs)

        tracker.corr, tracker.update, tracker.motion_probe = recording_corr, recording_update, recording_probe
        try:
            yield self
        finally:
            for name in names:
                if name in saved:
                    setattr(tracker, name, saved[name])
                else:
                    delattr(tracker, name)


def main() -> int:
    args = parse_args()
    import torch

    # Check before importing DPVO's CUDA extensions or creating output files.
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to generate correlation replay data; use a GPU machine.")
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
    metadata = {"format": FORMAT, "status": "incomplete", "tracker_config": asdict(tracker_cfg)}
    metadata_path = root / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    writer = ReplayWriter(root)
    try:
        image_dir = root / "images"
        image_dir.mkdir()
        frames, frame_names, intrinsics, src_w, src_h, dst_w, dst_h = preprocess_frames(
            image_paths, source_intrinsics, distortion, image_dir,
            args.width, args.height, args.max_long_edge,
        )
        write_adjusted_calibration(root / "calib.txt", intrinsics)
        metadata.update(
            weights=str(args.weights.resolve()), source_calib=str(args.calib.resolve()),
            config_yaml=str(args.config_yaml.resolve()) if args.config_yaml else None,
            seed=args.seed, cuda_device=args.cuda_device, torch_version=torch.__version__,
            frame_start=args.frame_start, frame_count=args.frame_count, frame_step=args.frame_step,
            source_frame_paths=[str(p.resolve()) for p in image_paths], generated_frame_files=frame_names,
            resize=dict(source_width=src_w, source_height=src_h, output_width=dst_w, output_height=dst_h),
            undistortion_applied=bool(distortion is not None and distortion.size),
            input_intrinsics=intrinsics.tolist(), include_terminate_updates=not args.skip_terminate_updates,
        )
        slam = DPVO(tracker_cfg, str(args.weights), ht=dst_h, wd=dst_w, viz=False)
        recorder = CorrelationRecorder(slam, writer)
        with torch.no_grad(), recorder.installed():
            for index, frame in enumerate(frames):
                recorder.start_frame(index, image_paths[index].resolve())
                image_tensor = torch.from_numpy(frame).permute(2, 0, 1).cuda(non_blocking=False)
                intrinsics_tensor = torch.from_numpy(intrinsics.copy()).cuda(non_blocking=False)
                slam(index, image_tensor, intrinsics_tensor)
            if not args.skip_terminate_updates:
                recorder.terminating = True
                recorder.input_frame_index = -1
                recorder.source_image = None
                recorder.next_iteration = 0
                slam.terminate()
        if not writer.records:
            raise RuntimeError("No correlation calls captured; try more input frames.")
        metadata["status"] = "complete"
    except BaseException as exc:
        metadata.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        metadata.update(captured_case_count=len(writer.records), cases=writer.records)
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote {len(writer.records)} correlation replay cases under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
