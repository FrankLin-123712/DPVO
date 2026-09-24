#!/usr/bin/env python3
"""Capture actual DPVO update calls for standalone C++/FireSim replay."""
from __future__ import annotations

import argparse
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import asdict
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

FORMAT = "dpvo_update_replay_v1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=REPO_ROOT / "dpvo.pth")
    parser.add_argument("--images", type=Path, default=REPO_ROOT / "datasets/EUROC/MH_01_easy/mav0/cam0/data")
    parser.add_argument("--calib", type=Path, default=REPO_ROOT / "calib/euroc.txt")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "testdata/fp32/update_replay_euroc_mh01_first16_p16")
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
    parser.add_argument("--include-motion-probes", action="store_true", help="Also capture network-only motion_probe cases.")
    args = parser.parse_args(argv)
    args.mixed_precision = False
    args.nn_fp16_weights = False
    apply_config_yaml(args)
    if args.mixed_precision:
        print("Overriding YAML MIXED_PRECISION=True: update replay captures use FP32.")
    args.mixed_precision = False
    if args.nn_fp16_weights:
        print("Overriding YAML NN_FP16_WEIGHTS=True: update replay captures use FP32 weights.")
    args.nn_fp16_weights = False
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
        raise ValueError("Non-finite values in update capture")
    return result


# The correlation writer hard-links immutable gmap/fmap versions across cases.
from generate_correlation_replay_testdata import ReplayWriter, capture_inputs as correlation_inputs


class UpdateRecorder:
    """Observe the original tracker calls; never re-run a network to make goldens."""

    def __init__(self, tracker: Any, writer: ReplayWriter, include_motion_probes=False):
        self.tracker, self.writer = tracker, writer
        self.include_motion_probes = include_motion_probes
        self.input_frame_index = -1
        self.source_image = None
        self.initialized_at_frame_start = False
        self.terminating = False
        self.next_iteration = 0
        self.current = None

    def start_frame(self, index: int, image: Path) -> None:
        self.input_frame_index, self.source_image = index, str(image)
        self.initialized_at_frame_start = bool(self.tracker.is_initialized)
        self.next_iteration = 0

    @contextmanager
    def installed(self):
        import dpvo.dpvo as module
        t = self.tracker
        objects = [(t, name) for name in ("update", "motion_probe", "corr")]
        objects += [(t.network.update, "forward"), (module.fastba, "BA")]
        saved = [(obj, name, name in obj.__dict__, obj.__dict__.get(name)) for obj, name in objects]
        original_update, original_probe, original_corr = t.update, t.motion_probe, t.corr
        original_forward, original_ba = t.network.update.forward, module.fastba.BA

        def forward(net, inp, corr, flow, ii, jj, kk):
            c = self.current
            if c is not None:
                if c[1].get("network_calls", 0):
                    raise ValueError("Expected one network call per update")
                if flow is not None:
                    raise ValueError("Only inference flow=None is supported")
                arrays, meta = c
                for name, tensor in (("net", net), ("ctx", inp), ("corr", corr)):
                    arrays[name] = snapshot(tensor, "float32")
                for name, tensor in (("ii", ii), ("jj", jj), ("kk", kk)):
                    arrays[name] = snapshot(tensor, "int64")
                meta["network_calls"] = 1
            result = original_forward(net, inp, corr, flow, ii, jj, kk)
            if c is not None:
                net_out, (delta, weight, _) = result
                for name, tensor in (("golden_net", net_out), ("golden_delta", delta), ("golden_weight", weight)):
                    c[0][name] = snapshot(tensor, "float32")
            return result

        def corr(coords, indicies=None):
            if self.current is not None and self.current[1]["stage"] != "motion_probe":
                arrays = self.current[0]
                arrays.update(correlation_inputs(t, coords, indicies))
                for slot in arrays["gmap_slot_ids"]:
                    arrays[f"imap_slot_{slot:03d}"] = snapshot(t.imap_[int(slot)], "float32")
            return original_corr(coords, indicies=indicies)

        def ba(*args, **kwargs):
            c = self.current
            if c is not None:
                c[1]["ba_calls"] += 1
            try:
                result = original_ba(*args, **kwargs)
            except BaseException:
                if c is not None:
                    c[1]["ba_failed"] = True
                raise
            return result

        def capture(stage, fn, *args, **kwargs):
            if self.current is not None:
                raise RuntimeError("Nested tracker update capture")
            arrays = OrderedDict()
            is_iteration = stage != "motion_probe"
            meta = dict(format=FORMAT, call_index=len(self.writer.records),
                        input_frame_index=self.input_frame_index, source_image=self.source_image,
                        stage=stage, iteration=self.next_iteration if is_iteration else -1,
                        tracker_n=int(t.n), tracker_m=int(t.m), tracker_counter=int(t.counter),
                        is_initialized=bool(t.is_initialized), height=int(t.ht), width=int(t.wd),
                        patches_per_frame=int(t.M), patch_memory_size=int(t.pmem), frame_memory_size=int(t.mem),
                        dtype="float32", byte_order="little", patch_size=3, pose_layout="tx_ty_tz_qx_qy_qz_qw",
                        ba_calls=0, ba_failed=False, iteration_supported=is_iteration,
                        optimization_window=int(t.cfg.OPTIMIZATION_WINDOW), ba_iterations=int(t.cfg.BA_ITERATIONS),
                        buffer_size=int(t.N), removal_window=int(t.cfg.REMOVAL_WINDOW),
                        patch_lifetime=int(t.cfg.PATCH_LIFETIME))
            if is_iteration:
                self.next_iteration += 1
                global_ba = bool((t.pg.ii < t.n - t.cfg.REMOVAL_WINDOW - 1).any()) and not t.ran_global_ba[t.n]
                meta["global_ba"] = bool(global_ba)
                meta["iteration_supported"] = not global_ba
                meta["fixed_pose_count"] = max(t.n - t.cfg.OPTIMIZATION_WINDOW if t.is_initialized else 1, 1)
                arrays["poses"] = snapshot(t.pg.poses_[:t.n], "float32")
                arrays["patches"] = snapshot(t.pg.patches_[:t.n], "float32")
                arrays["intrinsics"] = snapshot(t.pg.intrinsics_[:t.n], "float32")
                arrays["patch_to_frame"] = snapshot(t.ix[:t.m], "int64")
            self.current = (arrays, meta)
            try:
                result = fn(*args, **kwargs)
                if meta.get("network_calls") != 1:
                    raise ValueError("No network call captured")
                e = arrays["kk"].size
                expected = dict(net=(1,e,384), ctx=(1,e,384), corr=(1,e,882),
                                ii=(e,), jj=(e,), kk=(e,), golden_net=(1,e,384),
                                golden_delta=(1,e,2), golden_weight=(1,e,2))
                if e == 0:
                    raise ValueError("Empty update is not a replay case")
                if is_iteration:
                    arrays["golden_target"] = snapshot(t.pg.target, "float32")
                    arrays["golden_graph_weight"] = snapshot(t.pg.weight, "float32")
                    arrays["golden_poses"] = snapshot(t.pg.poses_[:t.n], "float32")
                    arrays["golden_patches"] = snapshot(t.pg.patches_[:t.n], "float32")
                    arrays["golden_points"] = snapshot(t.pg.points_[:t.m], "float32")
                    expected.update(coords=(e,2,3,3), poses=(t.n,7), patches=(t.n,t.M,3,3,3),
                                    intrinsics=(t.n,4), patch_to_frame=(t.m,), golden_poses=(t.n,7),
                                    golden_patches=(t.n,t.M,3,3,3), golden_points=(t.m,3),
                                    golden_target=(1,e,2), golden_graph_weight=(1,e,2))
                    if t.m != t.n * t.M:
                        raise ValueError("Expected active patch count n*M")
                    meta["iteration_supported"] &= meta["ba_calls"] == 1 and not meta["ba_failed"]
                for name, shape in expected.items():
                    if arrays[name].shape != shape:
                        raise ValueError(f"{name}: expected {shape}, got {arrays[name].shape}")
                meta["unique_kk"] = int(np.unique(arrays["kk"]).size)
                meta["unique_ij"] = int(np.unique(arrays["ii"] * 12345 + arrays["jj"]).size)
                self.writer.write(arrays, meta)
                return result
            finally:
                self.current = None

        def update(*args, **kwargs):
            stage = "terminate" if self.terminating else ("update" if self.initialized_at_frame_start else "initialization")
            return capture(stage, original_update, *args, **kwargs)

        def probe(*args, **kwargs):
            if self.include_motion_probes:
                return capture("motion_probe", original_probe, *args, **kwargs)
            return original_probe(*args, **kwargs)

        try:
            t.update, t.motion_probe, t.corr = update, probe, corr
            t.network.update.forward, module.fastba.BA = forward, ba
            yield self
        finally:
            for obj, name, existed, value in reversed(saved):
                if existed:
                    setattr(obj, name, value)
                else:
                    obj.__dict__.pop(name, None)


def main() -> int:
    args = parse_args()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to generate update replay data; use a GPU machine.")
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
            source_calib=str(args.calib.resolve()), seed=args.seed,
            config_yaml=str(args.config_yaml.resolve()) if args.config_yaml else None,
            cuda_device=args.cuda_device, torch_version=torch.__version__,
            frame_start=args.frame_start, frame_count=args.frame_count, frame_step=args.frame_step,
            source_frame_paths=[str(p.resolve()) for p in image_paths], generated_frame_files=names,
            resize=dict(source_width=sw, source_height=sh, output_width=w, output_height=h),
            undistortion_applied=bool(distortion is not None and distortion.size),
            input_intrinsics=intrinsics.tolist(), include_terminate_updates=not args.skip_terminate_updates,
            dtype="float32", byte_order="little", tf32=False, include_motion_probes=args.include_motion_probes)
        slam = DPVO(tracker_cfg, str(args.weights), ht=h, wd=w, viz=False)
        recorder = UpdateRecorder(slam, writer, args.include_motion_probes)
        with torch.no_grad(), recorder.installed():
            for index, frame in enumerate(frames):
                recorder.start_frame(index, image_paths[index].resolve())
                image_tensor = torch.from_numpy(frame).permute(2, 0, 1).cuda()
                intrinsics_tensor = torch.from_numpy(intrinsics.copy()).cuda()
                slam(index, image_tensor, intrinsics_tensor)
            if not args.skip_terminate_updates:
                recorder.terminating = True
                recorder.next_iteration = 0
                slam.terminate()
        if not writer.records:
            raise RuntimeError("No update cases captured; try more frames")
        metadata["status"] = "complete"
    except BaseException as exc:
        metadata.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        metadata.update(captured_case_count=len(writer.records), cases=writer.records)
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote {len(writer.records)} update replay cases under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
