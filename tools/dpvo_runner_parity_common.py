#!/usr/bin/env python3
from __future__ import annotations

import math
import os
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(TOOLS_DIR))

from dpvo import projective_ops as pops  # noqa: E402
from dpvo.ba import BA  # noqa: E402
from dpvo.lietorch import SE3  # noqa: E402
from export_models import (  # noqa: E402
    DIM,
    UpdateWrapperExplicitNeighbors,
    compute_neighbor_indices,
    load_weights,
)

PATCH_SIZE = 3
CORR_RADIUS = 3


@dataclass
class FrameData:
    image: torch.Tensor
    centers: torch.Tensor
    fmap: torch.Tensor
    patch_imap: torch.Tensor
    gmap: torch.Tensor
    patches: torch.Tensor
    colors: torch.Tensor


@dataclass
class SequenceData:
    frames: list[FrameData]
    poses: torch.Tensor
    intrinsics: torch.Tensor
    patches: torch.Tensor
    gmap: torch.Tensor
    fmap1: torch.Tensor
    fmap2: torch.Tensor
    ii: torch.Tensor
    jj: torch.Tensor
    kk: torch.Tensor
    coords: torch.Tensor
    corr: torch.Tensor
    ctx: torch.Tensor
    net: torch.Tensor
    update_net: torch.Tensor
    update_delta: torch.Tensor
    update_weight: torch.Tensor
    ba_poses: torch.Tensor
    ba_patches: torch.Tensor
    bounds: torch.Tensor
    fixed_pose_count: torch.Tensor


def load_normalized_image(image_path: Path, height: int, width: int) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    image = image.resize((width, height), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return 2.0 * tensor - 0.5


def scaled_intrinsics(
    calib_path: Path,
    src_width: int,
    src_height: int,
    dst_width: int,
    dst_height: int,
) -> torch.Tensor:
    values = calib_path.read_text(encoding="utf-8").strip().split()
    if len(values) < 4:
        raise ValueError(f"Expected at least 4 calibration values in {calib_path}")

    fx, fy, cx, cy = map(float, values[:4])
    sx = dst_width / float(src_width)
    sy = dst_height / float(src_height)
    intrinsics = torch.tensor(
        [fx * sx / 4.0, fy * sy / 4.0, cx * sx / 4.0, cy * sy / 4.0],
        dtype=torch.float32,
    )
    return intrinsics


def build_pose_sequence(frame_count: int) -> torch.Tensor:
    poses = torch.zeros(frame_count, 7, dtype=torch.float32)
    for frame in range(frame_count):
        poses[frame, 0] = 0.0125 * frame
        poses[frame, 1] = 0.0015 * frame
        poses[frame, 2] = 0.0025 * frame
        poses[frame, 6] = 1.0
    return poses


def _safe_fetch_chw(tensor: torch.Tensor, y: int, x: int) -> torch.Tensor:
    if y < 0 or x < 0 or y >= tensor.shape[1] or x >= tensor.shape[2]:
        return torch.zeros(tensor.shape[0], dtype=tensor.dtype)
    return tensor[:, y, x]


def bilinear_sample_chw(tensor: torch.Tensor, x: float, y: float) -> torch.Tensor:
    x0 = math.floor(x)
    y0 = math.floor(y)
    dx = float(x - x0)
    dy = float(y - y0)
    v00 = _safe_fetch_chw(tensor, y0, x0)
    v01 = _safe_fetch_chw(tensor, y0, x0 + 1)
    v10 = _safe_fetch_chw(tensor, y0 + 1, x0)
    v11 = _safe_fetch_chw(tensor, y0 + 1, x0 + 1)
    return (
        (1.0 - dy) * (1.0 - dx) * v00
        + (1.0 - dy) * dx * v01
        + dy * (1.0 - dx) * v10
        + dy * dx * v11
    )


def patchify_single_chw(tensor: torch.Tensor, centers: torch.Tensor, radius: int) -> torch.Tensor:
    channels = tensor.shape[0]
    diameter = 2 * radius + 1
    out = torch.empty(
        centers.shape[0],
        channels,
        diameter,
        diameter,
        dtype=tensor.dtype,
    )
    for patch_index, center in enumerate(centers.tolist()):
        cx = float(center[0])
        cy = float(center[1])
        for iy in range(diameter):
            for ix in range(diameter):
                sample_x = cx + float(ix - radius)
                sample_y = cy + float(iy - radius)
                out[patch_index, :, iy, ix] = bilinear_sample_chw(tensor, sample_x, sample_y)
    return out


def build_grid_tensor(height: int, width: int) -> torch.Tensor:
    x = torch.arange(width, dtype=torch.float32).view(1, 1, width).expand(1, height, width)
    y = torch.arange(height, dtype=torch.float32).view(1, height, 1).expand(1, height, width)
    ones = torch.ones(1, height, width, dtype=torch.float32)
    return torch.cat([x, y, ones], dim=0)


def select_patch_centers(image: torch.Tensor, patches_per_frame: int, suppression_radius: int = 4) -> torch.Tensor:
    _, height, width = image.shape
    feature_height = height // 4
    feature_width = width // 4
    gray = ((image + 0.5) * 127.5).sum(dim=0)
    gradient = torch.zeros(feature_height, feature_width, dtype=torch.float32)

    for y in range(height - 1):
        for x in range(width - 1):
            dx = float(gray[y, x + 1] - gray[y, x])
            dy = float(gray[y + 1, x] - gray[y, x])
            score = math.sqrt(dx * dx + dy * dy)
            fy = min(feature_height - 1, y // 4)
            fx = min(feature_width - 1, x // 4)
            gradient[fy, fx] = max(gradient[fy, fx], score)

    gradient[0, :] = -float("inf")
    gradient[-1, :] = -float("inf")
    gradient[:, 0] = -float("inf")
    gradient[:, -1] = -float("inf")

    centers: list[list[float]] = []
    scores = gradient.clone()
    for _ in range(patches_per_frame):
        flat_index = int(torch.argmax(scores).item())
        if not torch.isfinite(scores.view(-1)[flat_index]):
            break
        cy = flat_index // feature_width
        cx = flat_index % feature_width
        centers.append([float(cx), float(cy)])
        y0 = max(1, cy - suppression_radius)
        y1 = min(feature_height - 1, cy + suppression_radius + 1)
        x0 = max(1, cx - suppression_radius)
        x1 = min(feature_width - 1, cx + suppression_radius + 1)
        scores[y0:y1, x0:x1] = -float("inf")

    if len(centers) < patches_per_frame:
        for y in range(1, feature_height - 1):
            for x in range(1, feature_width - 1):
                centers.append([float(x), float(y)])
                if len(centers) == patches_per_frame:
                    break
            if len(centers) == patches_per_frame:
                break

    if len(centers) != patches_per_frame:
        raise RuntimeError("Unable to select the requested number of patch centers")

    return torch.tensor(centers, dtype=torch.float32)


def build_depths(centers: torch.Tensor, frame_index: int, frame_count: int, feature_width: int, feature_height: int) -> torch.Tensor:
    denom_w = max(1.0, float(feature_width - 1))
    denom_h = max(1.0, float(feature_height - 1))
    frame_term = 0.12 * frame_index / max(1, frame_count - 1)
    depths = []
    for center in centers.tolist():
        nx = float(center[0]) / denom_w
        ny = float(center[1]) / denom_h
        depths.append(0.85 + 0.22 * nx + 0.08 * ny + frame_term)
    return torch.tensor(depths, dtype=torch.float32)


def build_edge_graph(frame_count: int, patches_per_frame: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ii: list[int] = []
    jj: list[int] = []
    kk: list[int] = []
    for source_frame in range(frame_count - 1):
        for local_patch in range(patches_per_frame):
            patch_index = source_frame * patches_per_frame + local_patch
            for target_frame in range(source_frame + 1, frame_count):
                ii.append(source_frame)
                jj.append(target_frame)
                kk.append(patch_index)
    return (
        torch.tensor(ii, dtype=torch.long),
        torch.tensor(jj, dtype=torch.long),
        torch.tensor(kk, dtype=torch.long),
    )


def build_correlation_volume(
    gmap: torch.Tensor,
    fmap1: torch.Tensor,
    fmap2: torch.Tensor,
    jj: torch.Tensor,
    coords: torch.Tensor,
) -> torch.Tensor:
    edges = gmap.shape[0]
    diameter = 2 * CORR_RADIUS + 1
    out = torch.empty(1, edges, diameter * diameter * PATCH_SIZE * PATCH_SIZE * 2, dtype=torch.float32)

    for edge in range(edges):
        cursor = 0
        fmap1_frame = fmap1[int(jj[edge].item())]
        fmap2_frame = fmap2[int(jj[edge].item())]
        for xoff in range(-CORR_RADIUS, CORR_RADIUS + 1):
            for yoff in range(-CORR_RADIUS, CORR_RADIUS + 1):
                for py in range(PATCH_SIZE):
                    for px in range(PATCH_SIZE):
                        src = gmap[edge, :, py, px]
                        x1 = float(coords[edge, 0, py, px]) + float(xoff)
                        y1 = float(coords[edge, 1, py, px]) + float(yoff)
                        x2 = float(coords[edge, 0, py, px]) / 4.0 + float(xoff)
                        y2 = float(coords[edge, 1, py, px]) / 4.0 + float(yoff)
                        sample1 = bilinear_sample_chw(fmap1_frame, x1, y1)
                        sample2 = bilinear_sample_chw(fmap2_frame, x2, y2)
                        out[0, edge, cursor] = torch.dot(src, sample1)
                        out[0, edge, cursor + 1] = torch.dot(src, sample2)
                        cursor += 2

    return out


def flatten_patch_tensor(frames: Iterable[torch.Tensor]) -> torch.Tensor:
    return torch.cat(list(frames), dim=0).contiguous()


def write_manifest(output_dir: Path, tensors: OrderedDict[str, np.ndarray]) -> None:
    lines = [
        "# dpvo_runner_parity_case_v1",
        "# filename is inferred as <tensor_name>.bin",
    ]
    for name, array in tensors.items():
        shape = " ".join(str(dim) for dim in array.shape)
        lines.append(f"{name} {array.dtype.name} {shape}".rstrip())
    (output_dir / "manifest.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_case(output_dir: Path, tensors: OrderedDict[str, np.ndarray]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(output_dir, tensors)
    for name, array in tensors.items():
        np.ascontiguousarray(array).tofile(output_dir / f"{name}.bin")


def tensor_to_numpy(tensor: torch.Tensor, dtype: np.dtype | None = None) -> np.ndarray:
    array = tensor.detach().cpu().contiguous().numpy()
    if dtype is not None:
        return array.astype(dtype, copy=False)
    return array


def build_sequence_data(
    weights: Path,
    image_paths: list[Path],
    calib_path: Path,
    width: int,
    height: int,
    patches_per_frame: int,
) -> SequenceData:
    torch.set_num_threads(1)
    model = load_weights(weights).cpu().eval()
    update = UpdateWrapperExplicitNeighbors(model).cpu().eval()

    src_image = Image.open(image_paths[0])
    src_width, src_height = src_image.size
    src_image.close()
    scaled_intr = scaled_intrinsics(calib_path, src_width, src_height, width, height)

    frame_count = len(image_paths)
    poses = build_pose_sequence(frame_count)
    intrinsics = scaled_intr.repeat(frame_count, 1)

    frame_data: list[FrameData] = []
    fmap1_frames: list[torch.Tensor] = []
    fmap2_frames: list[torch.Tensor] = []
    patch_list: list[torch.Tensor] = []
    gmap_list: list[torch.Tensor] = []
    patch_imap_list: list[torch.Tensor] = []

    for frame_index, image_path in enumerate(image_paths):
        image = load_normalized_image(image_path, height, width)
        image_batched = image.unsqueeze(0).unsqueeze(0)
        with torch.inference_mode():
            raw_fmap = model.patchify.fnet(image_batched)
            raw_imap = model.patchify.inet(image_batched)

        fmap = (raw_fmap * 0.25).squeeze(0).squeeze(0).contiguous()
        imap_full = (raw_imap * 0.25).squeeze(0).squeeze(0).contiguous()
        feature_height = fmap.shape[1]
        feature_width = fmap.shape[2]
        centers = select_patch_centers(image, patches_per_frame)
        depths = build_depths(centers, frame_index, frame_count, feature_width, feature_height)

        patch_imap = patchify_single_chw(imap_full, centers, radius=0)
        gmap = patchify_single_chw(fmap, centers, radius=1)
        colors = patchify_single_chw(image, 4.0 * (centers + 0.5), radius=0)[:, :, 0, 0]
        grid = build_grid_tensor(feature_height, feature_width)
        patches = patchify_single_chw(grid, centers, radius=1)
        patches[:, 2, :, :] = depths.view(-1, 1, 1)

        frame_data.append(
            FrameData(
                image=image,
                centers=centers,
                fmap=fmap.unsqueeze(0).unsqueeze(0),
                patch_imap=patch_imap,
                gmap=gmap,
                patches=patches,
                colors=colors,
            )
        )
        fmap1_frames.append(fmap)
        fmap2_frames.append(F.avg_pool2d(fmap.unsqueeze(0), 4, 4).squeeze(0).contiguous())
        patch_list.append(patches)
        gmap_list.append(gmap)
        patch_imap_list.append(patch_imap)

    patches = flatten_patch_tensor(patch_list)
    gmap = flatten_patch_tensor(gmap_list)
    patch_imap = flatten_patch_tensor(patch_imap_list)
    fmap1 = torch.stack(fmap1_frames, dim=0).contiguous()
    fmap2 = torch.stack(fmap2_frames, dim=0).contiguous()
    ii, jj, kk = build_edge_graph(frame_count, patches_per_frame)

    poses_se3 = SE3(poses.unsqueeze(0))
    patches_batched = patches.unsqueeze(0)
    intrinsics_batched = intrinsics.unsqueeze(0)

    with torch.inference_mode():
        coords_python = pops.transform(poses_se3, patches_batched, intrinsics_batched, ii, jj, kk)
    coords = coords_python.squeeze(0).permute(0, 3, 1, 2).contiguous()
    corr = build_correlation_volume(gmap[kk], fmap1, fmap2, jj, coords)

    ctx = patch_imap[kk, :, 0, 0].unsqueeze(0).contiguous()
    net = torch.zeros(1, kk.numel(), DIM, dtype=torch.float32)
    ix, jx = compute_neighbor_indices(kk, jj)

    with torch.inference_mode():
        update_net, update_delta, update_weight = update(net, ctx, corr, ii, jj, kk, ix, jx)

    target = coords_python[:, :, PATCH_SIZE // 2, PATCH_SIZE // 2, :] + update_delta
    bounds = torch.tensor(
        [-64.0, -64.0, float(fmap1.shape[-1] + 64), float(fmap1.shape[-2] + 64)],
        dtype=torch.float32,
    )
    fixed_pose_count = torch.tensor([1], dtype=torch.int64)

    with torch.inference_mode():
        ba_poses, ba_patches = BA(
            poses_se3,
            patches_batched,
            intrinsics_batched,
            target,
            update_weight,
            1e-4,
            ii,
            jj,
            kk,
            bounds.tolist(),
            ep=1.0,
            fixedp=int(fixed_pose_count.item()),
        )

    return SequenceData(
        frames=frame_data,
        poses=poses,
        intrinsics=intrinsics,
        patches=patches,
        gmap=gmap,
        fmap1=fmap1,
        fmap2=fmap2,
        ii=ii,
        jj=jj,
        kk=kk,
        coords=coords,
        corr=corr,
        ctx=ctx,
        net=net,
        update_net=update_net,
        update_delta=update_delta,
        update_weight=update_weight,
        ba_poses=ba_poses.data.squeeze(0).contiguous(),
        ba_patches=ba_patches.squeeze(0).contiguous(),
        bounds=bounds,
        fixed_pose_count=fixed_pose_count,
    )
