#!/usr/bin/env python3
"""Download EuRoC MAV files from the ETH Research Collection.

The default command downloads the files listed on the EuRoC Research Collection
page:

  * Machine Hall Datasets
  * Vicon Room 1 Datasets
  * Vicon Room 2 Datasets
  * Calibration Datasets
  * euroc_mav_dataset.pdf

Dataset archives are also extracted and promoted into the layout expected by
DPVO/evaluate_euroc.py:

    datasets/EUROC/<sequence>/mav0/cam0/data/*.png

The source archives are kept under datasets/EUROC/_archives by default because
the task is to download the listed files.  Use --delete-archives-after-extract
to save disk space after a successful extraction.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = None


DPVO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EUROC_DIR = DPVO_ROOT / "datasets" / "EUROC"
DOWNLOAD_CHUNK_BYTES = 1024 * 1024
FALLBACK_PROGRESS_BYTES = 256 * 1024 * 1024

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

SCENE_GROUPS = {
    "machine_hall": [
        "MH_01_easy",
        "MH_02_easy",
        "MH_03_medium",
        "MH_04_difficult",
        "MH_05_difficult",
    ],
    "vicon_room_1": [
        "V1_01_easy",
        "V1_02_medium",
        "V1_03_difficult",
    ],
    "vicon_room_2": [
        "V2_01_easy",
        "V2_02_medium",
        "V2_03_difficult",
    ],
}


@dataclass(frozen=True)
class DownloadItem:
    key: str
    label: str
    filename: str
    url: str
    kind: str
    scenes: tuple[str, ...] = ()


DOWNLOAD_ITEMS = {
    "machine_hall": DownloadItem(
        key="machine_hall",
        label="Machine Hall Datasets",
        filename="machine_hall_datasets.zip",
        url="https://www.research-collection.ethz.ch/server/api/core/bitstreams/7b2419c1-62b5-4714-b7f8-485e5fe3e5fe/content",
        kind="sequence_zip",
        scenes=tuple(SCENE_GROUPS["machine_hall"]),
    ),
    "vicon_room_1": DownloadItem(
        key="vicon_room_1",
        label="Vicon Room 1 Datasets",
        filename="vicon_room_1_datasets.zip",
        url="https://www.research-collection.ethz.ch/server/api/core/bitstreams/02ecda9a-298f-498b-970c-b7c44334d880/content",
        kind="sequence_zip",
        scenes=tuple(SCENE_GROUPS["vicon_room_1"]),
    ),
    "vicon_room_2": DownloadItem(
        key="vicon_room_2",
        label="Vicon Room 2 Datasets",
        filename="vicon_room_2_datasets.zip",
        url="https://www.research-collection.ethz.ch/server/api/core/bitstreams/ea12bc01-3677-4b4c-853d-87c7870b8c44/content",
        kind="sequence_zip",
        scenes=tuple(SCENE_GROUPS["vicon_room_2"]),
    ),
    "calibration": DownloadItem(
        key="calibration",
        label="Calibration Datasets",
        filename="calibration_datasets.zip",
        url="https://www.research-collection.ethz.ch/server/api/core/bitstreams/5732e864-10f1-49e7-befb-669ee29ff770/content",
        kind="calibration_zip",
    ),
    "pdf": DownloadItem(
        key="pdf",
        label="euroc_mav_dataset.pdf",
        filename="euroc_mav_dataset.pdf",
        url="https://www.research-collection.ethz.ch/server/api/core/bitstreams/d861e63b-cfa9-4411-85a5-5ad6b3526e44/content",
        kind="document",
    ),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eurocdir", type=Path, default=DEFAULT_EUROC_DIR)
    parser.add_argument(
        "--archive-dir",
        type=Path,
        help="Directory for downloaded ZIP/PDF files. Default: <eurocdir>/_archives.",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        choices=tuple(DOWNLOAD_ITEMS),
        help=(
            "Research Collection files to download. Default: files needed for "
            "--scenes, plus calibration and pdf."
        ),
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        choices=EUROC_SCENES,
        default=EUROC_SCENES,
        help=(
            "EuRoC scenes that should be available after extraction. Because "
            "Research Collection downloads are grouped, this selects the "
            "corresponding Machine Hall / Vicon Room bundle(s)."
        ),
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Only download the selected files; do not extract ZIP archives.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace already-extracted sequence/calibration directories.",
    )
    parser.add_argument(
        "--delete-archives-after-extract",
        action="store_true",
        help="Delete downloaded ZIP archives after successful extraction. The PDF is always kept.",
    )
    parser.add_argument(
        "--print-urls",
        action="store_true",
        help="Print selected Research Collection URLs and exit without downloading.",
    )
    return parser.parse_args(argv)


def selected_items(args: argparse.Namespace) -> list[DownloadItem]:
    if args.files:
        return [DOWNLOAD_ITEMS[key] for key in args.files]

    selected_keys: list[str] = []
    requested_scenes = set(args.scenes)
    for key, scenes in SCENE_GROUPS.items():
        if requested_scenes.intersection(scenes):
            selected_keys.append(key)
    selected_keys.extend(["calibration", "pdf"])
    return [DOWNLOAD_ITEMS[key] for key in selected_keys]


def scene_is_present(scene_dir: Path) -> bool:
    return any((scene_dir / "mav0" / "cam0" / "data").glob("*.png"))


def nonempty_file_exists(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def response_content_length(response: object) -> int | None:
    length = response.headers.get("Content-Length")
    if length is None:
        return None
    try:
        parsed = int(length)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def copy_with_progress(response: object, destination: Path, label: str) -> None:
    total = response_content_length(response)
    copied = 0
    next_report = FALLBACK_PROGRESS_BYTES

    if tqdm is not None:
        with tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=label,
            leave=True,
        ) as progress:
            while True:
                chunk = response.read(DOWNLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                destination.write(chunk)
                progress.update(len(chunk))
        return

    while True:
        chunk = response.read(DOWNLOAD_CHUNK_BYTES)
        if not chunk:
            break
        destination.write(chunk)
        copied += len(chunk)
        if copied >= next_report:
            if total is None:
                print(f"{label}: downloaded {copied / (1024 * 1024):.1f} MiB")
            else:
                percent = 100.0 * copied / total
                print(
                    f"{label}: downloaded {copied / (1024 * 1024):.1f} MiB / "
                    f"{total / (1024 * 1024):.1f} MiB ({percent:.1f}%)"
                )
            next_report += FALLBACK_PROGRESS_BYTES
    if total is not None:
        print(f"{label}: downloaded {copied / (1024 * 1024):.1f} MiB")


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "Accept": "application/octet-stream,*/*",
                    "User-Agent": "dpvo-statistic-downloader",
                },
            )
            with urllib.request.urlopen(request) as response, temporary.open("wb") as handle:
                copy_with_progress(response, handle, destination.name)
            temporary.replace(destination)
            return
        except Exception as exc:
            last_error = exc
            if temporary.exists():
                temporary.unlink()
            if attempt < 3:
                print(f"download failed, retrying ({attempt}/3): {exc}")
                time.sleep(2 * attempt)
    assert last_error is not None
    raise last_error


def remove_existing(path: Path, overwrite: bool) -> bool:
    if not path.exists():
        return True
    if not overwrite:
        return False
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    return True


def extract_nested_zips(root: Path) -> None:
    for nested_zip in sorted(root.rglob("*.zip")):
        nested_target = nested_zip.with_suffix("")
        if nested_target.exists():
            continue
        nested_target.mkdir(parents=True)
        with zipfile.ZipFile(nested_zip) as zipped:
            zipped.extractall(nested_target)


def find_sequence_dirs(root: Path, expected_scenes: Iterable[str]) -> dict[str, Path]:
    expected = set(expected_scenes)
    found: dict[str, Path] = {}
    for data_dir in root.rglob("mav0/cam0/data"):
        scene_dir = data_dir.parents[2]
        scene_name = scene_dir.name
        if scene_name in expected:
            found.setdefault(scene_name, scene_dir)
    return found


def promote_sequence_dirs(
    extraction_root: Path,
    eurocdir: Path,
    expected_scenes: Sequence[str],
    overwrite: bool,
) -> None:
    found = find_sequence_dirs(extraction_root, expected_scenes)
    missing = [scene for scene in expected_scenes if scene not in found]
    if missing:
        raise RuntimeError(
            "archive did not contain expected ASL-format sequence(s): " + ", ".join(missing)
        )

    for scene in expected_scenes:
        target = eurocdir / scene
        if scene_is_present(target) and not overwrite:
            print(f"{scene}: already extracted")
            continue
        if not remove_existing(target, overwrite=True):
            raise RuntimeError(f"failed to remove existing sequence directory: {target}")
        found[scene].replace(target)
        print(f"{scene}: extracted")


def extract_sequence_archive(item: DownloadItem, archive: Path, eurocdir: Path, overwrite: bool) -> None:
    temporary = eurocdir / f".{item.key}.extracting"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        with zipfile.ZipFile(archive) as zipped:
            zipped.extractall(temporary)
        extract_nested_zips(temporary)
        promote_sequence_dirs(temporary, eurocdir, item.scenes, overwrite)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def extract_calibration_archive(archive: Path, eurocdir: Path, overwrite: bool) -> None:
    target = eurocdir / "calibration_datasets"
    if target.exists() and not overwrite:
        print("Calibration Datasets: already extracted")
        return
    temporary = eurocdir / ".calibration.extracting"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        with zipfile.ZipFile(archive) as zipped:
            zipped.extractall(temporary)
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)
        for child in temporary.iterdir():
            child.replace(target / child.name)
        print("Calibration Datasets: extracted")
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def print_selected_urls(items: Sequence[DownloadItem]) -> None:
    for item in items:
        print(f"{item.key} {item.label} {item.url}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    items = selected_items(args)
    if args.print_urls:
        print_selected_urls(items)
        return 0

    args.eurocdir.mkdir(parents=True, exist_ok=True)
    archive_dir = args.archive_dir or (args.eurocdir / "_archives")
    archive_dir.mkdir(parents=True, exist_ok=True)

    for item in items:
        destination = archive_dir / item.filename
        if nonempty_file_exists(destination):
            print(f"{item.label}: already downloaded -> {destination}")
        else:
            print(f"{item.label}: downloading {item.url}")
            try:
                download(item.url, destination)
            except Exception as exc:
                print(f"error: failed to download {item.label}: {exc}", file=sys.stderr)
                return 2

        if args.no_extract or item.kind == "document":
            continue

        try:
            if item.kind == "sequence_zip":
                extract_sequence_archive(item, destination, args.eurocdir, args.overwrite)
            elif item.kind == "calibration_zip":
                extract_calibration_archive(destination, args.eurocdir, args.overwrite)
        except Exception as exc:
            print(f"error: failed to extract {item.label}: {exc}", file=sys.stderr)
            return 2

        if args.delete_archives_after_extract and destination.suffix == ".zip":
            destination.unlink()
            print(f"{item.label}: deleted archive after extraction")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
