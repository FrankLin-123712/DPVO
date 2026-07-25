#!/usr/bin/env python3
"""Download KITTI odometry files into DPVO/datasets/KITTI.

Default downloads:
  * data_odometry_color.zip      (color images, about 65 GB)
  * data_odometry_calib.zip      (calibration files)
  * data_odometry_poses.zip      (ground-truth poses for 00-10)
  * devkit_odometry.zip          (official development kit)

The archives are kept under <kittidir>/_archives by default and extracted into
<kittidir>, yielding the layout expected by DPVO:

    datasets/KITTI/dataset/sequences/00/image_2/*.png
    datasets/KITTI/dataset/sequences/00/calib.txt
    datasets/KITTI/dataset/poses/00.txt

KITTI credentials are not stored in this file.  The public S3 URLs normally do
not require login.  If KITTI changes that behavior, pass --username/--password
or KITTI_USERNAME/KITTI_PASSWORD so the script can attempt a cookie login before
retrying downloads.
"""

from __future__ import annotations

import argparse
import getpass
import http.cookiejar
import os
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = None


DPVO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KITTI_DIR = DPVO_ROOT / "datasets" / "KITTI"
DOWNLOAD_CHUNK_BYTES = 1024 * 1024
FALLBACK_PROGRESS_BYTES = 512 * 1024 * 1024
LOGIN_URL = "https://www.cvlibs.net/datasets/kitti/user_login.php"


@dataclass(frozen=True)
class DownloadItem:
    key: str
    label: str
    filename: str
    url: str


DOWNLOAD_ITEMS = {
    "color": DownloadItem(
        key="color",
        label="odometry color images",
        filename="data_odometry_color.zip",
        url="https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_color.zip",
    ),
    "calib": DownloadItem(
        key="calib",
        label="odometry calibration files",
        filename="data_odometry_calib.zip",
        url="https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_calib.zip",
    ),
    "poses": DownloadItem(
        key="poses",
        label="odometry ground-truth poses",
        filename="data_odometry_poses.zip",
        url="https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_poses.zip",
    ),
    "devkit": DownloadItem(
        key="devkit",
        label="odometry development kit",
        filename="devkit_odometry.zip",
        url="https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_odometry.zip",
    ),
    "gray": DownloadItem(
        key="gray",
        label="odometry grayscale images",
        filename="data_odometry_gray.zip",
        url="https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_gray.zip",
    ),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kittidir", type=Path, default=DEFAULT_KITTI_DIR)
    parser.add_argument("--archive-dir", type=Path, help="Default: <kittidir>/_archives")
    parser.add_argument(
        "--files",
        nargs="+",
        choices=tuple(DOWNLOAD_ITEMS),
        default=["color", "calib", "poses", "devkit"],
        help="Files to download. Default: color calib poses devkit.",
    )
    parser.add_argument("--skip-color", action="store_true", help="Shortcut to omit the 65 GB color archive.")
    parser.add_argument("--no-extract", action="store_true", help="Download only; do not unzip archives.")
    parser.add_argument("--overwrite", action="store_true", help="Re-download existing archives and overwrite extracted files.")
    parser.add_argument("--delete-archives-after-extract", action="store_true")
    parser.add_argument("--username", default=os.environ.get("KITTI_USERNAME"))
    parser.add_argument("--password", default=os.environ.get("KITTI_PASSWORD"))
    parser.add_argument("--ask-password", action="store_true", help="Prompt for KITTI password without echoing it.")
    parser.add_argument("--login-url", default=LOGIN_URL)
    parser.add_argument("--print-urls", action="store_true", help="Print selected URLs and exit.")
    return parser.parse_args(argv)


def selected_items(args: argparse.Namespace) -> list[DownloadItem]:
    keys = list(args.files)
    if args.skip_color:
        keys = [key for key in keys if key != "color"]
    return [DOWNLOAD_ITEMS[key] for key in keys]


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


def make_request(url: str, start_byte: int = 0) -> urllib.request.Request:
    headers = {"User-Agent": "dpvo-kitti-downloader/1.0"}
    if start_byte:
        headers["Range"] = f"bytes={start_byte}-"
    return urllib.request.Request(url, headers=headers)


def copy_with_progress(response: object, handle: object, label: str, initial: int = 0) -> None:
    total = response_content_length(response)
    if total is not None:
        total += initial
    copied = initial
    next_report = initial + FALLBACK_PROGRESS_BYTES

    if tqdm is not None:
        with tqdm(
            total=total,
            initial=initial,
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
                handle.write(chunk)
                progress.update(len(chunk))
        return

    while True:
        chunk = response.read(DOWNLOAD_CHUNK_BYTES)
        if not chunk:
            break
        handle.write(chunk)
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


def login_opener(username: str | None, password: str | None, login_url: str) -> urllib.request.OpenerDirector:
    cookie_jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))
    if not username or not password:
        return opener

    candidates = [
        {"email": username, "password": password, "submit": "Login"},
        {"user": username, "pass": password, "action": "login", "submit": "Login"},
    ]
    for fields in candidates:
        encoded = urllib.parse.urlencode(fields).encode("utf-8")
        request = urllib.request.Request(
            login_url,
            data=encoded,
            headers={"User-Agent": "dpvo-kitti-downloader/1.0"},
        )
        try:
            with opener.open(request, timeout=30) as response:
                response.read(4096)
        except urllib.error.URLError:
            continue
    return opener


def download(opener: urllib.request.OpenerDirector, url: str, destination: Path, overwrite: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if nonempty_file_exists(destination) and not overwrite:
        print(f"skip existing {destination}")
        return

    temporary = destination.with_suffix(destination.suffix + ".part")
    existing = temporary.stat().st_size if temporary.exists() and not overwrite else 0
    if overwrite and temporary.exists():
        temporary.unlink()
        existing = 0

    for attempt in range(1, 4):
        try:
            request = make_request(url, existing)
            with opener.open(request, timeout=60) as response:
                status = getattr(response, "status", None)
                mode = "ab" if existing and status == 206 else "wb"
                if existing and status != 206:
                    existing = 0
                with temporary.open(mode) as handle:
                    copy_with_progress(response, handle, destination.name, existing)
            temporary.replace(destination)
            return
        except urllib.error.HTTPError as exc:
            if exc.code in {403, 404}:
                raise RuntimeError(f"failed to download {url}: HTTP {exc.code}") from exc
            print(f"download attempt {attempt} failed for {url}: HTTP {exc.code}", file=sys.stderr)
        except urllib.error.URLError as exc:
            print(f"download attempt {attempt} failed for {url}: {exc}", file=sys.stderr)
        time.sleep(2 * attempt)
    raise RuntimeError(f"failed to download after retries: {url}")


def extract_zip(archive: Path, destination: Path, overwrite: bool) -> None:
    if not archive.exists():
        raise FileNotFoundError(f"archive does not exist: {archive}")
    marker = destination / ".extract_complete" / archive.stem
    if marker.exists() and not overwrite:
        print(f"skip already extracted {archive.name}")
        return
    print(f"extract {archive.name} -> {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zip_file:
        zip_file.extractall(destination)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("ok\n")


def validate_layout(kittidir: Path) -> None:
    checks = [
        kittidir / "dataset" / "sequences" / "00" / "calib.txt",
        kittidir / "dataset" / "poses" / "00.txt",
    ]
    missing = [path for path in checks if not path.exists()]
    if missing:
        print("warning: expected KITTI files are still missing:", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
    image_dir = kittidir / "dataset" / "sequences" / "00" / "image_2"
    if not any(image_dir.glob("*.png")):
        print(f"warning: no color images found under {image_dir}", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    items = selected_items(args)
    archive_dir = args.archive_dir or (args.kittidir / "_archives")

    if args.ask_password and not args.password:
        args.password = getpass.getpass("KITTI password: ")

    if args.print_urls:
        for item in items:
            print(f"{item.key}\t{item.url}")
        return 0

    opener = login_opener(args.username, args.password, args.login_url)
    args.kittidir.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)

    for item in items:
        archive = archive_dir / item.filename
        print(f"download {item.label}: {item.url}")
        download(opener, item.url, archive, args.overwrite)
        if not args.no_extract:
            extract_zip(archive, args.kittidir, args.overwrite)
            if args.delete_archives_after_extract:
                archive.unlink(missing_ok=True)

    if not args.no_extract:
        validate_layout(args.kittidir)
    print(f"KITTI directory: {args.kittidir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
