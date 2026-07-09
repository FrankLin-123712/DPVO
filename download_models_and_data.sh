#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT_DIR"

if ! command -v gdown >/dev/null 2>&1; then
    echo "gdown is required. Install it with: python -m pip install gdown" >&2
    exit 1
fi

mkdir -p datasets

download() {
    local file_id=$1
    local output=$2
    local temporary="${output}.part"

    rm -f "$temporary"
    gdown "https://drive.google.com/uc?id=${file_id}" --output "$temporary"
}

download_pickle() {
    local file_id=$1
    local output=$2
    local temporary="${output}.part"

    download "$file_id" "$output"
    if [[ $(file --brief --mime-type "$temporary") == text/html ]]; then
        echo "Downloaded an HTML page instead of $output" >&2
        rm -f "$temporary"
        exit 1
    fi
    mv -f "$temporary" "$output"
}

download_archive() {
    local file_id=$1
    local output=$2
    local temporary="${output}.part"

    download "$file_id" "$output"
    if ! unzip -tq "$temporary" >/dev/null; then
        echo "Downloaded file is not a valid ZIP archive: $output" >&2
        rm -f "$temporary"
        exit 1
    fi
    mv -f "$temporary" "$output"
    unzip -o "$output"
}

download_pickle 1sct2wlHzfbv57Hu6xkpGq7Trv_DUy5Cc datasets/TartanAir.pickle
download_archive 1dRqftpImtHbbIPNBIseCv9EvrlHEnjhX models.zip
download_archive 1bS-oe3WQZZUycPhGs95l9D1ntqFKKhKJ movies.zip
