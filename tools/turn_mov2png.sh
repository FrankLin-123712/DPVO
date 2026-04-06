#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

INPUT_DIR=${INPUT_DIR:-"$REPO_ROOT/movies"}
OUTPUT_DIR=${OUTPUT_DIR:-"$REPO_ROOT/sequences"}
FPS=${FPS:-30}

# Make globbing case-insensitive so .MOV works too, and ignore when empty.
shopt -s nocaseglob nullglob

mkdir -p "$OUTPUT_DIR"
files=("$INPUT_DIR"/*.mov)
if (( ${#files[@]} == 0 )); then
  echo "No .mov/.MOV files found under $INPUT_DIR. Nothing to convert."
  exit 1
fi

for f in "${files[@]}"; do
  base=$(basename "$f")
  name=${base%.*}
  out="$OUTPUT_DIR/$name"
  mkdir -p "$out"
  ffmpeg -i "$f" -vf "fps=$FPS" -q:v 2 "$out/%06d.png"
done
