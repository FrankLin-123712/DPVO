#!/bin/bash
set -euo pipefail

# Make globbing case-insensitive so .MOV works too, and ignore when empty
shopt -s nocaseglob nullglob

mkdir -p sequences

files=(movies/*.mov)
if (( ${#files[@]} == 0 )); then
  echo "No .mov/.MOV files found under movies/. Nothing to convert."
  exit 1
fi

for f in "${files[@]}"; do
  name=$(basename "$f" .mov)
  out="sequences/$name"
  mkdir -p "$out"
  ffmpeg -i "$f" -vf "fps=30" -q:v 2 "$out/%06d.png"
done
