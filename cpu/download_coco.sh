#!/usr/bin/env bash
# Download COCO val2017 images and annotations for round 3 (INT8 calibration and mAP).
# Destination: ~/.cache/compiled-cnn/coco (outside the repo, reused across runs).
# Skips anything already present. Needs curl and unzip; about 1.1 GB of downloads.
set -euo pipefail

dest="$HOME/.cache/compiled-cnn/coco"   # fixed: cpu/common.py reads this path
mkdir -p "$dest"
cd "$dest"

fetch() {
    local url="$1" zip="$2" check="$3"
    if [[ -e "$check" ]]; then
        echo "ok: $check already present"
        return
    fi
    [[ -s "$zip" ]] || curl -fL --retry 3 -o "$zip" "$url"
    unzip -q -o "$zip"
    [[ -e "$check" ]] || { echo "error: $check missing after unzip" >&2; exit 1; }
    echo "ok: $check"
}

fetch http://images.cocodataset.org/zips/val2017.zip val2017.zip val2017/000000000139.jpg
fetch http://images.cocodataset.org/annotations/annotations_trainval2017.zip \
    annotations_trainval2017.zip annotations/instances_val2017.json

count=$(find val2017 -name '*.jpg' | wc -l)
echo "COCO val2017 ready in $dest ($count images)"
[[ "$count" -eq 5000 ]] || { echo "error: expected 5000 images" >&2; exit 1; }
