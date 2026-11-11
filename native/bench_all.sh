#!/usr/bin/env bash
# Six isolated processes, alternating backends within each precision.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p artifacts/bench
{
    for precision in fp32 fp16; do
        uv run python aoti/bench_eager.py --precision "$precision"
        env -u LD_LIBRARY_PATH aoti/cpp/build/yolo_aoti bench "artifacts/yolov10s_${precision}.pt2"
        env -u LD_LIBRARY_PATH trt/cpp/build/yolo_trt bench "artifacts/yolov10s_${precision}.plan"
    done
} | tee artifacts/bench/round2.txt
