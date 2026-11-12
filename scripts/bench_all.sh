#!/usr/bin/env bash
# Seven isolated processes; compare TF32 policies immediately after TRT FP32.
set -euo pipefail
cd "$(dirname "$0")/.."
BIN=${CNN_BIN:-build/bin}  # binaries from the top-level CMake build (cmake --preset release)
mkdir -p artifacts/bench
{
    for precision in fp32 fp16; do
        uv run python aoti/bench_eager.py --precision "$precision"
        env -u LD_LIBRARY_PATH "$BIN/yolo_aoti" bench "artifacts/yolov10s_${precision}.pt2"
        env -u LD_LIBRARY_PATH "$BIN/yolo_trt" bench "artifacts/yolov10s_${precision}.plan"
        if [[ "$precision" == fp32 ]]; then
            env -u LD_LIBRARY_PATH "$BIN/yolo_trt" bench artifacts/yolov10s_fp32_notf32.plan
        fi
    done
} | tee artifacts/bench/round2.txt
