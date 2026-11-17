#!/usr/bin/env bash
# Independent builds: deliberately do not share a TensorRT timing cache.
set -euo pipefail
cd "$(dirname "$0")/.."
out=artifacts/trt_tuning
mkdir -p "$out"
for stem in fp16_640_o3_b1 fp16_640_o3_b2 fp16_640_o5_b1 fp16_640_o5_b2 fp16_384x640_o5; do
    level=5
    onnx=artifacts/yolov10s_fp16.onnx
    if [[ "$stem" == *_o3_* ]]; then level=3; fi
    if [[ "$stem" == fp16_384x640_o5 ]]; then onnx=artifacts/yolov10s_fp16_384x640.onnx; fi
    plan="$out/$stem.plan"
    prefix="$out/build_$stem"
    rm -f "$plan" "$prefix.log" "$prefix.seconds"
    start=$(date +%s.%N)
    trtexec --onnx="$onnx" --saveEngine="$plan" --stronglyTyped --skipInference \
        --builderOptimizationLevel="$level" 2>&1 | tee "$prefix.log"
    finish=$(date +%s.%N)
    seconds=$(awk -v start="$start" -v finish="$finish" 'BEGIN {printf "%.6f", finish-start}')
    printf '%s\n' "$seconds" > "$prefix.seconds"
    bytes=$(stat -c %s "$plan")
    awk -v stem="$stem" -v s="$seconds" -v b="$bytes" \
        'BEGIN {printf "%s: build %.3f s | plan %.3f MiB\n", stem,s,b/1048576}'
done
