#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
out=artifacts/trt_tuning
mkdir -p "$out"
profile() {
    trtexec --loadEngine="$1" --dumpProfile --separateProfileRun \
        --exportProfile="$out/profile_$stem.json" --warmUp=1000 --iterations=500 \
        2>&1 | tee "$out/profile_$stem.log"
}
for stem in fp16_640_o3_b1 fp16_384x640_o5; do
    rm -f "$out/profile_$stem.json" "$out/profile_$stem.log"
    profile "$out/$stem.plan"
    if ! uv run python trt/profile_top.py "$out/profile_$stem.json"; then
        # Never replace a benchmarked engine to obtain more detailed profiling.
        level=3
        onnx=artifacts/yolov10s_fp16.onnx
        if [[ "$stem" == fp16_384x640_o5 ]]; then
            level=5
            onnx=artifacts/yolov10s_fp16_384x640.onnx
        fi
        rm -f "$out/profile_$stem.plan"
        trtexec --onnx="$onnx" --saveEngine="$out/profile_$stem.plan" \
            --stronglyTyped --skipInference --builderOptimizationLevel="$level" \
            --profilingVerbosity=detailed 2>&1 | tee "$out/build_profile_$stem.log"
        echo 'note: profiled a separate detailed build, not the benchmarked plan'
        profile "$out/profile_$stem.plan"
        uv run python trt/profile_top.py "$out/profile_$stem.json"
    fi
done
