#!/usr/bin/env bash
# Run from any directory. Keep default optimization; compare default FP32 with --noTF32.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p artifacts/trt
# TRT 11's trtexec prints only "Strongly Typed", omitting the TF32 flag.
# Query the unchanged builder defaults rather than infer precision from timing.
"${CXX:-c++}" -std=c++17 -I/opt/cuda/include -x c++ - -o artifacts/trt/builder_defaults -lnvinfer <<'CPP'
#include <NvInfer.h>
#include <iostream>
#include <memory>
struct Logger : nvinfer1::ILogger {
    void log(Severity s, const char* msg) noexcept override {
        if (s <= Severity::kWARNING) std::cerr << msg << '\n';
    }
};
int main() {
    Logger logger;
    std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
    if (!builder) return 1;
    std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    if (!config) return 1;
    std::cout << "tf32=" << (config->getFlag(nvinfer1::BuilderFlag::kTF32) ? "on" : "off")
              << " optimization_level=" << config->getBuilderOptimizationLevel() << '\n';
}
CPP
artifacts/trt/builder_defaults | tee artifacts/trt/builder_defaults.log
default_tf32=$(awk '{sub("tf32=", "", $1); print $1}' artifacts/trt/builder_defaults.log)
[[ "$default_tf32" == on || "$default_tf32" == off ]]
for variant in fp32 fp16 fp32_notf32; do
    precision=${variant%_notf32}
    stem="yolov10s_${variant}"
    plan="artifacts/${stem}.plan"
    prefix="artifacts/trt/build_${stem}"
    log="${prefix}.log"
    flags=()
    if [[ "$variant" == fp32_notf32 ]]; then flags+=(--noTF32); fi
    # Do not leave stale metadata/a plan looking successful after a failed build.
    rm -f "$plan" "${prefix}.seconds" "${prefix}.tf32"
    start=$(date +%s.%N)
    trtexec --onnx="artifacts/yolov10s_${precision}.onnx" --saveEngine="$plan" \
        --stronglyTyped --skipInference "${flags[@]}" 2>&1 | tee "$log"
    finish=$(date +%s.%N)
    seconds=$(awk -v start="$start" -v finish="$finish" 'BEGIN {printf "%.6f", finish-start}')
    printf '%s\n' "$seconds" > "${prefix}.seconds"
    # The log records builder precision policy, not proof every FP32 layer uses TF32.
    if [[ "$variant" == fp32_notf32 ]]; then
        tf32=off
        printf 'TF32 explicitly disabled: --noTF32\n' | tee -a "$log"
    elif [[ "$precision" == fp16 ]]; then
        tf32=n/a
    elif grep -Eq 'Precision:.*TF32' "$log"; then
        tf32=on
    elif grep -Eq 'Precision:.*FP32' "$log"; then
        tf32=off
    elif grep -q 'Precision: Strongly Typed' "$log"; then
        tf32=$default_tf32
        printf 'Builder defaults (unchanged): %s\n' "$(< artifacts/trt/builder_defaults.log)" | tee -a "$log"
    else
        echo "Cannot determine TF32 policy from $log" >&2
        exit 1
    fi
    printf '%s\n' "$tf32" > "${prefix}.tf32"
    bytes=$(stat -c %s "$plan")
    awk -v p="$variant" -v s="$seconds" -v b="$bytes" -v tf32="$tf32" \
        'BEGIN {printf "%s: build %.3f s | plan %.3f MiB | tf32: %s\n", p,s,b/1048576,tf32}'
done
