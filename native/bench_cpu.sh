#!/usr/bin/env bash
# Sequential processes; never run alongside calibration, download or COCO eval.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p artifacts/bench
{
  uv run python cpu/bench.py --check-host
  uv run python cpu/bench.py --profile
  for variant in eager-cpu-fp32 ort-cpu-fp32 ov-cpu-fp32 ov-cpu-int8 ov-cpu-fp32-default; do
    if [[ "$variant" == ov-cpu-fp32-default ]]; then
      uv run python cpu/bench.py --variant "$variant"
    else
      taskset -c 0,2,4,6,8,10 uv run python cpu/bench.py --variant "$variant"
    fi
  done
} 2>&1 | tee artifacts/bench/round3.txt
