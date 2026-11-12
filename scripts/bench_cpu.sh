#!/usr/bin/env bash
# Three thermally controlled rounds; never overlap calibration/download/COCO eval.
set -euo pipefail
cd "$(dirname "$0")/.."
runs=artifacts/bench/round3-runs
mkdir -p "$runs"
# A failed new session must not leave a stale successful summary or run behind.
rm -f artifacts/bench/round3-summary.{txt,json} "$runs"/round-{1,2,3}-*.json
variants=(eager-cpu-fp32 ort-cpu-fp32 ov-cpu-fp32 ov-cpu-int8 ov-cpu-fp32-default)
{
  uv run python cpu/bench.py --profile
  for round in 1 2 3; do
    shift=$(( (round - 1) * 2 ))
    for index in 0 1 2 3 4; do
      variant=${variants[$(( (index + shift) % 5 ))]}
      output="$runs/round-$round-$variant.json"
      # Initial attempt plus at most five retries, each with its own cooldown.
      for attempt in 0 1 2 3 4 5; do
        echo "round: $round | variant: $variant | attempt: $((attempt + 1))/6 | cooldown: 60 s"
        sleep 60
        status=0
        uv run python cpu/bench.py --check-host || status=$?
        if (( status == 0 )); then
          command=(uv run python cpu/bench.py --variant "$variant" --output-json "$output")
          if [[ "$variant" != ov-cpu-fp32-default ]]; then
            command=(taskset -c 0,2,4,6,8,10 "${command[@]}")
          fi
          "${command[@]}" || status=$?
        fi
        if (( status == 0 )); then
          break
        fi
        if (( status != 75 )); then
          echo "ERROR: round $round $variant failed (exit $status); not a contention discard" >&2
          exit "$status"
        fi
        if (( attempt == 5 )); then
          echo "ERROR: round $round $variant discarded after initial attempt and five retries" >&2
          exit 75
        fi
        echo "retry: contention discard; next attempt follows a fresh 60 s cooldown"
      done
    done
  done
  uv run python cpu/bench.py --summarize "$runs"
} 2>&1 | tee artifacts/bench/round3.txt
