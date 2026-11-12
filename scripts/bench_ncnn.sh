#!/usr/bin/env bash
# Five variants by default (nine with --with-intel), three rotating rounds.
set -euo pipefail
cd "$(dirname "$0")/.."
BIN=${CNN_BIN:-build/bin}  # binaries from the top-level CMake build (cmake --preset release)
variants=(nvidia-fp32-resident nvidia-fp16-resident nvidia-fp32-host nvidia-fp16-host cpu-fp32-host)
if (( $# == 1 )) && [[ "$1" == --with-intel ]]; then
  variants+=(intel-fp32-resident intel-fp16-resident intel-fp32-host intel-fp16-host)
elif (( $# != 0 )); then
  echo 'usage: bash scripts/bench_ncnn.sh [--with-intel]' >&2
  exit 2
fi
count=${#variants[@]}
# Coprime shifts visit every starting position for both five and nine variants.
rotation=2
if (( count == 9 )); then rotation=4; fi
runs=artifacts/bench/round4-runs
mkdir -p "$runs"
rm -f artifacts/bench/round4-summary.txt "$runs"/round-{1,2,3}-*.txt
{
  echo "session: variants=$count | rounds=3 | rotation=$rotation | Intel opt-in=$((count == 9))"
  for round in 1 2 3; do
    shift=$(( (round - 1) * rotation ))
    for (( index=0; index<count; ++index )); do
      variant=${variants[$(( (index + shift) % count ))]}
      IFS=- read -r device precision mode <<< "$variant"
      output="$runs/round-$round-$variant.txt"
      for attempt in 0 1 2 3 4 5; do
        echo "round: $round | variant: $variant | attempt: $((attempt + 1))/6 | cooldown: 60 s"
        sleep 60
        status=0
        uv run python cpu/bench.py --check-host || status=$?
        if (( status == 0 )); then
          command=("$BIN/yolo_ncnn" bench artifacts/ncnn/yolov10s.ncnn.param
                   artifacts/ncnn/yolov10s.ncnn.bin --device "$device" --precision "$precision" --mode "$mode")
          if [[ "$device" == cpu ]]; then
            command=(taskset -c 0,2,4,6,8,10 "${command[@]}")
          fi
          # Reuse the unchanged monitor during the native run, too. Its measured
          # child is excluded just as cpu/bench.py excludes its own measured PID.
          # No other process or compiler is exempted; the thresholds are unchanged.
          uv run python - "$output" "${command[@]}" <<'PY' || status=$?
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path('cpu').resolve()))
from bench import ContentionMonitor

monitor = ContentionMonitor()
process = subprocess.Popen(sys.argv[2:], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
original = monitor.psutil.process_iter

def external_processes(*args, **kwargs):
    return (p for p in original(*args, **kwargs) if p.pid != process.pid)

monitor.psutil.process_iter = external_processes
monitor.thread.start()
try:
    text, _ = process.communicate()
finally:
    # On interruption do not orphan a native benchmark process.
    if process.poll() is None:
        process.terminate()
        process.wait()
    monitor.stop.set()
    monitor.thread.join()
print(text, end='', flush=True)
# A model error is not a contention retry, even if contention also occurred.
if process.returncode:
    sys.exit(process.returncode if process.returncode > 0 else 1)
try:
    monitor.finish()
except RuntimeError as error:
    print(str(error), file=sys.stderr, flush=True)
    sys.exit(75)
Path(sys.argv[1]).write_text(text)
PY
        fi
        if (( status == 0 )); then break; fi
        if (( status != 75 )); then
          echo "ERROR: round $round $variant failed (exit $status); not a contention discard" >&2
          exit "$status"
        fi
        if (( attempt == 5 )); then
          echo "ERROR: round $round $variant discarded after initial attempt and five retries" >&2
          exit 75
        fi
        echo 'retry: contention discard; next attempt follows a fresh 60 s cooldown'
      done
    done
  done
  uv run python - "$runs" "${variants[@]}" <<'PY'
from pathlib import Path
import re
from statistics import median
import sys

runs = Path(sys.argv[1])
text = ('summary: medians of 3 accepted runs; 60 s cooldown before each attempt\n'
        'variant | median mean ms | median p95 ms | median FPS | min/max mean ms\n')
for variant in sys.argv[2:]:
    rows = []
    for round in (1, 2, 3):
        block = (runs / f'round-{round}-{variant}.txt').read_text()
        device, precision, mode = variant.split('-')
        prefix = 'ncnn-cpu' if device == 'cpu' else f'ncnn-vk-{device}'
        if f'variant: {prefix} {precision} {mode}\n' not in block:
            raise ValueError(f'wrong variant: {variant}, round {round}')
        latency = re.search(r'latency ms: mean ([\d.]+) \| p50 ([\d.]+) \| p95 ([\d.]+)', block)
        fps = re.search(r'throughput: ([\d.]+) it/s', block)
        if not latency or not fps:
            raise ValueError(f'incomplete block: {variant}, round {round}')
        rows.append((float(latency[1]), float(latency[3]), float(fps[1])))
    means = [r[0] for r in rows]
    text += (f'{variant} | {median(means):.2f} | {median(r[1] for r in rows):.2f} | '
             f'{median(r[2] for r in rows):.1f} | {min(means):.2f}/{max(means):.2f}\n')
(runs.parent / 'round4-summary.txt').write_text(text)
print(text, end='')
PY
} 2>&1 | tee artifacts/bench/round4.txt
