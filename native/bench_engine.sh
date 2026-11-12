#!/usr/bin/env bash
# PBI-098: four variants, three rotating rounds, up to five contention retries.
set -euo pipefail
cd "$(dirname "$0")/.."
if (( $# != 0 )); then echo 'usage: [BENCH_SMOKE=1] [BENCH_CONTENDED=1] bash native/bench_engine.sh' >&2; exit 2; fi
variants=(engine-fp32-6t engine-int8-6t ov-cpu-fp32 ov-cpu-int8)
rounds=3
cooldown=60
runs=artifacts/engine/bench/engine-runs
summary=artifacts/engine/bench/engine-summary.txt
native_options=()
if [[ ${BENCH_SMOKE:-0} == 1 ]]; then
  rounds=1
  cooldown=5
  runs+=/smoke
  summary="$runs/engine-summary.txt"
  native_options=(--warmup 1 --iters 5)
fi
mkdir -p "$runs"
# Only this runner's files; never remove or overwrite another round's artifacts.
rm -f "$summary" "$runs"/round-{1,2,3}-{engine-fp32-6t,engine-int8-6t,ov-cpu-fp32,ov-cpu-int8}.{txt,json}
{
  echo "session: engine | rounds=$rounds | cooldown=$cooldown s | smoke=${BENCH_SMOKE:-0}"
  for (( round=1; round<=rounds; ++round )); do
    shift=$(( round - 1 ))
    for index in 0 1 2 3; do
      variant=${variants[$(( (index + shift) % 4 ))]}
      output="$runs/round-$round-$variant.txt"
      for attempt in 0 1 2 3 4 5; do
        echo "round: $round | variant: $variant | attempt: $((attempt + 1))/6 | cooldown: $cooldown s"
        sleep "$cooldown"
        status=0
        uv run python cpu/bench.py --check-host || status=$?
        if (( status == 0 )); then
          if [[ "$variant" == ov-cpu-* ]]; then
            # Keep cpu/bench.py's own monitor, precision, warmup and iterations
            # exactly as in native/bench_cpu.sh, also during smoke runs.
            taskset -c 0,2,4,6,8,10 uv run python cpu/bench.py --variant "$variant" \
              --output-json "$runs/round-$round-$variant.json" | tee "$output.pending" || status=$?
            if (( status == 0 )); then mv "$output.pending" "$output"; else rm -f "$output.pending"; fi
          else
            threads=6
            cpus=0,2,4,6,8,10
            precision=${variant#engine-}
            precision=${precision%-6t}
            command=(taskset -c "$cpus" engine/build/yolo_engine bench --precision "$precision" --threads "$threads" "${native_options[@]}")
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
    if process.poll() is None:
        process.terminate()
        process.wait()
    monitor.stop.set()
    monitor.thread.join()
if monitor.load:
    import numpy as np
    text += (f'external CPU: mean {np.mean(monitor.load):.1f}% | max {np.max(monitor.load):.1f}% '
             f'(100% = one core; contended mode={int(monitor.tolerate)})\n')
print(text, end='', flush=True)
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
        echo "retry: contention discard; next attempt follows a fresh $cooldown s cooldown"
      done
    done
  done
  uv run python - "$runs" "$summary" "$rounds" "$cooldown" "${variants[@]}" <<'PY'
from pathlib import Path
import re
import os
from statistics import median
import sys

runs, destination = map(Path, sys.argv[1:3])
rounds, cooldown = map(int, sys.argv[3:5])
text = (f'summary: medians of {rounds} accepted runs; {cooldown} s cooldown before each attempt\n'
        + ('SMOKE ONLY: not a thermal result; engine uses 1 warmup + 5 iterations\n' if rounds == 1 else '')
        + ('CONTENDED: browser/chat left open (BENCH_CONTENDED=1); runs recorded, not discarded\n'
           if os.environ.get('BENCH_CONTENDED') == '1' else '')
        + 'ideal estimate = median p5 (fastest 5% of iterations); margin = mean/ideal - 1\n'
        + 'variant | median mean ms | median p95 ms | median FPS | min/max mean ms | ideal (p5) ms | margin vs ideal | external CPU mean\n')
for variant in sys.argv[5:]:
    rows = []
    for round in range(1, rounds+1):
        block = (runs / f'round-{round}-{variant}.txt').read_text()
        expected = (f'variant: {variant}\n' if variant.startswith('ov-cpu-') else
                    f'variant: engine {variant.split("-")[1]} impl=opt threads=6 cpus=0,2,4,6,8,10\n')
        if expected not in block:
            raise ValueError(f'wrong variant: {variant}, round {round}')
        latency = re.search(r'latency ms: mean ([\d.]+) \| p50 ([\d.]+) \| p95 ([\d.]+)', block)
        fps = re.search(r'throughput: ([\d.]+) it/s', block)
        if not latency or not fps:
            raise ValueError(f'incomplete block: {variant}, round {round}')
        best = re.search(r'best-case ms: p5 ([\d.]+)', block)
        load = re.search(r'external CPU: mean ([\d.]+)%', block)
        if not best:
            raise ValueError(f'missing best-case line: {variant}, round {round}')
        rows.append((float(latency[1]), float(latency[3]), float(fps[1]), float(best[1]),
                     float(load[1]) if load else float('nan')))
    means = [r[0] for r in rows]
    ideal = median(r[3] for r in rows)
    text += (f'{variant} | {median(means):.2f} | {median(r[1] for r in rows):.2f} | '
             f'{median(r[2] for r in rows):.1f} | {min(means):.2f}/{max(means):.2f} | '
             f'{ideal:.2f} | +{(median(means) / ideal - 1) * 100:.1f}% | {median(r[4] for r in rows):.1f}%\n')
destination.write_text(text)
print(text, end='')
PY
} 2>&1 | tee "$runs/session.txt"
