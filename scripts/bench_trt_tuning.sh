#!/usr/bin/env bash
# Eight isolated processes per round, rotating the first variant each round.
set -euo pipefail
cd "$(dirname "$0")/.."
if (( $# != 0 )); then
    echo 'usage: [BENCH_ROUNDS=N] [BENCH_COOLDOWN=S] bash scripts/bench_trt_tuning.sh' >&2
    exit 2
fi
BIN=${CNN_BIN:-build/bin}
rounds=${BENCH_ROUNDS:-3}
cooldown=${BENCH_COOLDOWN:-30}
[[ "$rounds" =~ ^[1-9][0-9]*$ ]] || { echo 'BENCH_ROUNDS must be a positive integer' >&2; exit 2; }
[[ "$cooldown" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo 'BENCH_COOLDOWN must be nonnegative seconds' >&2; exit 2; }
variants=(eager-640 eager-384 A-o3-b1 A-o3-b2 B-o5-b1 B-o5-b2 C-o5-b1-graph D-384x640-o5-graph)
runs=artifacts/trt_tuning/bench
if (( rounds != 3 )) || ! awk -v s="$cooldown" 'BEGIN {exit !(s == 30)}'; then runs+=/smoke; fi
mkdir -p "$runs"
# Only this script's own outputs; never touch other benchmark directories.
for variant in "${variants[@]}"; do rm -f "$runs"/round-*-"$variant".txt; done
rm -f "$runs/summary.txt"
for (( round=1; round<=rounds; ++round )); do
    for (( index=0; index<${#variants[@]}; ++index )); do
        variant=${variants[$(( (index + round - 1) % ${#variants[@]} ))]}
        echo "round: $round | variant: $variant | cooldown: $cooldown s"
        sleep "$cooldown"
        case "$variant" in
            eager-640) command=(uv run python aoti/bench_eager.py --precision fp16) ;;
            eager-384) command=(uv run python aoti/bench_eager.py --precision fp16 --height 384) ;;
            A-o3-b1|A-o3-b2|B-o5-b1|B-o5-b2)
                level=3
                if [[ "$variant" == B-* ]]; then level=5; fi
                build=${variant##*-}
                command=(env -u LD_LIBRARY_PATH "$BIN/yolo_trt" bench
                    "artifacts/trt_tuning/fp16_640_o${level}_${build}.plan"
                    --ref-dir artifacts/ref --label "$variant") ;;
            C-o5-b1-graph)
                command=(env -u LD_LIBRARY_PATH "$BIN/yolo_trt" bench
                    artifacts/trt_tuning/fp16_640_o5_b1.plan
                    --ref-dir artifacts/ref --graph --label "$variant") ;;
            D-384x640-o5-graph)
                command=(env -u LD_LIBRARY_PATH "$BIN/yolo_trt" bench
                    artifacts/trt_tuning/fp16_384x640_o5.plan
                    --ref-dir artifacts/ref_384x640 --graph --label "$variant") ;;
        esac
        "${command[@]}" 2>&1 | tee "$runs/round-$round-$variant.txt"
    done
done
uv run python scripts/summarize_trt_tuning.py "$runs"
