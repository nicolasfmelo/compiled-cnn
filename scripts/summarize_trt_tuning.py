#!/usr/bin/env python3
"""Summarize per-process TensorRT tuning runs (medians across rounds)."""
import argparse
import math
import re
from pathlib import Path
from statistics import median

VARIANTS = (
    "eager-640", "eager-384", "A-o3-b1", "A-o3-b2", "B-o5-b1", "B-o5-b2",
    "C-o5-b1-graph", "D-384x640-o5-graph",
)
LATENCY = r"latency ms: mean ([\d.]+) \| p50 ([\d.]+) \| p95 ([\d.]+)$"


def summarize(directory):
    rows = {variant: [] for variant in VARIANTS}
    for path in sorted(directory.glob("round-*-*.txt")):
        name = re.fullmatch(r"round-([1-9]\d*)-(.+)\.txt", path.name)
        if not name or name[2] not in rows:
            raise ValueError(f"unexpected benchmark file: {path}")
        variant = name[2]
        text = path.read_text()
        expected = {"eager-640": "eager-py fp16", "eager-384": "eager-py fp16 384x640"}.get(variant, variant)
        if f"variant: {expected}\n" not in text:
            raise ValueError(f"wrong or missing variant in {path}")
        host = re.findall("^" + LATENCY, text, re.MULTILINE)
        gpu = re.findall("^gpu " + LATENCY, text, re.MULTILINE)
        if len(host) != 1 or (not variant.startswith("eager-") and len(gpu) != 1):
            raise ValueError(f"missing/duplicate latency in {path}")
        values = tuple(map(float, host[0] + (gpu[0] if gpu else ())))
        if any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError(f"invalid latency in {path}")
        rows[variant].append(values)
    missing = [variant for variant, runs in rows.items() if not runs]
    if missing:
        raise ValueError("missing benchmark variants: " + ", ".join(missing))
    medians = {variant: tuple(median(column) for column in zip(*runs)) for variant, runs in rows.items()}
    lines = [
        "summary: medians across rounds; latency in ms; speedups use host mean",
        "variant | rounds | host mean | host p50 | host p95 | gpu mean | gpu p50 | gpu p95 | vs A-o3-b1 | vs same-size eager",
    ]
    for variant in VARIANTS:
        values = medians[variant]
        eager = "eager-384" if variant in ("eager-384", "D-384x640-o5-graph") else "eager-640"
        columns = [f"{value:.2f}" for value in values]
        if variant.startswith("eager-"):
            columns = columns[:3] + ["n/a"] * 3
        lines.append(f"{variant} | {len(rows[variant])} | " + " | ".join(columns)
                     + f" | {medians['A-o3-b1'][0] / values[0]:.2f}x"
                     + f" | {medians[eager][0] / values[0]:.2f}x")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bench_dir", type=Path)
    args = parser.parse_args()
    try:
        text = summarize(args.bench_dir)
    except (ValueError, OSError) as error:
        parser.exit(1, f"{error}\n")
    (args.bench_dir / "summary.txt").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
