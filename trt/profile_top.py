#!/usr/bin/env python3
"""Show the ten most expensive layers from a trtexec exported profile."""
import argparse
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    args = parser.parse_args()
    entries = json.loads(args.profile.read_text())
    layers = []
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("name") or "averageMs" not in entry:
            continue
        average = float(entry["averageMs"])
        if not math.isfinite(average) or average < 0:
            parser.exit(1, "invalid per-layer average time\n")
        layers.append((average, entry["name"]))
    if not layers:
        parser.exit(1, "profile has no per-layer entries\n")
    total = sum(average for average, _ in layers)
    print(f"profile: {args.profile}")
    print("rank | average ms | percent | layer")
    for rank, (average, name) in enumerate(sorted(layers, key=lambda row: row[0], reverse=True)[:10], 1):
        print(f"{rank:2d} | {average:.6f} | {100 * average / total if total else 0:.2f}% | {name}")
    print(f"summed average layer time ms: {total:.6f}")


if __name__ == "__main__":
    main()
