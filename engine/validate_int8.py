#!/usr/bin/env python3
"""Validate INT8 with the unchanged round-3 matcher and sanity-check strict FP32."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cpu.common import ARTIFACTS, compare, print_validation, read_binary, samples
from engine.engine_client import EngineClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path)
    args = parser.parse_args()
    destination = ARTIFACTS/'engine/int8'
    destination.mkdir(parents=True, exist_ok=True)
    accepted = True
    for precision in ('fp32','int8'):
        results = {}
        with EngineClient(precision, binary=args.binary) as client:
            for name in samples():
                x = read_binary(ARTIFACTS/f'ref/{name}.input.bin', (1,3,640,640))
                ref = read_binary(ARTIFACTS/f'ref/{name}.eager_fp32.bin', (1,300,6))
                results[name] = compare(client.infer(x), ref, int8=precision=='int8')
                print_validation(name, 'engine-'+precision, results[name])
        (destination/f'validate-engine-{precision}.json').write_text(json.dumps(results, indent=2)+'\n')
        count = sum(r['ok'] for r in results.values())
        ok = count == 8 and len(results) == 8
        accepted &= ok
        print(f"engine-{precision}: {count}/{len(results)} {'PASS' if ok else 'FAIL'}", flush=True)
    return int(not accepted)


if __name__ == '__main__':
    sys.exit(main())
