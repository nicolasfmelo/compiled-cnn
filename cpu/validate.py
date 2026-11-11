#!/usr/bin/env python3
"""Validate all CPU variants against the immutable GPU eager FP32 references."""
import argparse
import json
import subprocess
import sys
from common import ARTIFACTS, VARIANTS, compare, load_runtime, print_validation, read_binary, samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=VARIANTS)
    args = parser.parse_args()
    if args.variant is None:
        codes = [subprocess.run([sys.executable, __file__, '--variant', v]).returncode for v in VARIANTS]
        return int(any(codes))
    infer, settings, _, _ = load_runtime(args.variant)
    print('settings: ' + settings, flush=True)
    results = {}
    for name in samples():
        x = read_binary(ARTIFACTS / f'ref/{name}.input.bin', (1, 3, 640, 640))
        ref = read_binary(ARTIFACTS / f'ref/{name}.eager_fp32.bin', (1, 300, 6))
        result = compare(infer(x), ref, int8=args.variant == 'ov-cpu-int8')
        results[name] = result
        print_validation(name, args.variant, result)
    destination = ARTIFACTS / 'cpu'
    destination.mkdir(exist_ok=True)
    (destination / f'validate-{args.variant}.json').write_text(json.dumps(results, indent=2) + '\n')
    return int(not all(r['ok'] for r in results.values()))


if __name__ == '__main__':
    sys.exit(main())
