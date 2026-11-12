#!/usr/bin/env python3
"""Calibrate dense Conv inputs on COCO val2017 image-id ranks 501–800.

Two deterministic passes: exact extrema, then 65536-bin uniform histograms.
Percentile endpoints are rounded outwards to histogram edges (no random sampling).
Only artifacts/engine/ is written; ONNX weights and references are never changed.
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'artifacts/engine/yolov10s_raw.onnx')
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/engine/int8/calibration.json')
    args = parser.parse_args()
    # --help must terminate before importing runtimes or touching the dataset.
    import json
    import os
    import numpy as np
    import onnx
    import onnxruntime as ort
    sys.path.insert(0, str(ROOT))
    from cpu.common import PCORES, coco_images, read_image
    os.sched_setaffinity(0, PCORES)
    model = onnx.shape_inference.infer_shapes(onnx.load(args.input))
    convs = {n.name: n.input[0] for n in model.graph.node if n.op_type == 'Conv'
             and next((a.i for a in n.attribute if a.name == 'group'), 1) == 1}
    names = sorted(set(convs.values()))
    descriptors = {v.name: v for v in [*model.graph.input, *model.graph.value_info, *model.graph.output]}
    del model.graph.output[:]
    model.graph.output.extend(descriptors[name] for name in names)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 6
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(model.SerializeToString(), options, providers=['CPUExecutionProvider'])
    _, images = coco_images(500, 800)
    extrema = {name: [float('inf'), -float('inf')] for name in names}
    bins = 65536
    histograms = {name: np.zeros(bins, np.int64) for name in names}
    for phase in range(2):
        for index, image in enumerate(images, 1):
            outputs = session.run(names, {model.graph.input[0].name: read_image(image)})
            for name, tensor in zip(names, outputs):
                if not np.isfinite(tensor).all():
                    raise ValueError(f'non-finite calibration tensor: {name}')
                if phase == 0:
                    extrema[name][0] = min(extrema[name][0], float(tensor.min()))
                    extrema[name][1] = max(extrema[name][1], float(tensor.max()))
                else:
                    lo, hi = extrema[name]
                    if hi == lo:
                        histograms[name][0] += tensor.size
                    else:
                        histograms[name] += np.histogram(tensor, bins=bins, range=(lo, hi))[0]
            if index % 50 == 0:
                print(f'calibration pass {phase+1}/2: {index}/300', flush=True)
    tensors = {}
    for name in names:
        lo, hi = extrema[name]
        cdf = histograms[name].cumsum()
        ranges = {}
        for method, fraction in [('p9999', .9999), ('p99999', .99999)]:
            tail = (1-fraction)/2
            left = int(np.searchsorted(cdf, tail*int(cdf[-1]), side='right'))
            right = int(np.searchsorted(cdf, (1-tail)*int(cdf[-1]), side='left'))+1
            ranges[method] = [lo+(hi-lo)*left/bins, lo+(hi-lo)*min(right,bins)/bins]
        tensors[name] = dict(min=lo, max=hi, count=int(cdf[-1]), **ranges)
    result = dict(tensors=tensors, conv_inputs=convs,
                  protocol=dict(ranks=[501,800], images=300, image_ids=[im['id'] for im in images],
                                histogram_bins=bins, percentile='equal tails, outward bin edges'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n')
    print(f'Wrote {args.output}: {len(convs)} dense convolutions, {len(tensors)} input tensors')


if __name__ == '__main__':
    main()
