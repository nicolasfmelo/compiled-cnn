#!/usr/bin/env python3
"""Convert the fixed ONNX and calibrate NNCF on disjoint COCO val ranks 501–800."""
import argparse
import json
import os
import time

from common import ARTIFACTS, PCORES, coco_images, read_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ignore-postprocess', action='store_true',
                        help='keep the terminal detection postprocessing subgraph in FP32')
    parser.add_argument('--preset', choices=('performance', 'mixed'), default='mixed',
                        help='mixed (selected after strict detection checks); performance reproduces the initial attempt')
    args = parser.parse_args()
    os.sched_setaffinity(0, PCORES)
    import nncf
    import openvino as ov
    import torch
    torch.set_num_threads(6)
    _, images = coco_images(500, 800)
    model = ov.convert_model(ARTIFACTS / 'yolov10s_fp32.onnx')
    fp32 = ARTIFACTS / 'yolov10s_fp32_ov/model.xml'
    fp32.parent.mkdir(exist_ok=True)
    ov.save_model(model, fp32, compress_to_fp16=False)
    ignored = []
    if args.ignore_postprocess:
        # Find both TopK stages, then exclude their descendants. These are
        # selection/index/packing operations, not feature extraction convolutions.
        pending = [op for op in model.get_ordered_ops() if op.get_type_name() == 'TopK']
        seen = set()
        while pending:
            op = pending.pop()
            name = op.get_friendly_name()
            if name in seen:
                continue
            seen.add(name)
            for output in op.outputs():
                pending.extend(port.get_node() for port in output.get_target_inputs())
        ignored = sorted(seen)
    dataset = nncf.Dataset(images, read_image)
    start = time.perf_counter()
    quantized = nncf.quantize(model, dataset, subset_size=300,
                             preset=nncf.QuantizationPreset(args.preset),
                             target_device=nncf.TargetDevice.CPU,
                             ignored_scope=nncf.IgnoredScope(names=ignored) if ignored else None)
    seconds = time.perf_counter() - start
    int8 = ARTIFACTS / 'yolov10s_int8_ov/model.xml'
    int8.parent.mkdir(exist_ok=True)
    ov.save_model(quantized, int8, compress_to_fp16=False)
    metadata = dict(calibration_images=300, ranks=[501, 800], image_ids=[im['id'] for im in images],
                    preset=args.preset, model_type=None, ignored_scope=ignored, quantize_seconds=seconds,
                    fp32_bytes=sum(p.stat().st_size for p in fp32.parent.iterdir()),
                    int8_bytes=sum(p.stat().st_size for p in int8.parent.iterdir()),
                    openvino=ov.__version__, nncf=nncf.__version__)
    destination = ARTIFACTS / 'cpu'
    destination.mkdir(exist_ok=True)
    (destination / 'quantization.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
