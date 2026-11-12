#!/usr/bin/env python3
"""Export the fused static YOLOv10s graph; never regenerate eager references."""
import sys
import json
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pnnx
import torch
from aoti.export import load_model


def main():
    torch.set_num_threads(6)
    model = load_model('fp32').cpu().float()
    assert model.is_fused() and model.model[-1].end2end
    out = ROOT / 'artifacts/ncnn'
    out.mkdir(parents=True, exist_ok=True)
    head = model.model[-1]
    postprocess = head.postprocess
    # Cut ONLY the selection/packing step. Output is [1,8400,84], decoded
    # xyxy + 80 sigmoid scores from the trained one-to-one branch (no NMS).
    head.postprocess = types.MethodType(lambda self, preds: preds, head)
    bus = torch.from_numpy(np.fromfile(ROOT / 'artifacts/ref/bus.input.bin', dtype='<f4')
                           .reshape(1, 3, 640, 640))
    with torch.inference_mode():
        raw = model(bus)  # Also populate the static anchor cache before tracing.
        expected = postprocess(raw)
        raw.numpy().astype('<f4').tofile(out / 'bus.raw.bin')
        expected.numpy().astype('<f4').tofile(out / 'bus.postprocess.bin')
        pnnx.export(model, str(out / 'yolov10s.pt'),
                    inputs=torch.zeros(1, 3, 640, 640), fp16=False)
    param = (out / 'yolov10s.ncnn.param').read_text()
    unsupported = [line for line in param.splitlines()[2:]
                   if '.' in line.split()[0]]
    if unsupported:
        raise RuntimeError('Unconverted operators: ' + str(unsupported))
    (out / 'conversion.json').write_text(json.dumps(dict(
        pnnx='20261020', weights='fp32 (fp16=False)', input=[1, 3, 640, 640],
        output=[1, 8400, 84], cut='Detect.postprocess input after permute',
        unsupported_full_graph=['torch.topk', 'torch.gather', 'Tensor.to'],
        psa='converted without custom operators'), indent=2) + '\n')
    print('Export PASS: decoded one-to-one [1,8400,84]; C++ top-k fixtures saved')


if __name__ == '__main__':
    main()
