#!/usr/bin/env python3
"""Export the static fused FP32 decoded one-to-one head (not top-k)."""
import argparse
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/engine/yolov10s_raw.onnx')
    args = parser.parse_args()
    # Heavy imports and model loading deliberately happen after argument parsing.
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from aoti.export import load_model

    torch.set_num_threads(6)
    model = load_model('fp32').cpu().float()
    assert model.is_fused() and model.model[-1].end2end
    head = model.model[-1]
    head.postprocess = types.MethodType(lambda self, preds: preds, head)
    data = np.fromfile(ROOT / 'artifacts/ref/bus.input.bin', dtype='<f4').reshape(1, 3, 640, 640)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        assert tuple(model(torch.from_numpy(data)).shape) == (1, 8400, 84)
        torch.onnx.export(model, (torch.from_numpy(data),), str(args.output),
                          input_names=['images'], output_names=['raw'], opset_version=18,
                          dynamo=False, do_constant_folding=True, external_data=False)
    graph = onnx.load(args.output)
    onnx.checker.check_model(graph, full_check=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 6
    session = ort.InferenceSession(str(args.output), options, providers=['CPUExecutionProvider'])
    raw, = session.run(None, {'images': data})
    assert raw.shape == (1, 8400, 84) and np.isfinite(raw).all()
    expected = np.fromfile(ROOT / 'artifacts/ncnn/bus.raw.bin', dtype='<f4').reshape(raw.shape)
    print(f'Export PASS: opset 18, FP32 [1,8400,84], onnx.checker full_check PASS; '
          f'ORT vs ncnn fixture max_abs_diff={np.max(np.abs(raw - expected)):.9g}')


if __name__ == '__main__':
    main()
