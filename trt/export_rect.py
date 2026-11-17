#!/usr/bin/env python3
"""Generate separate 384x640 eager FP16 references and a static FP16 ONNX model."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    # Parse first: --help must not load the model or generate any artifacts.
    argparse.ArgumentParser(description=__doc__).parse_args()

    import cv2
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch

    from aoti.export import (ARTIFACTS, configure_cuda, load_model,
                             original_detections, preprocess)
    from trt.export_onnx import OPSET, TypedOutput

    names = (ARTIFACTS / "ref/manifest.txt").read_text().splitlines()
    if len(names) != 8 or len(set(names)) != 8 or any(
        not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name)
        for name in names
    ):
        raise RuntimeError("expected 8 distinct sample names in ref/manifest.txt")
    images = {name: cv2.imread(str(ARTIFACTS / f"data/{name}.jpg")) for name in names}
    if any(image is None for image in images.values()):
        raise RuntimeError("failed to read a reference image")
    configure_cuda()
    model = load_model("fp16")
    ref = ARTIFACTS / "ref_384x640"
    ref.mkdir(parents=True, exist_ok=True)
    example = None
    with torch.inference_mode():
        for name, image in images.items():
            data = preprocess(image, (384, 640))
            data.astype("<f4").tofile(ref / f"{name}.input.bin")
            x = torch.from_numpy(data).to(device="cuda", dtype=torch.float16)
            output = model(x)
            assert isinstance(output, torch.Tensor) and output.shape == (1, 300, 6)
            output = output.float().cpu().numpy()
            output.astype("<f4").tofile(ref / f"{name}.eager_fp16.bin")
            dets = original_detections(output, image, (384, 640))
            (ref / f"{name}.eager_fp16_dets.txt").write_text("".join(
                f"{d[0]} " + " ".join(f"{v:.8f}" for v in d[1:]) + "\n" for d in dets))
            print(f"{name} dets={len(dets)}", flush=True)
            if example is None:
                example = x
        (ref / "manifest.txt").write_text("".join(f"{name}\n" for name in names))
        typed = TypedOutput(model).eval()
        eager = typed(example).float().cpu().numpy()
        path = ARTIFACTS / "yolov10s_fp16_384x640.onnx"
        torch.onnx.export(
            typed, (example,), str(path), input_names=["images"], output_names=["output0"],
            opset_version=OPSET, dynamo=True, external_data=False,
        )
    graph = onnx.load(path)
    onnx.checker.check_model(graph, full_check=True)
    for value, shape in ((graph.graph.input[0], (1, 3, 384, 640)),
                         (graph.graph.output[0], (1, 300, 6))):
        assert value.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16
        assert tuple(d.dim_value for d in value.type.tensor_type.shape.dim) == shape
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    output, = session.run(["output0"], {"images": example.cpu().numpy()})
    assert output.shape == eager.shape == (1, 300, 6) and np.isfinite(output).all()
    diff = float(np.max(np.abs(output.astype(np.float32) - eager)))
    print(f"fp16 384x640: opset {OPSET}, shape [1,300,6], onnx.checker PASS; "
          f"ORT CPU PASS; onnx-vs-eager max abs diff {diff:.9g}", flush=True)


if __name__ == "__main__":
    main()
