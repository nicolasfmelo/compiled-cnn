#!/usr/bin/env python3
"""Export the round-1 end-to-end model; never regenerate its fixed references."""
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aoti.export import ARTIFACTS, configure_cuda, load_model

OPSET = 18


class TypedOutput(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        # Ultralytics casts the class column to float32 before concatenation,
        # promoting even the half head's boxes/scores. These values and class
        # integers are exactly representable in half; restore the I/O contract.
        return self.model(images).to(images.dtype)


def main():
    configure_cuda()  # Strict reference settings, not the benchmark's TF32 settings.
    data = np.fromfile(ARTIFACTS / "ref/bus.input.bin", dtype="<f4").reshape(1, 3, 640, 640)
    for precision in ("fp32", "fp16"):
        model = TypedOutput(load_model(precision)).eval()
        dtype = torch.float16 if precision == "fp16" else torch.float32
        x = torch.from_numpy(data).to(device="cuda", dtype=dtype)
        path = ARTIFACTS / f"yolov10s_{precision}.onnx"
        with torch.inference_mode():
            eager = model(x).float().cpu().numpy()
            assert eager.shape == (1, 300, 6)
            torch.onnx.export(
                model, (x,), str(path), input_names=["images"], output_names=["output0"],
                opset_version=OPSET, dynamo=True, external_data=False,
            )
        graph = onnx.load(path)
        onnx.checker.check_model(graph, full_check=True)
        expected_type = onnx.TensorProto.FLOAT16 if precision == "fp16" else onnx.TensorProto.FLOAT
        assert graph.graph.input[0].type.tensor_type.elem_type == expected_type
        assert graph.graph.output[0].type.tensor_type.elem_type == expected_type
        # CPU ORT is used only for this one-shot graph check, not for benchmarks.
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        output, = session.run(["output0"], {"images": x.cpu().numpy()})
        assert output.shape == eager.shape and np.isfinite(output).all()
        diff = float(np.max(np.abs(output.astype(np.float32) - eager)))
        print(f"{precision}: opset {OPSET}, shape [1,300,6], onnx.checker PASS; "
              f"onnx-vs-eager max abs diff {diff:.9g}", flush=True)
        del session, model, x
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
