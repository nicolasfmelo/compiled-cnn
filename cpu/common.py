"""Shared CPU runtime setup and native/common.h-compatible validation."""
import json
import os
from pathlib import Path
import re
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ARTIFACTS = ROOT / "artifacts"
COCO_ROOT = Path.home() / ".cache/compiled-cnn/coco"
VARIANTS = ("eager-cpu-fp32", "ort-cpu-fp32", "ov-cpu-fp32", "ov-cpu-int8", "ov-cpu-fp32-default")
PCORES = {0, 2, 4, 6, 8, 10}
OV_PROPERTIES = dict(INFERENCE_PRECISION_HINT="f32", PERFORMANCE_HINT="LATENCY",
                     INFERENCE_NUM_THREADS=6, ENABLE_HYPER_THREADING=False, ENABLE_CPU_PINNING=True)


def read_binary(path, shape):
    data = np.fromfile(path, dtype="<f4")
    if data.size != np.prod(shape) or not np.isfinite(data).all():
        raise ValueError(f"missing, wrong-sized or non-finite reference: {path}")
    return data.reshape(shape)


def samples():
    names = (ARTIFACTS / "ref/manifest.txt").read_text().splitlines()
    names = [n for n in names if n]
    if not names or any(not re.fullmatch(r"[a-zA-Z0-9_-]+", n) for n in names):
        raise ValueError("empty or invalid manifest")
    return names


def iou(a, b):
    def area(box):
        return np.maximum(np.float32(0), box[2] - box[0]) * np.maximum(np.float32(0), box[3] - box[1])
    intersection = area(np.array([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])], dtype=np.float32))
    total = area(a) + area(b) - intersection
    return float(intersection / total) if total > 0 else 0.0


def compare(output, reference, int8=False):
    """Strict mode mirrors native match/validate, including count-mismatch early return.

    INT8 retains greedy geometry-only pairing, but only at IoU >= 0.80.
    Remaining detections on both sides are allowed only below score 0.30.
    """
    output = np.asarray(output, dtype=np.float32)
    if output.shape != (1, 300, 6) or not np.isfinite(output).all():
        raise ValueError("invalid inference output")
    actual = output[0][output[0, :, 4] >= 0.25]
    expected = reference[0][reference[0, :, 4] >= 0.25]
    reasons, unmatched = [], []
    boxes_max = scores_max = 0.0
    min_iou = 1.0
    if not int8 and len(actual) != len(expected):
        reasons.append("detection count mismatch")
        min_iou = 0.0
    else:
        aa, bb = list(range(len(actual))), list(range(len(expected)))
        while aa and bb:
            # Strict > in the C++ nested loops means the first tie wins.
            best, ai, bi = -1.0, 0, 0
            for i in aa:
                for j in bb:
                    value = iou(actual[i], expected[j])
                    if value > best:
                        best, ai, bi = value, i, j
            if int8 and best < float(np.float32(0.80)):
                break
            aa.remove(ai)
            bb.remove(bi)
            a, b = actual[ai], expected[bi]
            min_iou = min(min_iou, best)
            boxes_max = max(boxes_max, float(np.max(np.abs(a[:4] - b[:4]))))
            scores_max = max(scores_max, float(abs(a[4] - b[4])))
            if int(a[5]) != int(b[5]):
                reasons.append("class mismatch; ")
            if best < float(np.float32(0.80 if int8 else 0.95)):
                reasons.append("IoU below threshold; ")
        if int8:
            for side, rows, indices in (("actual", actual, aa), ("expected", expected, bb)):
                for i in indices:
                    row = rows[i]
                    unmatched.append(dict(side=side, score=float(row[4]), cls=int(row[5]), box=row[:4].tolist()))
                    if row[4] >= 0.30:
                        reasons.append("unmatched score >= 0.30; ")
    if not int8 and boxes_max > 0.5:
        reasons.append("box tolerance exceeded; ")
    if scores_max > float(np.float32(0.05 if int8 else 0.001)):
        reasons.append("score tolerance exceeded; ")
    return dict(ok=not reasons, boxes_max=boxes_max, scores_max=scores_max,
                dets=f"{len(actual)}/{len(expected)}", min_iou=min_iou,
                overall_max=float(np.max(np.abs(output - reference))),
                reason="".join(reasons), unmatched=unmatched)


def print_validation(name, variant, result):
    print(f"{'PASS' if result['ok'] else 'FAIL'} {name} {variant} "
          f"boxes_max={result['boxes_max']:.6f} scores_max={result['scores_max']:.6f} "
          f"dets={result['dets']} min_iou={result['min_iou']:.6f} overall_max={result['overall_max']:.6f}"
          + (f" reason={result['reason']}" if not result['ok'] else ""), flush=True)
    for row in result['unmatched']:
        print("unmatched: " + json.dumps(dict(sample=name, variant=variant, **row)), flush=True)


def load_runtime(variant):
    """Return synchronous infer, settings, model size, load time; imports excluded."""
    if variant not in VARIANTS:
        raise ValueError(variant)
    if variant != "ov-cpu-fp32-default":
        os.sched_setaffinity(0, PCORES)
    if variant.startswith("eager"):
        import torch
        from aoti.export import load_model
        torch.set_num_threads(6)
        torch.set_num_interop_threads(1)
        start = time.perf_counter()
        model = load_model("fp32").cpu().float()
        cached_array = cached_tensor = None
        def infer(x):
            nonlocal cached_array, cached_tensor
            if x is not cached_array:
                cached_array, cached_tensor = x, torch.from_numpy(x)
            with torch.inference_mode():
                return model(cached_tensor).numpy()
        settings = "threads=6 interop_threads=1 inference_mode=on"
        size = (ROOT / "yolov10s.pt").stat().st_size
    elif variant.startswith("ort"):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 6
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        path = ARTIFACTS / "yolov10s_fp32.onnx"
        start = time.perf_counter()
        model = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        infer = lambda x: model.run(["output0"], {"images": x})[0]
        settings = "provider=CPUExecutionProvider intra_threads=6 inter_threads=1 ORT_SEQUENTIAL ORT_ENABLE_ALL"
        size = path.stat().st_size
    else:
        import openvino as ov
        path = ARTIFACTS / ("yolov10s_int8_ov" if variant == "ov-cpu-int8" else "yolov10s_fp32_ov") / "model.xml"
        core = ov.Core()
        start = time.perf_counter()
        model = core.compile_model(core.read_model(path), "CPU", {} if variant.endswith("default") else OV_PROPERTIES)
        request = model.create_infer_request()
        def infer(x):
            return request.infer({0: x}, share_inputs=True)[model.output(0)]
        settings = " ".join(f"{key}={model.get_property(key)}" for key in
                            (*OV_PROPERTIES, "NUM_STREAMS"))
        size = path.stat().st_size + path.with_suffix(".bin").stat().st_size
    load_ms = (time.perf_counter() - start) * 1000
    settings += " affinity=" + ",".join(map(str, sorted(os.sched_getaffinity(0))))
    return infer, settings, size, load_ms


def coco_images(start=0, stop=500):
    annotation = COCO_ROOT / "annotations/instances_val2017.json"
    info = json.loads(annotation.read_text())
    images = sorted(info['images'], key=lambda x: x['id'])[start:stop]
    if len(images) != stop - start:
        raise ValueError("incomplete COCO selection")
    return info, images


def read_image(info):
    import cv2
    from aoti.export import preprocess
    image = cv2.imread(str(COCO_ROOT / "val2017" / info['file_name']))
    if image is None:
        raise FileNotFoundError(info['file_name'])
    return preprocess(image)
