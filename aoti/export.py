#!/usr/bin/env python3
"""Export static, fused YOLOv10s packages and reproducible eager references."""
import argparse
import json
import os
import shutil
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ASSETS

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"


def configure_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def configure_build_cuda():
    """Present the pinned wheels as a CUDA_HOME without invoking system nvcc."""
    site = Path(torch.__file__).resolve().parent.parent
    runtime = site / "nvidia/cuda_runtime"
    triton_include = site / "triton/backends/nvidia/include"
    cuda_home = ARTIFACTS / "cuda-12.8"
    include = cuda_home / "include"
    include.mkdir(parents=True, exist_ok=True)
    # cuda_fp16.h includes nv/target: supplied by the matching CCCL header wheel.
    for headers in (runtime / "include", site / "nvidia/cuda_cccl/include"):
        for source in headers.iterdir():
            target = include / source.name
            if not target.exists():
                target.symlink_to(source, target_is_directory=source.is_dir())
    # nvidia-cuda-runtime omits crt/, but pinned Triton bundles CUDA 12.8 crt/.
    if not (include / "crt").exists():
        (include / "crt").symlink_to(triton_include / "crt", target_is_directory=True)
    if not (cuda_home / "lib64").exists():
        (cuda_home / "lib64").symlink_to(runtime / "lib", target_is_directory=True)
    os.environ["CUDA_HOME"] = str(cuda_home)
    # Keep potentially large native build caches out of the /tmp tmpfs.
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(ARTIFACTS / "inductor_cache")
    os.environ["TRITON_CACHE_DIR"] = str(ARTIFACTS / "triton_cache")
    # cpp_extension caches CUDA_HOME at import; an Ultralytics import may precede us.
    from torch.utils import cpp_extension
    cpp_extension.CUDA_HOME = str(cuda_home)
    print(f"CUDA build: 12.8 runtime + CCCL wheel headers, Triton crt/; CUDA_HOME={cuda_home}", flush=True)


def load_model(precision):
    model = YOLO(str(ROOT / "yolov10s.pt")).model.eval()
    # New Ultralytics defaults _end2end=False for this older checkpoint.
    # Select the trained one-to-one branch BEFORE fuse removes an unused head.
    head = model.model[-1]
    head.end2end = True
    model.fuse(verbose=False)
    head.export = True
    assert head.end2end
    model = model.to("cuda")
    return model.half() if precision == "fp16" else model.float()


def preprocess(image):
    boxed = LetterBox(new_shape=(640, 640), auto=False)(image=image)
    rgb = boxed[:, :, ::-1].transpose(2, 0, 1)
    return np.ascontiguousarray(rgb[None], dtype="<f4") / np.float32(255)


def original_detections(output, image):
    height, width = image.shape[:2]
    gain = min(640 / height, 640 / width)
    left = round((640 - round(width * gain)) / 2 - 0.1)
    top = round((640 - round(height * gain)) / 2 - 0.1)
    result = []
    for row in output[0]:
        x1, y1, x2, y2, score, cls = map(float, row)
        if score >= 0.25:
            box = [np.clip((x1 - left) / gain, 0, width),
                   np.clip((y1 - top) / gain, 0, height),
                   np.clip((x2 - left) / gain, 0, width),
                   np.clip((y2 - top) / gain, 0, height)]
            result.append([int(cls), score, *map(float, box)])
    return result


def collect_images(capture_webcam):
    data = ARTIFACTS / "data"
    data.mkdir(parents=True, exist_ok=True)
    for name in ("bus", "zidane"):
        shutil.copyfile(ASSETS / f"{name}.jpg", data / f"{name}.jpg")
    status = "not requested"
    if capture_webcam:
        # Reuse an existing complete capture rather than recapture private frames.
        if all((data / f"webcam_{i:02d}.jpg").exists() for i in range(6)):
            status = "reused 6 previously captured frames"
        else:
            cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
            captured = 0
            try:
                if not cap.isOpened():
                    raise RuntimeError("cannot open /dev/video0")
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                for i in range(36):
                    ok, frame = cap.read()
                    if not ok:
                        raise RuntimeError(f"frame read failed at {i}")
                    if i >= 30:
                        if not cv2.imwrite(str(data / f"webcam_{i - 30:02d}.jpg"), frame):
                            raise RuntimeError("frame write failed")
                        captured += 1
                status = (f"captured {captured} frames after 30 exposure frames; "
                          f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
                          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} MJPG requested")
            except (RuntimeError, cv2.error) as exc:
                status = f"capture failed ({exc}); continuing with {captured} frames"
            finally:
                cap.release()
    print(f"webcam: {status}", flush=True)
    paths = [data / "bus.jpg", data / "zidane.jpg", *sorted(data.glob("webcam_*.jpg"))]
    return paths, status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-webcam", action="store_true")
    args = parser.parse_args()
    configure_build_cuda()
    configure_cuda()
    paths, webcam = collect_images(args.capture_webcam)
    ref = ARTIFACTS / "ref"
    ref.mkdir(parents=True, exist_ok=True)
    images = {p.stem: cv2.imread(str(p)) for p in paths}
    if any(image is None for image in images.values()):
        raise RuntimeError("failed to read a reference image")
    inputs = {name: preprocess(image) for name, image in images.items()}
    for name, array in inputs.items():
        array.astype("<f4").tofile(ref / f"{name}.input.bin")
    (ref / "manifest.txt").write_text("".join(f"{name}\n" for name in inputs))
    metadata = {"torch": torch.__version__, "cuda": torch.version.cuda,
                "webcam": webcam, "samples": list(inputs), "packages": {}}
    for precision in ("fp32", "fp16"):
        model = load_model(precision)
        dtype = torch.float16 if precision == "fp16" else torch.float32
        example = torch.zeros((1, 3, 640, 640), device="cuda", dtype=dtype)
        with torch.inference_mode():
            for name, array in inputs.items():
                output = model(torch.from_numpy(array).to(device="cuda", dtype=dtype))
                assert isinstance(output, torch.Tensor) and output.shape == (1, 300, 6), (
                    type(output), getattr(output, "shape", None))
                output = output.float().cpu().numpy()
                output.astype("<f4").tofile(ref / f"{name}.eager_{precision}.bin")
                dets = original_detections(output, images[name])
                suffix = "eager_dets" if precision == "fp32" else "eager_fp16_dets"
                (ref / f"{name}.{suffix}.txt").write_text("".join(
                    f"{d[0]} " + " ".join(f"{v:.8f}" for v in d[1:]) + "\n" for d in dets))
            # Include torch.export as well as native compilation in compile time.
            torch.cuda.synchronize()
            start = time.perf_counter()
            ep = torch.export.export(model, (example,))
            package = ARTIFACTS / f"yolov10s_{precision}.pt2"
            torch._inductor.aoti_compile_and_package(ep, package_path=str(package))
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
        with zipfile.ZipFile(package) as archive:
            entries = [{"name": i.filename, "bytes": i.file_size} for i in archive.infolist()]
        metadata["packages"][precision] = {
            "compile_seconds": seconds, "bytes": package.stat().st_size, "entries": entries}
        (ARTIFACTS / "export_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"{precision}: shape [1,300,6], compile {seconds:.3f} s, "
              f"package {package.stat().st_size / 2**20:.3f} MiB", flush=True)
        del model, ep, example
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
