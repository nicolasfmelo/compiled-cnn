#!/usr/bin/env python3
"""YOLOv10s webcam inference and GPU benchmark."""

import argparse
import os
import statistics
import sys
import time

import cv2
import pynvml
import torch
from ultralytics import YOLO


MIB = 1024 * 1024
WARMUP_FRAMES = 5
WINDOW = "YOLOv10s"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--half", action="store_true", help="use FP16 inference")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--frames", type=int, default=0, help="measured frames (0: unlimited)")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--source", help="video file instead of camera")
    args = parser.parse_args()
    if args.frames < 0 or args.width <= 0 or args.height <= 0 or args.imgsz <= 0:
        parser.error("frames must be nonnegative; width, height and imgsz must be positive")
    if not 0 <= args.conf <= 1:
        parser.error("conf must be between 0 and 1")
    return args


def nvml_memory(handle):
    """Return this process's and the whole GPU's memory, or None if unavailable."""
    process, used, total = None, None, None
    if handle is None:
        return process, used, total
    try:
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used, total = info.used / MIB, info.total / MIB
    except (pynvml.NVMLError, OSError):
        pass
    try:
        for item in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
            if item.pid == os.getpid() and item.usedGpuMemory is not None:
                # NVML uses an unsigned all-ones sentinel for unavailable values.
                if item.usedGpuMemory < (1 << 63):
                    process = item.usedGpuMemory / MIB
                break
    except (pynvml.NVMLError, OSError):
        pass
    return process, used, total


def fmt(value, digits=1):
    return f"{value:.{digits}f}" if value is not None else "n/a"


def percentile(values, fraction):
    values = sorted(values)
    index = (len(values) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def overlay(frame, fps, model_ms, reserved, process):
    lines = [f"FPS: {fps:.1f}  |  model: {model_ms:.1f} ms",
             f"Torch reserved: {reserved:.1f} MiB  |  process: {fmt(process)} MiB"]
    for i, text in enumerate(lines):
        position = (10, 28 + i * 27)
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 255), 1, cv2.LINE_AA)


def main():
    args = parse_args()
    capture = None
    nvml_initialized = False
    handle = None
    camera_info = "n/a"
    device_name = "n/a"
    latencies = []
    loop_seconds = 0.0
    ema_fps = 0.0
    cuda_ready = False
    exit_code = 0
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available; a CUDA-capable GPU and CUDA-enabled PyTorch are required")
        torch.cuda.set_device(0)
        cuda_ready = True
        device_name = torch.cuda.get_device_name(0)
        try:
            pynvml.nvmlInit()
            nvml_initialized = True
            # Map the CUDA device to NVML by PCI bus ID, not NVML's device ordering.
            props = torch.cuda.get_device_properties(0)
            # CUDA_VISIBLE_DEVICES may reorder GPUs; map the CUDA PCI address.
            if all(hasattr(props, name) for name in ("pci_domain_id", "pci_bus_id", "pci_device_id")):
                pci_id = f"{props.pci_domain_id:04x}:{props.pci_bus_id:02x}:{props.pci_device_id:02x}.0"
                handle = pynvml.nvmlDeviceGetHandleByPciBusId(pci_id)
            else:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except (pynvml.NVMLError, OSError):
            handle = None

        capture = (cv2.VideoCapture(args.source) if args.source is not None
                   else cv2.VideoCapture(args.camera, cv2.CAP_V4L2))
        if not capture.isOpened():
            raise RuntimeError(f"cannot open {'video ' + args.source if args.source else 'camera ' + str(args.camera)}")
        if args.source is None:
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
            capture.set(cv2.CAP_PROP_FPS, 30)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
        codec = "".join(chr((fourcc >> (8 * i)) & 0xff) for i in range(4)).strip("\x00") or "unknown"
        camera_info = f"{width}x{height} @ {fps:.1f} ({codec})"
        print(f"{'source' if args.source is not None else 'camera'}: {camera_info}", flush=True)

        model = YOLO("yolov10s.pt")
        for _ in range(WARMUP_FRAMES):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("could not read enough frames for warm-up")
            model.predict(frame, device=0, half=args.half, imgsz=args.imgsz,
                          conf=args.conf, verbose=False)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(0)

        while not args.frames or len(latencies) < args.frames:
            loop_start = time.perf_counter()
            ok, frame = capture.read()
            if not ok:
                if args.source is None or not latencies:
                    print("Warning: could not read a frame", file=sys.stderr)
                break
            torch.cuda.synchronize()
            model_start = time.perf_counter()
            results = model.predict(frame, device=0, half=args.half, imgsz=args.imgsz,
                                    conf=args.conf, verbose=False)
            torch.cuda.synchronize()
            model_ms = (time.perf_counter() - model_start) * 1000
            annotated = results[0].plot()
            process, _, _ = nvml_memory(handle)
            reserved = torch.cuda.memory_reserved(0) / MIB
            overlay(annotated, ema_fps, model_ms, reserved, process)
            quit_requested = False
            if not args.no_show:
                cv2.imshow(WINDOW, annotated)
                key = cv2.waitKey(1) & 0xff
                quit_requested = key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1
            elapsed = time.perf_counter() - loop_start
            loop_seconds += elapsed
            latencies.append(model_ms)
            current_fps = 1 / elapsed
            ema_fps = current_fps if len(latencies) == 1 else 0.9 * ema_fps + 0.1 * current_fps
            if quit_requested:
                break
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except (RuntimeError, cv2.error, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        exit_code = 1
    finally:
        if capture is not None:
            capture.release()
        if not args.no_show:
            cv2.destroyAllWindows()
        process, used, total = nvml_memory(handle)
        if cuda_ready:
            allocated = torch.cuda.memory_allocated(0) / MIB
            allocated_peak = torch.cuda.max_memory_allocated(0) / MIB
            reserved = torch.cuda.memory_reserved(0) / MIB
            reserved_peak = torch.cuda.max_memory_reserved(0) / MIB
        else:
            allocated = allocated_peak = reserved = reserved_peak = None
        print("=== YOLOv10s benchmark ===")
        print(f"device: {device_name} | imgsz={args.imgsz} | half={args.half}")
        print(f"camera: {camera_info}")
        print(f"frames measured: {len(latencies)}")
        print(f"loop FPS: {fmt(len(latencies) / loop_seconds if loop_seconds else None)} (mean)")
        mean = statistics.mean(latencies) if latencies else None
        p50 = percentile(latencies, 0.5) if latencies else None
        p95 = percentile(latencies, 0.95) if latencies else None
        print(f"model latency ms: mean {fmt(mean)} | p50 {fmt(p50)} | p95 {fmt(p95)}")
        print(f"torch allocated MiB: {fmt(allocated)} (peak {fmt(allocated_peak)})")
        print(f"torch reserved MiB: {fmt(reserved)} (peak {fmt(reserved_peak)})")
        print(f"process GPU MiB (NVML): {fmt(process)}")
        print(f"GPU total used MiB: {fmt(used)} / {fmt(total)}")
        if nvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except (pynvml.NVMLError, OSError):
                pass
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
