#!/usr/bin/env python3
"""Forward-only eager baseline; each precision runs in its own process."""
import argparse
import os
import resource
import statistics
import subprocess
import sys
import time

import numpy as np
import pynvml
import torch

from export import ARTIFACTS, load_model


def gpu_memory():
    initialized = False
    try:
        pynvml.nvmlInit()
        initialized = True
        # Search all devices and match this PID, independent of CUDA device order.
        total = 0
        found = False
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
                if proc.pid == os.getpid() and proc.usedGpuMemory is not None and proc.usedGpuMemory < 2**63:
                    total += proc.usedGpuMemory
                    found = True
        return f"{total / 2**20:.1f}" if found else "n/a"
    except (pynvml.NVMLError, OSError):
        return "n/a"
    finally:
        if initialized:
            pynvml.nvmlShutdown()


def benchmark(precision, cudnn_benchmark=False):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    torch.cuda.set_device(0)
    # Match libtorch explicitly; reference generation intentionally keeps TF32 off.
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = cudnn_benchmark
    # CUDA context creation and input transfer are outside model load/forward.
    torch.cuda.init()
    data = np.fromfile(ARTIFACTS / "ref/bus.input.bin", dtype="<f4").reshape(1, 3, 640, 640)
    dtype = torch.float16 if precision == "fp16" else torch.float32
    x = torch.from_numpy(data).to(device="cuda", dtype=dtype)
    torch.cuda.synchronize()
    start = time.perf_counter()
    model = load_model(precision)
    torch.cuda.synchronize()
    load_ms = (time.perf_counter() - start) * 1000
    with torch.inference_mode():
        def iteration():
            torch.cuda.synchronize()
            start = time.perf_counter()
            output = model(x)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            assert output.shape == (1, 300, 6)
            return elapsed

        first_ms = iteration()
        for _ in range(50):
            iteration()
        latencies = [iteration() for _ in range(500)]
    mean = statistics.mean(latencies)
    p50, p95 = np.percentile(latencies, [50, 95])
    suffix = " cudnn-bench" if cudnn_benchmark else ""
    print(f"variant: eager-py {precision}{suffix}")
    print("settings: cudnn_tf32=" + ("on" if torch.backends.cudnn.allow_tf32 else "off")
          + " matmul_tf32=" + ("on" if torch.backends.cuda.matmul.allow_tf32 else "off")
          + " cudnn_benchmark=" + ("on" if torch.backends.cudnn.benchmark else "off"))
    print(f"latency ms: mean {mean:.2f} | p50 {p50:.2f} | p95 {p95:.2f}")
    print(f"throughput: {1000 / mean:.1f} it/s")
    print(f"process GPU MiB (NVML): {gpu_memory()}")
    print(f"peak RSS MiB: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.1f}")
    print(f"load ms: {load_ms:.2f} | first run ms: {first_ms:.2f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", choices=("fp32", "fp16"))
    parser.add_argument("--cudnn-benchmark", action="store_true", help="enable cuDNN algorithm autotuning")
    args = parser.parse_args()
    if args.precision:
        benchmark(args.precision, args.cudnn_benchmark)
    else:
        for precision in ("fp32", "fp16"):
            command = [sys.executable, __file__, "--precision", precision]
            if args.cudnn_benchmark:
                command.append("--cudnn-benchmark")
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
