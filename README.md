# Compiled CNN: YOLOv10s on native runtimes

How much faster does a detection network get when it is compiled to a native artifact and run from C++ instead of PyTorch in Python, what does the artifact still depend on, and do the outputs stay the same?

This repository answers that for **YOLOv10s** (Ultralytics) on one laptop, across four rounds plus a hand-written engine:

| Round | Backend(s) | Device |
|---|---|---|
| 0 | Ultralytics `predict` on the webcam (starting point) | NVIDIA GPU |
| 1 | **AOTInductor** (`torch.export` → `.pt2`, loaded from C++ via libtorch) | NVIDIA GPU |
| 2 | **TensorRT 11.4** (ONNX → `.plan` engine, C++ without libtorch) | NVIDIA GPU |
| 3 | PyTorch eager, **ONNX Runtime**, **OpenVINO** FP32 and **INT8** (NNCF, AVX-VNNI) | CPU |
| 4 | **ncnn** over **Vulkan** (vendor-neutral GPU) and ncnn on CPU | NVIDIA GPU, Intel iGPU, CPU |
| Engine | **Own C++ engine** generated from the ONNX graph: AVX2/FMA kernels, no inference library, no BLAS | CPU |

Every backend is validated against the same fixed PyTorch reference with fixed tolerances, and benchmarked with the same protocol. Full write-ups (in Portuguese) are in [`docs/report/`](docs/report/) and in the per-round technical reports in [`docs/`](docs/).

## Results at a glance

Latency of one forward pass, batch 1, static 640×640 input, input already in device memory. FPS = 1000 / mean latency (model only: no camera, no drawing).

**NVIDIA RTX 3050 6GB Laptop**

| Backend | FP32 | FP16 | Speedup vs eager (FP16) | Validation |
|---|---|---|---|---|
| PyTorch eager (Python) | 10.02 ms | 5.59 ms | 1.00× | reference |
| AOTInductor (C++) | 8.46 ms | 3.92 ms | 1.43× | 16/16 |
| **TensorRT (C++)** | **7.39 ms** | **2.94 ms (340 FPS)** | **1.90×** | 23/24 ¹ |
| TensorRT FP32 with `--noTF32` | 7.86 ms | — | — | 8/8 |
| ncnn Vulkan (C++, cooperative matrix) | 18.55 ms | 9.69 ms | 0.58× | 16/16 |

**Intel Core i5-13450HX (6 threads pinned to the 6 P-cores)**

| Backend | Precision | Latency | FPS | % of FP32 peak | Validation |
|---|---|---|---|---|---|
| PyTorch eager | FP32 | 55.56 ms | 18.0 | 47% | 8/8 |
| ncnn (C++) | FP32 | 56.89 ms | 17.6 | — | 8/8 |
| ONNX Runtime | FP32 | 43.63 ms | 22.9 | 63% | 8/8 |
| OpenVINO | FP32 | 40.20 ms | 24.9 | 70% | 8/8 |
| **Own engine (C++, generated)** ³ | FP32 | **46.67 ms** (ideal ≈ 45.4) | 21.4 | 58% ⁴ | 8/8 |
| **OpenVINO** | **INT8** | **14.91 ms** | **67.1** | — | 8/8, mAP −0.31 pt ² |

**Intel UHD Graphics (iGPU), ncnn Vulkan:** ~230 ms FP16, ~433 ms FP32 (one uncontrolled round; the benchmark was stopped on purpose because the device is far too slow for this model).

¹ The default TensorRT FP32 engine allows TF32 and misses the score tolerance on 1 of 8 images (0.0022 vs 0.001, same box and class). Built with `--noTF32` it matches the strict-FP32 reference almost bit for bit.
² mAP50-95 on 500 COCO val2017 images: 49.33 (FP32) → 49.02 (INT8). Calibration used 300 different val images.
³ Measured with a browser and a chat app running (they could not be closed). In the same session OpenVINO FP32 measured 43.71 ms, so the engine is **1.07× slower than OpenVINO** under equal conditions. "Ideal" is the p5 latency (fastest 5% of iterations), an estimate of the uncontended latency; OpenVINO's estimate (40.78 ms) agrees with its idle-host 40.20 ms. One thread: 215.81 ms.
⁴ Against the fixed 800 GFLOPS assumption (6 × 32 FLOP/cycle × 4.2 GHz), not the measured clock.

Notes on the numbers:
- CPU and ncnn rows are **medians of 3 thermally controlled rounds** (60 s cooldown before every run, rotating order). The GPU rows for eager/AOTInductor/TensorRT come from one back-to-back session; laptop clocks were not locked, so expect about ±3% run to run.
- "% of FP32 peak" = 21.7 GFLOPs per inference / latency, divided by 6 cores × 32 FLOP/cycle × measured P-core clock.

**Takeaways**
- On NVIDIA, TensorRT is the fastest and leaves the leanest runtime (only `libnvinfer` and the CUDA runtime). The cost is a ~1 minute engine build and an engine tied to this GPU and TensorRT version.
- AOTInductor removes Python but not libtorch; good middle ground with a light build.
- Vulkan is portable but far behind CUDA on this NVIDIA GPU, even with tensor cores via cooperative matrix.
- On this CPU, FP16 does not help (no AVX-512 FP16 / AMX). INT8 with AVX-VNNI is the big lever: 2.7× over OpenVINO FP32 for 0.31 mAP.
- OpenVINO FP32 already reaches ~70% of the theoretical FP32 peak. A hand-written engine, generated from the ONNX graph with AVX2 kernels and its own thread pool, got to ~93% of OpenVINO's speed (46.67 vs 43.71 ms in the same session), ahead of PyTorch eager, ncnn and close to ONNX Runtime. Almost all of the remaining time is in dense convolutions.

## Hardware

The results depend heavily on what the hardware can and cannot do, so here is the machine in detail.

| Component | Details |
|---|---|
| **GPU** | NVIDIA GeForce RTX 3050 6GB Laptop GPU: GA107, Ampere, compute capability 8.6, 20 SMs (2560 CUDA cores, 80 third-generation tensor cores), 6 GB GDDR6, up to 2100 MHz, 95 W max power limit. Driver 620.70. |
| **CPU** | Intel Core i5-13450HX (Raptor Lake, hybrid): **6 P-cores** with Hyper-Threading (logical CPUs 0–11, up to 4.6 GHz) + **4 E-cores** (logical CPUs 12–15, up to 3.4 GHz), 16 threads, 9.5 MiB L2, 20 MiB L3. ISA: AVX2, FMA, F16C, **AVX-VNNI**. No AVX-512, no AVX-512 FP16, no AMX. |
| **iGPU** | Intel UHD Graphics (Raptor Lake-S, PCI 8086:A78B), up to 1450 MHz, Mesa 26.2 (Vulkan 1.4). Supports FP16; no cooperative matrix. |
| **Memory** | 16 GB (15 GiB visible). |
| **OS / toolchain** | CachyOS (Arch-based), Linux 7.3, GCC 16.2, CMake 4.4, Ninja 1.13. System CUDA toolkit 13.4 (used by TensorRT); CUDA 12.8 from the PyTorch wheels (used by eager and AOTInductor). |

Why it matters:
- **Tensor cores** (FP16, TF32) are why FP16 roughly doubles GPU throughput, and why TensorRT's default TF32 changes FP32 numerics slightly.
- **Vulkan cooperative matrix** (`VK_KHR_cooperative_matrix`, `VK_NV_cooperative_matrix2`) lets ncnn use those tensor cores from Vulkan; disabling it slows ncnn FP16 from ~9.8 to ~13 ms.
- **Hybrid CPU:** threads that land on E-cores make P-cores wait at every layer, so CPU runs use one thread per P-core (`taskset -c 0,2,4,6,8,10`).
- **F16C only converts** FP16↔FP32; arithmetic stays FP32, so FP16 brings no CPU speedup here. **AVX-VNNI** does 8-bit dot products in hardware, which is why INT8 pays off.
- **Theoretical FP32 peak of the 6 P-cores:** 6 × 32 FLOP/cycle (two 256-bit FMA units) × clock ≈ 800 GFLOPS at ~4.2 GHz.
- **The laptop throttles.** In a back-to-back CPU sequence the P-core clock fell from 4.2 to 3.1 GHz and penalized whichever variant ran last; that is why later rounds use cooldowns and rotating rounds.

## Repository layout

```
webcam_yolo.py          Round 0: YOLOv10s on the webcam in a window, with FPS and GPU memory
aoti/                   Round 1: export + AOTInductor compile, eager reference, eager GPU benchmark
  cpp/                  yolo_aoti: C++ runner (libtorch) with validate / bench / detect
trt/                    Round 2: ONNX export, TensorRT engine build
  cpp/                  yolo_trt: C++ runner (TensorRT only, no libtorch)
cpu/                    Round 3: CPU validation, INT8 quantization, COCO mAP, CPU benchmark
ncnn/                   Round 4: pnnx conversion to ncnn
  cpp/                  yolo_ncnn: C++ runner (ncnn, Vulkan or CPU) + YOLOv10 post-process
engine/                 Own CPU engine: raw ONNX export, code generator (gen/), runtime kernels and thread pool
  cpp/                  yolo_engine: C++ runner with validate / bench / detect (no inference library)
native/                 Shared C++ headers (validation, letterbox, NVML, stats, YOLOv10 post-process) and benchmark scripts
docs/
  report/               Final report (PDF and HTML, Portuguese) and its generator
  REPRODUCE.md          Step-by-step reproduction guide (English)
  *-results.md          Technical report per round (Portuguese)
  superpowers/specs/    Approved design of each round
artifacts/              Generated files (models, engines, references, logs). Not versioned.
```

## Quick start

Requirements: Linux, an NVIDIA GPU with a recent driver, [uv](https://docs.astral.sh/uv/), and (for the webcam demo) a V4L2 webcam. Python 3.12 is installed by uv if needed.

```bash
uv sync
uv run python -c "from ultralytics import YOLO; YOLO('yolov10s.pt')"   # downloads the weights to the repo root
uv run webcam_yolo.py                 # window with detections, FPS and GPU memory; quit with q or Esc
uv run webcam_yolo.py --half          # FP16
uv run webcam_yolo.py --no-show --frames 300   # headless benchmark summary
```

If the webcam only reaches ~15 FPS in low light, its auto exposure is stretching frames: `v4l2-ctl -d /dev/video0 -c exposure_dynamic_framerate=0` restores 30 FPS (darker image; resets on reboot).

## Reproducing the experiment

The full guide, with system packages, every command, expected output and troubleshooting, is in **[docs/REPRODUCE.md](docs/REPRODUCE.md)**. In short:

```bash
# Round 1 – AOTInductor (GPU)
uv run python aoti/export.py --capture-webcam      # .pt2 packages + fixed eager references in artifacts/
uv run python aoti/bench_eager.py
cmake -S aoti/cpp -B aoti/cpp/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$(uv run python -c 'import torch;print(torch.utils.cmake_prefix_path)')"
cmake --build aoti/cpp/build
aoti/cpp/build/yolo_aoti validate artifacts/yolov10s_fp16.pt2 artifacts/ref

# Round 2 – TensorRT (GPU)
uv run python trt/export_onnx.py && bash trt/build_engines.sh
cmake -S trt/cpp -B trt/cpp/build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build trt/cpp/build
trt/cpp/build/yolo_trt validate artifacts/yolov10s_fp16.plan artifacts/ref
bash native/bench_all.sh                            # eager vs AOTInductor vs TensorRT

# Round 3 – CPU
bash cpu/download_coco.sh                           # COCO val2017 + annotations to ~/.cache/compiled-cnn/coco
uv run python cpu/quantize_int8.py
uv run python cpu/validate.py && uv run python cpu/eval_map.py
bash native/bench_cpu.sh                            # ~20 min, thermally controlled

# Round 4 – ncnn / Vulkan
uv run python ncnn/export_ncnn.py
cmake -S ncnn/cpp -B ncnn/cpp/build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build ncnn/cpp/build
bash native/bench_ncnn.sh                           # add --with-intel to include the (slow) iGPU

# Own CPU engine
uv run python engine/export_raw_onnx.py            # ONNX cut before the top-300 selection
uv run python engine/gen/generate.py               # generates model.cpp + embedded weights
cmake -S engine -B engine/build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build engine/build
taskset -c 0,2,4,6,8,10 engine/build/yolo_engine validate artifacts/ref --threads 6
bash native/bench_engine.sh                         # engine 6t/1t vs OpenVINO; BENCH_CONTENDED=1 on a busy host
```

## How results are validated

- **Fixed reference:** 8 images (2 from Ultralytics, 6 webcam frames), preprocessed once (letterbox 640, pad 114, RGB, /255) and run through PyTorch eager with TF32 off. Every backend reads the same files.
- **Tolerances**, against the eager run of the same precision: FP32 boxes ≤ 0.5 px and scores ≤ 0.001; FP16 boxes ≤ 2 px and scores ≤ 0.005; detections with score ≥ 0.25 must match in count, class and IoU ≥ 0.95. Differences are measured on detections paired by IoU, not by row index (TopK may order near-ties differently).
- **INT8** cannot meet strict tolerances, so it is judged by detection agreement (IoU ≥ 0.80, score diff ≤ 0.05) and by COCO mAP50-95 drop ≤ 1.0 point.
- **C++ preprocessing** is checked separately (`detect --check`): the runners letterbox the JPEG themselves and must reproduce the eager detections (IoU ≥ 0.90).
- Tolerances were fixed before each run and never loosened; failures are reported with their numbers.

## Caveats

- One laptop, one GPU, unlocked clocks. Batch 1 and a static 640×640 input only.
- 8 validation images check numerical equivalence, not accuracy; accuracy is only measured (mAP) for the CPU INT8 comparison.
- TensorRT `.plan` engines and AOTInductor `.pt2` packages are specific to the GPU architecture (and, for TensorRT, to the TensorRT version).

## Next step

INT8 in the own engine with AVX-VNNI (target: close to OpenVINO INT8's 14.9 ms), and on the FP32 side, direct convolutions in a blocked layout (NCHWc8) to close the remaining ~7% to OpenVINO.

## Licenses

The YOLOv10s weights and the Ultralytics package are AGPL-3.0. `stb_image.h` (in `native/third_party/`) is public domain / MIT. TensorRT, CUDA and cuDNN are under NVIDIA's licenses.
