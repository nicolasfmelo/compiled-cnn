# Compiled CNN: YOLOv10s on native runtimes

How much faster does a detection network get when it is compiled to a native artifact and run from C++ instead of PyTorch in Python? What does that artifact still depend on, and do the outputs stay the same?

This repository answers those questions for **YOLOv10s** (Ultralytics) on one laptop. The work ran in four rounds with existing compilers and runtimes, then built a C++ inference engine of our own:

| Round | Backend(s) | Device | What it tests |
|---|---|---|---|
| 0 | Ultralytics `predict` on the webcam | NVIDIA GPU | Starting point: the model as people usually run it |
| 1 | **AOTInductor** (`torch.export` → `.pt2`, loaded from C++ via libtorch) | NVIDIA GPU | PyTorch's own ahead-of-time compiler, no Python at run time |
| 2 | **TensorRT 11.4** (ONNX → `.plan` engine, C++ without libtorch) | NVIDIA GPU | NVIDIA's optimizing compiler, leanest GPU runtime |
| 3 | PyTorch eager, **ONNX Runtime**, **OpenVINO** FP32 and **INT8** (NNCF) | CPU | Mature CPU runtimes and the ceiling of the hardware |
| 4 | **ncnn** over **Vulkan** and ncnn on CPU | NVIDIA GPU, Intel iGPU, CPU | Vendor-neutral GPU path |
| Engine | **Own C++ engine** generated from the ONNX graph: AVX2/FMA kernels and INT8 with AVX-VNNI, own calibration, no inference library, no BLAS | CPU | How close a dependency-free, hand-written engine gets |

Every backend is checked against the same fixed PyTorch reference with fixed tolerances and benchmarked with the same protocol. The full write-up is the final report ([PDF](docs/report/relatorio-pbi-095.pdf) and [HTML](docs/report/relatorio-pbi-095.html), in Portuguese). Each round also has a technical report in [`docs/`](docs/) (also in Portuguese).

## Results at a glance

Except for the separate webcam-profile table, the tables show the latency of one forward pass: batch 1, static 640×640 input, with the input already in device memory. FPS = 1000 / mean latency. This is the model alone, with no camera and no drawing.

**NVIDIA RTX 3050 6GB Laptop**

| Backend | FP32 | FP16 | Speedup vs eager (FP16) | Validation |
|---|---|---|---|---|
| PyTorch eager (Python) | 10.02 ms | 5.59 ms | 1.00× | reference |
| AOTInductor (C++) | 8.46 ms | 3.92 ms | 1.43× | 16/16 |
| **TensorRT (C++)** | **7.39 ms** | **2.94 ms (340 FPS)** | **1.90×** | 23/24 ¹ |
| TensorRT FP16, builder level 5 + CUDA Graph (C++) ⁵ | — | 2.78 ms (360 FPS) | 2.02× | 8/8 |
| TensorRT FP32 with `--noTF32` | 7.86 ms | — | — | 8/8 |
| ncnn Vulkan (C++, cooperative matrix) | 18.55 ms | 9.69 ms | 0.58× | 16/16 |

**NVIDIA RTX 3050 6GB Laptop, webcam profile (640×384 input, PBI-099)**

| Backend | Latency | vs eager, same size | vs default 640×640 engine |
|---|---|---|---|
| PyTorch eager FP16 | 3.61 ms | 1.00× | — |
| TensorRT FP16, level 5 + CUDA Graph | 1.69 ms (592 FPS) | 2.14× | 1.76× |

Seven of the eight validation images are 16:9, where 640×384 keeps the same effective resolution as 640×640. Strong detections keep box and class, but low-confidence detections touching the top or bottom edge can drop below the 0.25 threshold ([technical report, pt-BR](docs/tensorrt-fp16-tuning-results.md)).

All PBI-099 engines pass 8/8 against eager at the same input size; the rect-vs-square comparison is an experiment result, not a validation.

**Intel Core i5-13450HX, 6 threads pinned to the 6 P-cores**

| Backend | Precision | Latency | FPS | % of FP32 peak | Validation |
|---|---|---|---|---|---|
| PyTorch eager | FP32 | 55.56 ms | 18.0 | 47% | 8/8 |
| ncnn (C++) | FP32 | 56.89 ms | 17.6 | — | 8/8 |
| ONNX Runtime | FP32 | 43.63 ms | 22.9 | 63% | 8/8 |
| OpenVINO | FP32 | 40.20 ms | 24.9 | 70% | 8/8 |
| **Own engine (C++, generated)** ³ | FP32 | **46.67 ms** (ideal ≈ 45.4) | 21.4 | 58% ⁴ | 8/8 |
| **OpenVINO** | **INT8** | **14.91 ms** | **67.1** | — | 8/8, mAP −0.31 pt ² |
| Own engine (C++, generated) ³ | INT8 | 20.94 ms (ideal ≈ 19.8) | 47.8 | — | 8/8, mAP −0.26 pt |

**Intel UHD Graphics (iGPU), ncnn Vulkan:** ~230 ms in FP16 and ~433 ms in FP32. This is a single uncontrolled round; the benchmark was stopped on purpose because the device is far too slow for this model.

¹ The default TensorRT FP32 engine allows TF32 and misses the score tolerance on 1 of 8 images (0.0022 vs 0.001, same box and class). Built with `--noTF32`, it matches the strict FP32 reference almost bit for bit.
² mAP50-95 on 500 COCO val2017 images: 49.33 (FP32) → 49.02 (INT8). Calibration used 300 different val images.
³ Measured while a browser and a chat app were running (they could not be closed), with OpenVINO in the same session. FP32: engine 46.67 ms vs OpenVINO 43.71 ms (**1.07×**). INT8: engine 20.94 ms vs OpenVINO INT8 17.01 ms (**1.23×**). "Ideal" is the p5 latency (the fastest 5% of iterations), an estimate of the latency without that load. OpenVINO's estimates (40.78 and 15.06 ms) agree with its idle-host results of 40.20 and 14.91 ms. FP32 engine on one thread: 215.81 ms.
⁴ Against a fixed peak of 800 GFLOPS (6 × 32 FLOP/cycle × 4.2 GHz), not the measured clock.
⁵ Measured in the PBI-099 session, where eager FP16 ran at 5.61 ms and the default TensorRT FP16 engine at 2.98 ms; the gain comes from the CUDA Graph, while level 5 adds nothing measurable.

**Notes on the numbers**
- CPU and ncnn rows are **medians of 3 thermally controlled rounds**: a 60 s cooldown before every run, and the order rotates between rounds.
- The GPU rows for eager, AOTInductor and TensorRT come from one back-to-back session. Laptop clocks were not locked, so expect about ±3% from run to run.
- "% of FP32 peak" = 21.7 GFLOPs per inference / latency, divided by 6 cores × 32 FLOP/cycle × the measured P-core clock.

**Takeaways**
- **NVIDIA GPU:** TensorRT is the fastest and leaves the leanest runtime (only `libnvinfer` and the CUDA runtime). The cost is a ~1 minute engine build, and the engine is tied to this GPU and TensorRT version.
- **TensorRT tuning:** CUDA Graph reduces launch overhead; the separate 640×384 webcam profile is faster but changes detections near image edges.
- **AOTInductor** removes Python but not libtorch. It is a good middle ground with a light build.
- **Vulkan** is portable but far behind CUDA on this NVIDIA GPU, even when it uses the tensor cores through cooperative matrix.
- **FP16 on this CPU** does not help, because there is no AVX-512 FP16 and no AMX.
- **INT8 with AVX-VNNI** is the big CPU lever: OpenVINO INT8 is 2.7× faster than OpenVINO FP32 and loses 0.31 mAP point.
- **Our own engine in FP32** reaches ~93% of OpenVINO's speed (46.67 vs 43.71 ms in the same session) with no dependencies. It is ahead of PyTorch eager and ncnn and close to ONNX Runtime.
- **Our own engine in INT8** reaches ~81% of OpenVINO INT8's speed and loses less accuracy (−0.26 vs −0.31 mAP point).

## Baseline hardware

All numbers come from one machine, which is the **baseline** of this study. Absolute latencies will differ on other hardware. The comparisons between backends, and the reasons behind them, mostly carry over to machines with the same features. Those features are listed below.

| Component | Details |
|---|---|
| **Machine** | Laptop on mains power, performance platform profile, CPU governor `powersave` (Intel P-state with HWP), clocks not locked. |
| **GPU** | NVIDIA GeForce RTX 3050 6GB Laptop GPU: GA107 (Ampere), compute capability 8.6, 20 SMs (2560 CUDA cores, 80 third-generation tensor cores), 6 GB GDDR6, up to 2100 MHz, 95 W max power limit. Driver 620.70. |
| **CPU** | Intel Core i5-13450HX (Raptor Lake, hybrid). **6 P-cores** with Hyper-Threading (logical CPUs 0–11, up to 4.6 GHz) + **4 E-cores** (logical CPUs 12–15, up to 3.4 GHz): 16 threads. Per P-core: 48 KiB L1d and 1.25 MiB L2. Shared: 20 MiB L3. ISA: AVX2, FMA, F16C, **AVX-VNNI**. No AVX-512, no AVX-512 FP16, no AMX. |
| **iGPU** | Intel UHD Graphics (Raptor Lake-S, PCI 8086:A78B), up to 1450 MHz, Mesa 26.2 (Vulkan 1.4). Supports FP16; no cooperative matrix. |
| **Memory** | 16 GB (15 GiB visible). |
| **OS and toolchain** | CachyOS (Arch-based), Linux 7.3, GCC 16.2, CMake 4.4, Ninja 1.13, uv 0.11. System CUDA toolkit 13.4 (used by TensorRT). CUDA 12.8 from the PyTorch wheels (used by eager and AOTInductor). |

Why these features matter:
- **Tensor cores** (FP16, TF32) are why FP16 roughly doubles GPU throughput. They are also why TensorRT's default TF32 changes FP32 numerics slightly.
- **Vulkan cooperative matrix** (`VK_KHR_cooperative_matrix`, `VK_NV_cooperative_matrix2`) lets ncnn use those tensor cores from Vulkan. Disabling it slows ncnn FP16 from ~9.8 to ~13 ms.
- **Hybrid CPU:** a thread that lands on an E-core makes the P-cores wait at every layer. CPU runs therefore use one thread per P-core: `taskset -c 0,2,4,6,8,10`.
- **F16C only converts** between FP16 and FP32; the arithmetic stays FP32, so FP16 brings no CPU speedup here.
- **AVX-VNNI** computes 8-bit dot products in hardware (`vpdpbusd`: 32 multiply-adds per instruction against 8 for an FP32 FMA). That is why INT8 pays off.
- **Theoretical FP32 peak of the 6 P-cores:** 6 × 32 FLOP/cycle (two 256-bit FMA units) × clock ≈ 800 GFLOPS at ~4.2 GHz.
- **The laptop throttles.** In a back-to-back CPU sequence, the P-core clock fell from 4.2 to 3.1 GHz and penalized whichever variant ran last. That is why the CPU benchmarks use cooldowns and rotating rounds.

To compare your machine with the baseline:

```bash
lscpu | grep -E 'Model name|Flags' | grep -oE 'Model name.*|avx2|avx_vnni|avx512[a-z_]*|amx[a-z_]*' | sort -u
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv
vulkaninfo --summary | grep -E 'deviceName|apiVersion'
```

## Backends and their dependencies

What each backend needs to **build** the artifact, and what the final program still needs to **run**. The run-time libraries were taken from `ldd` of each binary in `build/bin/`.

| Backend | Build needs | Artifact | Run time needs | Artifact is tied to |
|---|---|---|---|---|
| PyTorch eager | Python 3.12, PyTorch 2.11 (cu128), Ultralytics 8.4 | `yolov10s.pt` (15.9 MiB) | Python, PyTorch, CUDA 12.8 (wheels) | nothing (portable) |
| AOTInductor `yolo_aoti` | PyTorch 2.11 wheels, CUDA 12.8 headers from the wheels (`nvidia-cuda-cccl-cu12`), GCC | `.pt2` package with embedded weights (16–30 MiB) | **libtorch**: `libtorch*`, `libc10*`, plus `libcudart.so.12`, `libcublas`, `libcudnn` and other CUDA libraries, all from the wheels; no Python | GPU architecture (sm_86) and PyTorch version |
| TensorRT `yolo_trt` | ONNX export (onnx, onnxscript), TensorRT 11.4 `trtexec`, CUDA 13.4 toolkit, cuDNN 9 | `.plan` engine (82–109 MiB) | `libnvinfer.so.11`, `libcudart.so.13`, libc/libstdc++ | exact GPU model, TensorRT version and driver |
| ONNX Runtime (Python) | `onnxruntime` 1.31 | `.onnx` (28.4 MiB) | Python, onnxruntime | nothing |
| OpenVINO FP32 / INT8 (Python) | `openvino` 2026.4, `nncf` 3.4 (INT8 calibration), COCO images | IR `.xml/.bin` (28.0 / 7.7 MiB) | Python, OpenVINO runtime | nothing (INT8 needs VNNI or AVX-512 to be fast) |
| ncnn `yolo_ncnn` | `pnnx` (Python) for conversion, system `ncnn` 20261020 built with Vulkan | `.param` + `.bin` (27.8 MiB) | `libncnn.so.1`, `libglslang`, `libSPIRV-Tools`, `libgomp`; Vulkan loader + GPU driver | nothing (any Vulkan 1.1 GPU or CPU) |
| **Own engine `yolo_engine`** | Python (onnx, numpy) for the generator; onnxruntime for INT8 calibration; GCC with AVX2/AVX-VNNI support | **one executable** with weights inside (66.5 MB: FP32 reference, FP32 packed and INT8) | **only** `libc`, `libstdc++`, `libm`, `libgcc_s` | x86-64 with AVX2 + FMA (`-march=x86-64-v3`); AVX-VNNI for `--precision int8` |

System packages used (Arch / CachyOS): `nvidia-utils` 620.70, `cuda` 13.4, `cudnn` 9.28, `tensorrt` 11.4 (AUR), `ncnn` 20261020, `vulkan-icd-loader` 1.4, `vulkan-intel`, `gcc` 16.2, `cmake` 4.4, `ninja` 1.13. Python packages are pinned in [`pyproject.toml`](pyproject.toml) / [`uv.lock`](uv.lock).

## Repository layout

```
CMakeLists.txt          Top-level build of every C++ program (binaries in build/bin/)
CMakePresets.json       Presets: release (all runners found) and tsan (engine tests with ThreadSanitizer)
pyproject.toml, uv.lock Python environment (uv)

webcam_yolo.py          Round 0: YOLOv10s on the webcam in a window, with FPS and GPU memory
aoti/                   Round 1: export.py (fixed references + .pt2 packages), bench_eager.py
  cpp/                  yolo_aoti: C++ runner on libtorch (validate / bench / detect)
trt/                    Round 2: export_onnx.py, build_engines.sh (trtexec)
  export_rect.py        PBI-099: rectangular FP16 ONNX and eager references
  build_tuned.sh        Independent level 3/5 TensorRT builds
  profile_tuned.sh, profile_top.py  Per-layer profiles and top layers
  eager_rect_scores.py  Compare eager scores at both input sizes
  cpp/                  yolo_trt: C++ runner on TensorRT only
cpu/                    Round 3: CPU validation, INT8 quantization (NNCF), COCO mAP, CPU benchmark, COCO download
ncnn/                   Round 4: export_ncnn.py (pnnx conversion)
  cpp/                  yolo_ncnn: C++ runner on ncnn (Vulkan or CPU)
engine/                 Own CPU engine
  export_raw_onnx.py    ONNX cut before the top-300 selection
  gen/generate.py       Code generator: ONNX -> model.cpp + packed weights (FP32 and INT8)
  quant/calibrate.py    INT8 calibration on COCO
  runtime/              Kernels (AVX2 GEMM, AVX-VNNI INT8, depthwise, attention), arena, thread pool
  cpp/main.cpp          yolo_engine (validate / bench / detect / serve)
  validate_int8.py, eval_map.py, engine_client.py, tests/
native/                 Shared C++ headers (cnn_native target): validation, letterbox, NVML, stats, YOLOv10 post-process
scripts/                Benchmark scripts (bench_all, bench_cpu, bench_ncnn, bench_engine, bench_trt_tuning)
  summarize_trt_tuning.py  PBI-099 median and speedup summary
docs/
  report/               Final report (PDF and HTML, Portuguese) and its generator
  REPRODUCE.md          Detailed reproduction guide and troubleshooting (English)
  *-results.md          Technical report per round (Portuguese)
  superpowers/specs/    Approved design of each round
artifacts/              Everything generated: models, engines, references, logs, benchmark outputs. Not versioned.
```

## How to run

The steps below rebuild everything from scratch. All commands are run from the repository root. Generated files go to `artifacts/`.

You don't need every backend. **Steps 1–2 are always needed.** After that, pick the rounds you care about: each later step says what it needs. The CPU rounds and the own engine need no GPU-specific package; the GPU references from step 2 are enough. [docs/REPRODUCE.md](docs/REPRODUCE.md) has the same flow with more detail, the expected outputs and a troubleshooting table.

### Step 0 — System packages

| You want | Install |
|---|---|
| anything | NVIDIA driver (`nvidia-utils`), `gcc`, `cmake`, `ninja`, `git`, [uv](https://docs.astral.sh/uv/) |
| TensorRT (round 2) | `cuda`, `cudnn`, `tensorrt` (AUR) |
| ncnn / Vulkan (round 4) | `ncnn`, `vulkan-icd-loader`, `vulkan-tools`; `vulkan-intel` for the Intel iGPU |
| CPU INT8 and mAP (round 3, engine INT8) | `curl`, `unzip` (to download COCO) |

```bash
sudo pacman -Syu
sudo pacman -S --needed cuda cudnn cmake ninja ncnn vulkan-icd-loader vulkan-tools vulkan-intel v4l-utils curl unzip
paru -S --needed --removemake tensorrt        # only for round 2
```

Reboot after a kernel or NVIDIA driver update, then check with `nvidia-smi`.

### Step 1 — Python environment and weights

```bash
uv sync                                                              # PyTorch 2.11+cu128, Ultralytics, ONNX, OpenVINO, NNCF, pnnx...
uv run python -c "from ultralytics import YOLO; YOLO('yolov10s.pt')" # downloads the weights to the repo root
```

**What it does:** creates `.venv/` with every Python dependency, pinned by `uv.lock`. The PyTorch wheels bring their own CUDA 12.8; the system CUDA is only used by TensorRT. Quick check: `uv run python -c "import torch; print(torch.cuda.is_available())"` prints `True`.

Optional webcam demo (round 0): `uv run webcam_yolo.py` opens a window with detections, FPS and GPU memory. Quit with `q`. Add `--half` for FP16.

### Step 2 — Fixed references and the AOTInductor packages

```bash
uv run python aoti/export.py --capture-webcam     # drop --capture-webcam if you have no webcam
```

**What it does:** this step creates the ground truth for every later step. It writes:
- `artifacts/data/`: 8 test images (`bus.jpg`, `zidane.jpg` and 6 webcam frames);
- `artifacts/ref/`: each image preprocessed once (letterbox 640, RGB, /255), plus the PyTorch eager FP32 and FP16 outputs, all with TF32 off;
- `artifacts/yolov10s_{fp32,fp16}.pt2`: the AOTInductor packages for round 1.

**Run it once and keep `artifacts/ref/`.** Every backend is compared against these files.

### Step 3 — Export each backend's artifact

Run only the lines of the backends you want:

```bash
uv run python trt/export_onnx.py && bash trt/build_engines.sh    # TensorRT: ONNX, then 3 engines with trtexec (~1 min each)
uv run python trt/export_rect.py                                # optional: webcam-profile ONNX and references
bash trt/build_tuned.sh                                         # optional: PBI-099 engines for webcam-profile benchmark
bash cpu/download_coco.sh                                        # COCO val2017 to ~/.cache/compiled-cnn/coco (~1 GB, for INT8 and mAP)
uv run python cpu/quantize_int8.py                               # OpenVINO: FP32 IR + INT8 IR calibrated with NNCF (~40 s)
uv run python ncnn/export_ncnn.py                                # ncnn: pnnx conversion to .param/.bin
uv run python engine/export_raw_onnx.py                          # own engine: ONNX cut before the top-300 selection
uv run python engine/quant/calibrate.py                          # own engine INT8: activation ranges on 300 COCO images
uv run python engine/gen/generate.py \
  --calibration artifacts/engine/int8/calibration.json --int8-method p99999   # own engine: generates model.cpp + weights
```

**What it does:** each line turns the same trained model into the format that backend uses.
- **TensorRT** compiles an engine specific to this GPU.
- **OpenVINO** converts the ONNX and quantizes it to INT8 with NNCF, calibrated on COCO val2017 images 501–800.
- **ncnn** converts with `pnnx`. ncnn has no TopK, so its graph ends at the raw head and the top-300 selection runs in C++.
- **Own engine:**
  1. The generator reads the ONNX and writes a C++ file with one call per fused operation, fixed shapes and all buffers planned in advance.
  2. It also writes the weights already in the layout the kernels read. They are embedded in the executable with `.incbin`.
  3. With `--calibration`, the generator also emits INT8 weights (per output channel) and activation scales. Without that flag, the model is FP32 only.

`trt/export_onnx.py` and `ncnn/export_ncnn.py` have no command-line options: running them, even with `--help`, runs the export.

### Step 4 — Build all C++ programs

```bash
cmake --preset release            # configure: prints which runners will be built and why others are skipped
cmake --build --preset release    # binaries in build/bin/
ctest --preset release            # unit tests of the own engine (kernels, INT8, thread pool)
```

**What it does:** the top-level `CMakeLists.txt` builds the four runners into `build/bin/`. Each runner is built only if its dependencies are found:
- **`yolo_aoti`** needs the PyTorch wheel in `.venv`;
- **`yolo_trt`** needs TensorRT and `/opt/cuda`;
- **`yolo_ncnn`** needs the `ncnn` package;
- **`yolo_engine`** needs the generated model from step 3.

The configure step prints a summary such as `AOTI: ON`, `TRT: skipped (TensorRT ... not found)`. To force a choice, pass `-DCNN_TRT=OFF`, or `ON` to make a missing dependency an error. The same applies to `CNN_AOTI`, `CNN_NCNN` and `CNN_ENGINE`.

After you regenerate the engine model (step 3), run `cmake --build --preset release` again. `cmake --preset tsan && cmake --build --preset tsan && ctest --preset tsan` runs the engine tests under ThreadSanitizer.

### Step 5 — Validate every backend against the references

```bash
B=build/bin
$B/yolo_aoti validate artifacts/yolov10s_fp16.pt2 artifacts/ref            # also fp32
$B/yolo_trt  validate artifacts/yolov10s_fp16.plan artifacts/ref           # also fp32, fp32_notf32
$B/yolo_ncnn validate artifacts/ncnn/yolov10s.ncnn.param artifacts/ncnn/yolov10s.ncnn.bin artifacts/ref --device nvidia --precision fp16
$B/yolo_engine validate artifacts/ref --threads 6                          # own engine FP32
uv run python engine/validate_int8.py                                      # own engine INT8 (round-3 INT8 criterion)
uv run python cpu/validate.py                                              # eager, ONNX Runtime, OpenVINO FP32/INT8 on CPU
$B/yolo_trt detect artifacts/yolov10s_fp16.plan artifacts/data/bus.jpg --check artifacts/ref/bus.eager_dets.txt
```

**What it does:** each runner processes the 8 reference inputs and compares its detections with the eager output, using the tolerances in [How results are validated](#how-results-are-validated). It prints one `PASS`/`FAIL` line per image and exits with code 0 only if all pass. `detect` additionally loads the JPEG and does the letterbox in C++, which checks the C++ preprocessing too.

Accuracy on COCO (500 images): `uv run python cpu/eval_map.py` (eager, OpenVINO FP32/INT8) and `uv run python engine/eval_map.py` (own engine FP32/INT8).

### Step 6 — Benchmarks

```bash
bash scripts/bench_all.sh        # GPU: eager vs AOTInductor vs TensorRT, 7 variants          -> artifacts/bench/round2.txt
bash scripts/bench_trt_tuning.sh # PBI-099: TensorRT FP16 ablation, 3 rotating rounds
bash scripts/bench_cpu.sh        # CPU: eager, ORT, OpenVINO FP32/INT8, 3 rounds (~20 min)     -> artifacts/bench/round3-summary.txt
bash scripts/bench_ncnn.sh       # ncnn Vulkan + CPU, 3 rounds (~20 min); --with-intel adds the iGPU
bash scripts/bench_engine.sh     # own engine FP32/INT8 vs OpenVINO FP32/INT8 (~20 min)     -> artifacts/engine/bench/engine-summary.txt
```

**What it does:** the GPU benchmark runs 50 warm-up and 500 measured iterations; the CPU benchmarks run 20 and 200. Each variant runs in its own process. The CPU scripts add the thermal protocol:
- a 60 s cooldown before every run;
- 3 rounds, with the order rotated between rounds;
- the median of the 3 rounds;
- a contention monitor that discards and retries a run if another process uses the CPU.

Close the browser and stop builds before running them. If you can't, `BENCH_CONTENDED=1 bash scripts/bench_engine.sh` records the external load instead of discarding runs, and reports an "ideal" latency estimated from the fastest 5% of iterations.

### Step 7 — Final report

```bash
uv run --no-project --with weasyprint --with matplotlib python docs/report/build_report.py
```

Writes `docs/report/relatorio-pbi-095.html` and `.pdf` from the same source. The numbers are written in the generator; update them there after rerunning the benchmarks.

## How results are validated

- **Fixed reference:** 8 images, preprocessed once and run through PyTorch eager with TF32 off (step 2). Every backend reads the same files.
- **Strict tolerances** for FP32 and FP16, against the eager run of the same precision:
  - FP32: boxes within 0.5 px and scores within 0.001;
  - FP16: boxes within 2 px and scores within 0.005;
  - in both, detections with score ≥ 0.25 must match in count, class and IoU ≥ 0.95.

  Differences are measured on detections paired by IoU, not by row index, because TopK may order near-ties differently.
- **INT8** cannot meet strict tolerances, so it is judged two ways:
  - by agreement with the FP32 detections: pairs at IoU ≥ 0.80 and score difference ≤ 0.05; an unpaired detection is allowed only below score 0.30;
  - by a COCO mAP50-95 drop of at most 1.0 point, on 500 val images that are separate from the calibration images.
- **C++ preprocessing** is checked separately with `detect --check` (IoU ≥ 0.90 against the eager detections).
- Tolerances were fixed before each run and never loosened. Failures are reported with their numbers.
- Total: **149 of 152** validations pass. The 3 exceptions have known causes: TF32 in TensorRT FP32 (1 image) and FP16 on the Intel iGPU (2 images).

## Caveats

- One laptop, one GPU, unlocked clocks. Batch 1 and static inputs: 640×640 and the separate 640×384 webcam profile.
- The 8 validation images check numerical equivalence, not accuracy. Accuracy (mAP) was measured for the CPU INT8 comparisons only.
- TensorRT `.plan` engines and AOTInductor `.pt2` packages are specific to the GPU architecture (and, for TensorRT, to the TensorRT version).
- The own-engine benchmarks ran with a browser and a chat app open. OpenVINO ran in the same sessions for a fair comparison, and the p5-based "ideal" is an estimate.

## Next step

Keep activations in u8 between consecutive INT8 convolutions, by quantizing in the producer's epilogue instead of in the consumer's packing. That should close most of the 1.23× gap to OpenVINO INT8. On the FP32 side, the next step is direct convolutions in a blocked layout (NCHWc8).

## Licenses

This repository is licensed under the [GNU AGPL-3.0](LICENSE), matching its main dependency: the YOLOv10s weights and the Ultralytics package are AGPL-3.0. `stb_image.h` (in `native/third_party/`) is public domain / MIT. TensorRT, CUDA and cuDNN are under NVIDIA's licenses.
