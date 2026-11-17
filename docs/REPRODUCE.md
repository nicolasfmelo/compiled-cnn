# Reproduction guide

Step-by-step instructions to rebuild every artifact, validate every backend and rerun every benchmark of this experiment. Commands are run from the repository root. Generated files go to `artifacts/` (not versioned).

Tested on CachyOS (Arch-based) with the hardware described in the [README](../README.md#baseline-hardware). Other distributions work if you install the equivalent packages; absolute numbers will differ on other hardware.

## 1. System prerequisites

| Needed for | Package(s) (Arch / CachyOS) | Tested version |
|---|---|---|
| Everything | NVIDIA driver (`nvidia-utils`), `gcc`, `cmake`, `ninja`, `git` | driver 620.70, GCC 16.2, CMake 4.4 |
| Python side | [uv](https://docs.astral.sh/uv/) (installs Python 3.12 if needed) | uv 0.11 |
| Round 2 (TensorRT runner) | `cuda` (system toolkit), `tensorrt` (AUR), `cudnn` | CUDA 13.4, TensorRT 11.4.0.106, cuDNN 9.28 |
| Round 4 (ncnn) | `ncnn`, `vulkan-icd-loader`, `vulkan-tools`; `vulkan-intel` for the Intel iGPU | ncnn 20261020, Vulkan 1.4 |
| Webcam (rounds 0/1) | a V4L2 webcam; `v4l-utils` optional | — |
| Round 3 | `curl`, `unzip` (COCO download) | — |

Install the official-repo packages first, then TensorRT from the AUR:

```bash
sudo pacman -Syu                      # update the whole system first (see troubleshooting)
sudo pacman -S --needed cuda cmake ninja ncnn vulkan-icd-loader vulkan-tools vulkan-intel v4l-utils curl unzip
paru -S --needed --removemake tensorrt
```

Reboot after a kernel or NVIDIA driver update before using CUDA. Check the result:

```bash
nvidia-smi                            # driver loaded, GPU visible
trtexec --help | head -1              # TensorRT installed
ls /usr/include/NvInfer.h /usr/include/ncnn/net.h
vulkaninfo --summary | grep deviceName   # should list the NVIDIA GPU (and the Intel iGPU)
```

You only need the packages of the rounds you want to reproduce: rounds 0, 1 and 3 need just the driver, the compiler tools and uv.

## 2. Python environment and weights

```bash
uv sync                                                               # torch 2.11+cu128, ultralytics, onnx, openvino, nncf, pnnx, ...
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # expect: 2.11.0+cu128 True
uv run python -c "from ultralytics import YOLO; YOLO('yolov10s.pt')"  # downloads yolov10s.pt to the repo root
```

The PyTorch wheels bring their own CUDA 12.8 runtime; the system CUDA toolkit is only used by the TensorRT runner. `aoti/export.py` builds AOTInductor packages against the CUDA 12.8 headers from the wheels on purpose, so the two CUDA versions never mix.

## Building the C++ programs

All C++ code is built by the top-level `CMakeLists.txt`, with presets:

```bash
cmake --preset release            # configure into build/; prints which runners are enabled
cmake --build --preset release    # binaries in build/bin/
ctest --preset release            # own-engine unit tests
```

| Runner | Enabled when | Option |
|---|---|---|
| `yolo_aoti` | the PyTorch wheel exists in `.venv` (`uv sync`); override with `-DCNN_TORCH_CMAKE=<torch>/share/cmake` | `CNN_AOTI` |
| `yolo_trt` | `NvInfer.h` is found and the CUDA toolkit is in `/opt/cuda` | `CNN_TRT` |
| `yolo_ncnn` | the `ncnn` CMake package is found | `CNN_NCNN` |
| `yolo_engine` | `artifacts/engine/generated/model.cpp` exists (section 8); override with `-DENGINE_GENERATED_DIR=` | `CNN_ENGINE` |

Each option is `AUTO` (default: build if found), `ON` (error if not found) or `OFF`. The configure step prints a summary, for example:

```text
-- compiled-cnn runners (binaries in .../build/bin):
  AOTI: ON  -> aoti/cpp
  TRT: skipped (TensorRT (NvInfer.h) or CUDA toolkit (/opt/cuda) not found)
  NCNN: ON  -> ncnn/cpp
  ENGINE: skipped (generated model missing: run engine/export_raw_onnx.py and engine/gen/generate.py first)
```

Re-run `cmake --preset release` after installing a dependency or generating the engine model. The runners share the header-only `cnn_native` target (`native/`). The benchmark scripts read binaries from `build/bin` (override with `CNN_BIN=<dir>`). For the engine under ThreadSanitizer, use `cmake --preset tsan && cmake --build --preset tsan && ctest --preset tsan`.

## 3. Round 0: webcam demo

```bash
uv run webcam_yolo.py                          # window; quit with q, Esc or by closing it
uv run webcam_yolo.py --half                   # FP16
uv run webcam_yolo.py --no-show --frames 300   # headless; prints the benchmark summary
```

The summary reports loop FPS, model latency (mean/p50/p95) and GPU memory (PyTorch allocated/reserved and the process memory seen by the driver). In low light the webcam may deliver only 15 FPS; `v4l2-ctl -d /dev/video0 -c exposure_dynamic_framerate=0` brings it back to 30 FPS (resets on reboot; `=1` to undo).

## 4. Round 1: AOTInductor (GPU)

**Export, references and eager baseline**

```bash
uv run python aoti/export.py --capture-webcam
```

This creates, under `artifacts/`:
- `yolov10s_fp32.pt2` and `yolov10s_fp16.pt2` (AOTInductor packages with embedded weights);
- `data/` with `bus.jpg`, `zidane.jpg` and 6 webcam frames (`--capture-webcam`; without a webcam, drop the flag and only the 2 images are used);
- `ref/` with the preprocessed inputs, the eager FP32/FP16 outputs and the eager detections. **These references are the ground truth for every round; generate them once and keep them.**

```bash
uv run python aoti/bench_eager.py                     # eager-py fp32 and fp16, each in its own process
uv run python aoti/bench_eager.py --cudnn-benchmark   # variant with cuDNN autotuning
```

**C++ runner**

```bash
cmake --preset release && cmake --build --preset release   # see "Building the C++ programs" above

B=build/bin/yolo_aoti
$B validate artifacts/yolov10s_fp32.pt2 artifacts/ref
$B validate artifacts/yolov10s_fp16.pt2 artifacts/ref
$B detect artifacts/yolov10s_fp32.pt2 artifacts/data/bus.jpg --check artifacts/ref/bus.eager_dets.txt
$B bench artifacts/yolov10s_fp16.pt2                 # add --cudnn-benchmark for the autotuned variant
```

Expected `validate` output, one line per image, exit code 0 when all pass:

```text
PASS bus fp32 boxes_max=0.022919 scores_max=0.000593 dets=5/5 min_iou=0.999774 overall_max=504.086304
```

`overall_max` is the raw row-by-row difference and is informational only (TopK may reorder low-score rows).

## 5. Round 2: TensorRT (GPU)

```bash
uv run python trt/export_onnx.py      # artifacts/yolov10s_fp32.onnx and yolov10s_fp16.onnx (no CLI options)
bash trt/build_engines.sh             # three engines: fp32, fp16 and fp32_notf32 (~1 min each)
cmake --build --preset release        # rebuilds; yolo_trt is enabled once TensorRT is installed

B=build/bin/yolo_trt
for plan in yolov10s_fp32 yolov10s_fp32_notf32 yolov10s_fp16; do
  $B validate artifacts/$plan.plan artifacts/ref
done
$B detect artifacts/yolov10s_fp16.plan artifacts/data/bus.jpg --check artifacts/ref/bus.eager_dets.txt
```

Expected: `yolov10s_fp32.plan` (TF32 allowed by default) may show one `FAIL ... score tolerance exceeded` line and exit with code 1; this is the documented TF32 effect. `yolov10s_fp32_notf32.plan` and `yolov10s_fp16.plan` pass 8/8.

Benchmark eager vs AOTInductor vs TensorRT (7 variants, one process each):

```bash
bash scripts/bench_all.sh              # writes artifacts/bench/round2.txt
```

`ldd build/bin/yolo_trt` should list only `libnvinfer.so.11`, `libcudart.so.13` and system libraries.

## 6. Round 3: CPU

```bash
bash cpu/download_coco.sh             # ~1.1 GB, to ~/.cache/compiled-cnn/coco; skips what is already there
uv run python cpu/quantize_int8.py    # OpenVINO FP32 IR + NNCF INT8 IR (preset mixed), ~40 s
uv run python cpu/validate.py         # 5 variants x 8 images; expect 40 PASS
uv run python cpu/eval_map.py         # mAP on val2017 images 1-500 for eager, OV FP32, OV INT8 (~several minutes)
uv run python cpu/test_common.py && uv run python cpu/test_bench.py   # unit tests of the helpers
```

INT8 calibration uses val2017 images 501–800 (by image id), disjoint from the 500 evaluation images. `quantize_int8.py --preset performance` reproduces the first attempt, which failed the detection criterion.

Benchmark (about 20 minutes):

```bash
bash scripts/bench_cpu.sh              # 5 variants x 3 rotating rounds, 60 s cooldown, median summary
cat artifacts/bench/round3-summary.txt
```

Before each run the script checks that the machine is idle and discards the run if other processes use the CPU (a browser, a build or a test suite is enough). Close them during the benchmark. Background services with low load are tolerated.

## 7. Round 4: ncnn with Vulkan

```bash
uv run python ncnn/export_ncnn.py     # pnnx conversion to artifacts/ncnn/yolov10s.ncnn.{param,bin} (no CLI options)
cmake --build --preset release

B=build/bin/yolo_ncnn
P="artifacts/ncnn/yolov10s.ncnn.param artifacts/ncnn/yolov10s.ncnn.bin"
$B check-postprocess artifacts/ncnn                  # C++ YOLOv10 post-process vs Python: bit-match
$B validate $P artifacts/ref --device nvidia --precision fp16
$B validate $P artifacts/ref --device intel --precision fp32
$B validate $P artifacts/ref --device cpu --precision fp32
$B detect $P artifacts/data/bus.jpg --device nvidia --precision fp16 --check artifacts/ref/bus.eager_dets.txt
$B bench $P --device nvidia --precision fp16 --mode resident   # input already on the GPU
$B bench $P --device nvidia --precision fp16 --mode host       # includes upload/download
```

(In zsh, `$P` does not split into two arguments; run the commands in bash or write both paths out.)

ncnn does not support the TopK/Gather of the YOLOv10 head, so the ncnn graph ends at the raw one-to-one head and `yolo_ncnn` runs the post-process in C++; its time is printed separately and included in the total. Expected validation: NVIDIA FP32/FP16 8/8, Intel FP32 8/8, Intel FP16 6/8 (two score differences of 0.005 and 0.009 against a 0.005 limit), CPU 8/8.

Benchmark:

```bash
bash scripts/bench_ncnn.sh             # NVIDIA fp32/fp16 x resident/host + CPU fp32, 3 rounds (~20 min)
bash scripts/bench_ncnn.sh --with-intel   # adds the 4 Intel iGPU variants (each run takes 2-4 minutes)
```

Diagnostics flags of `yolo_ncnn`: `--storage-only` (FP16 storage, FP32 arithmetic), `--no-coop` (disable cooperative matrix), `--no-winograd`, `--no-sgemm`.

## 8. Own CPU engine (PBI-097)

A C++ engine written for this one graph: a Python generator reads the ONNX and emits a specialized forward pass (`model.cpp`) plus a weights blob embedded in the executable with `.incbin`. Kernels (AVX2/FMA GEMM 6×16, packed im2col, depthwise, attention), the arena planner and the pinned thread pool are in `engine/runtime/`. The binary links only libc, libstdc++, libm and libgcc_s. It needs only the compiler toolchain and the uv environment, no extra system packages.

```bash
uv run python engine/export_raw_onnx.py     # artifacts/engine/yolov10s_raw.onnx, output [1,8400,84] (before top-300)
uv run python engine/gen/generate.py        # artifacts/engine/generated/{model.cpp,model.h,weights.bin,weights.S,plan.txt}
cmake --preset release                 # re-run configure once: yolo_engine is enabled when the generated model exists
cmake --build --preset release
ctest --preset release                 # kernel, INT8 and thread-pool unit tests
uv run python engine/tests/test_generator.py           # generator tests (synthetic ONNX graphs)

B=build/bin/yolo_engine
$B validate artifacts/ref --impl ref --threads 1       # plain reference loops (~22 s per image)
$B validate artifacts/ref --impl opt --threads 1
taskset -c 0,2,4,6,8,10 $B validate artifacts/ref --impl opt --threads 6
$B detect artifacts/data/bus.jpg --check artifacts/ref/bus.eager_dets.txt
taskset -c 0,2,4,6,8,10 $B bench --threads 6 --profile   # time per layer kind
ldd $B
```

Unlike the other export scripts, `engine/export_raw_onnx.py` and `engine/gen/generate.py` have a real CLI (`--help` is safe). The top-300 selection runs in C++ (`native/postprocess.h`, shared with the ncnn runner, bit-exact with the Python implementation). Expected: all three `validate` runs 8/8 PASS; `detect` PASS; two generator runs produce identical files.

INT8 path (PBI-098; needs the COCO download from section 6). Dense convolutions run in INT8 with AVX-VNNI; weights are symmetric per output channel, activations asymmetric per tensor, calibrated on val2017 ranks 501–800:

```bash
uv run python engine/quant/calibrate.py     # artifacts/engine/int8/calibration.json (deterministic, ~1 min)
uv run python engine/gen/generate.py --calibration artifacts/engine/int8/calibration.json --int8-method p99999
cmake --build --preset release
uv run python engine/validate_int8.py       # round-3 INT8 criterion via `yolo_engine serve`; expect 8/8
uv run python engine/eval_map.py            # mAP on the 500 evaluation images: fp32 49.33, int8 49.07, drop 0.26
taskset -c 0,2,4,6,8,10 build/bin/yolo_engine bench --precision int8 --threads 6 --profile
```

Without `--calibration` the generator emits the FP32-only model, byte-identical to the PBI-097 output. `--int8-method minmax` reproduces the first attempt (fails the INT8 criterion on 2 images); `--fp32-layers` keeps named convolutions in FP32. `--precision int8` fails with a clear error on CPUs without AVX-VNNI.

Benchmark (engine FP32 and INT8 with 6 threads, OpenVINO FP32 and INT8, same session, 3 rotating rounds, ~20 min):

```bash
bash scripts/bench_engine.sh                     # idle host: runs with external load are discarded and retried
BENCH_CONTENDED=1 bash scripts/bench_engine.sh   # busy host: records external load instead of discarding
cat artifacts/engine/bench/engine-summary.txt
```

The summary includes an "ideal" estimate (median p5 latency, i.e. the fastest 5% of iterations) and the margin of the mean over it, which matters when the host cannot be idled. Build processes still discard a run in both modes. `BENCH_SMOKE=1` does a one-round parse check.

## 9. TensorRT FP16 tuning (PBI-099)

Start with the fixed references, FP16 ONNX from section 5 and a built `yolo_trt`. This creates a separate static 640×384 input (NCHW `1×3×384×640`) and five independent plans; level 3 takes about 1 minute per build and level 5 about 3.5–4 minutes. These commands overwrite their own tuning outputs, not the round-2 plans or references.

```bash
uv run python trt/export_rect.py
bash trt/build_tuned.sh
B=build/bin/yolo_trt
T=artifacts/trt_tuning
$B validate $T/fp16_640_o3_b1.plan artifacts/ref
$B validate $T/fp16_640_o3_b2.plan artifacts/ref
$B validate $T/fp16_640_o5_b1.plan artifacts/ref
$B validate $T/fp16_640_o5_b2.plan artifacts/ref
$B validate $T/fp16_640_o5_b1.plan artifacts/ref --graph
$B validate $T/fp16_384x640_o5.plan artifacts/ref_384x640 --graph
$B detect $T/fp16_384x640_o5.plan artifacts/data/bus.jpg --check artifacts/ref_384x640/bus.eager_fp16_dets.txt
$B detect $T/fp16_384x640_o5.plan artifacts/data/zidane.jpg --check artifacts/ref_384x640/zidane.eager_fp16_dets.txt
$B compare $T/fp16_640_o5_b1.plan $T/fp16_384x640_o5.plan artifacts/data/zidane.jpg artifacts/data/webcam_0{0..5}.jpg
uv run python trt/eager_rect_scores.py
bash trt/profile_tuned.sh
bash scripts/bench_trt_tuning.sh
```

Each validation is expected to pass 8/8 against eager at the same size. The separate rect-vs-square experiment has four expected FAIL lines: zidane exceeds the score-difference limit, and webcam_03–05 have one square-only low-confidence detection each. `compare` returns nonzero, so do not use `set -e` for this sequence; its two-engine logger warning is harmless. The eager scores confirm the drop also occurs in PyTorch, with a preprocessing caveat for webcam_03; see the [technical report](tensorrt-fp16-tuning-results.md). The benchmark uses three rotating rounds, 30 s cooldown, 50 warm-ups and 500 measured iterations; the recorded webcam-profile host mean was 1.69 ms.

## 10. Final report (PDF and HTML)

```bash
uv run --no-project --with weasyprint --with matplotlib python docs/report/build_report.py
```

Writes `docs/report/relatorio-pbi-095.html` (self-contained) and `docs/report/relatorio-pbi-095.pdf` from the same source. Numbers in the report are fixed in the generator; update them there after rerunning benchmarks.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `pacman`/`paru` fails with 404 on a package signature | Local package database out of date. Run `sudo pacman -Syu`, reboot if the kernel or NVIDIA driver changed, then retry. |
| AUR `tensorrt` build fails with `[Errno 17] File exists: .../dist/bindings` | Race between Python wheel targets in a parallel build. Rebuild with one job: `cd ~/.cache/paru/clone/tensorrt && makepkg --config <copy of /etc/makepkg.conf with MAKEFLAGS="-j1"> -sefr`, then `sudo pacman -U tensorrt-*.pkg.tar.zst`. The `-e` flag reuses the C++ build already done. |
| `nvidia-smi` or CUDA fails after a system update | Kernel module and driver libraries differ until reboot. Reboot. |
| AOTInductor compilation picks headers from `/opt/cuda` | Use `aoti/export.py`, which builds a local `CUDA_HOME` from the wheels' CUDA 12.8 headers (needs `nvidia-cuda-cccl-cu12`, already in `pyproject.toml`). |
| CPU or ncnn benchmark keeps discarding runs | Another process is using the CPU. Close the browser and stop builds/tests; check with `uptime` and `top`. For the engine benchmark, `BENCH_CONTENDED=1` records the load instead. |
| A benchmark result looks too slow | Laptop throttling. Use the provided scripts (cooldown + rotating rounds) instead of running variants back to back. |
| Webcam stuck at 15 FPS | `v4l2-ctl -d /dev/video0 -c exposure_dynamic_framerate=0`. |
| `yolo_ncnn` reports device not found | Check `vulkaninfo --summary`; install `vulkan-intel` for the iGPU and make sure the NVIDIA Vulkan ICD is present. |
