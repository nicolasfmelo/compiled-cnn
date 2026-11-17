# TensorRT FP16 tuning (PBI-099) Implementation Plan

> Execution: the `delegate` skill (pi executors in their own worktrees), not subagent-driven-development. Each task below is one delegation with its own spec file in the card's round folder; the spec is the authoritative task text.

**Goal:** Measure the TensorRT FP16 ablation A (640×640, builder level 3) → B (level 5) → C (B + CUDA Graph) → D (C at 640×384) for YOLOv10s on the RTX 3050, validated against PyTorch eager, and report it.

**Architecture:** Generalize the shared C++ helpers (`native/common.h`) to an input shape read from the engine, keeping 640×640 as the default so the other runners do not change. Add a 384×640 reference/ONNX exporter, a tuned build script, CUDA Graph launch and CUDA-event timing in `yolo_trt`, a rect-vs-square comparison mode, and a rotating benchmark script. All new artifacts get new names; no existing artifact is overwritten.

**Tech Stack:** TensorRT 11.4 (`trtexec`, `nvinfer`), CUDA 13.4 runtime, C++17 + CMake/Ninja, Python 3 via `uv` (torch, ultralytics, onnx, onnxruntime).

**Spec:** `docs/superpowers/specs/2026-11-16-trt-fp16-tuning-design.md`

## Global Constraints

- FP16 only; static engines built with `--stronglyTyped`; no INT8, no sparsity, no FP32 changes.
- Input 384×640 means NCHW `1×3×384×640` (height 384, width 640); letterbox `gain = min(640/w, 384/h)`, same rounding and gray 114 padding as today.
- FP16 tolerances unchanged: boxes ≤ 2 px, scores ≤ 5e-3, IoU-matched pairs ≥ 0.95 among detections with score ≥ 0.25, same count and class.
- Rect vs square (D vs C) on the 7 16:9 images: same count and class, IoU ≥ 0.90, score difference ≤ 0.02.
- `artifacts/ref/`, `artifacts/yolov10s_fp16.onnx`, `artifacts/*.plan` and every other existing artifact are never modified.
- `yolo_aoti`, `yolo_ncnn`, `yolo_engine` and `scripts/bench_all.sh` keep their behavior.
- Benchmark protocol per run: input already on the GPU, 50 warm-up, 500 measured iterations; 3 rotating rounds, 30 s cooldown, median.

## Review Focus

- A plan whose input is not `1×3×H×W` with H and W multiples of 32, or whose output is not `1×300×6`: `yolo_trt` must refuse it with a clear error, not read out of bounds.
- `validate` with a ref dir whose `.input.bin` size does not match the engine shape (e.g. a 640 plan with `artifacts/ref_384x640`): must fail with the existing "wrong-sized reference" error.
- CUDA Graph capture on a plan where capture fails: the runner must report the CUDA/TensorRT error and exit 1, never fall back silently to `enqueueV3` while labeling the run as graph.
- Re-running `trt/build_tuned.sh` or `trt/export_rect.py`: must regenerate only their own outputs and never touch the round-2 plans or `artifacts/ref/`.
- `BENCH_ROUNDS=1 BENCH_COOLDOWN=0` smoke runs must write under a separate smoke folder, so they never mix with the real 3-round results.

---

## Round 1 (one delegation, `deep-engineer`): `trt-tuning`

Spec: `~/.local/share/delegate/_cards/Enty/PBI-099/rounds/2026-11-16-1-compiled-cnn/trt-tuning.md`

Steps, all in the same run, one commit each:
1. Shape-generic helpers in `native/common.h` and `aoti/export.py` (defaults unchanged).
2. `trt/export_rect.py`: 384×640 eager FP16 references and ONNX.
3. `trt/build_tuned.sh` and `trt/profile_tuned.sh` (+ `trt/profile_top.py`).
4. `yolo_trt`: shape from the engine, `--graph`, CUDA-event GPU time, `--ref-dir`/`--label`, `compare` mode.
5. `aoti/bench_eager.py --height 384` and `scripts/bench_trt_tuning.sh` (+ `scripts/summarize_trt_tuning.py`).

## Between rounds (me)

Verify and integrate round 1, then run `bash scripts/bench_trt_tuning.sh` (3 rounds) and `bash trt/profile_tuned.sh` myself on an idle GPU.

## Round 2 (one delegation, `fast-executor`): `trt-tuning-docs`

Spec written after the benchmark, with the measured numbers: `docs/tensorrt-fp16-tuning-results.md` (pt-BR), README and `docs/REPRODUCE.md` (English), final report section in `docs/report/` (pt-BR) rebuilt with `build_report.py`.
