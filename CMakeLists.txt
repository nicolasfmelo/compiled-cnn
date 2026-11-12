# Top-level build for every C++ program of the experiment.
#
#   cmake --preset release && cmake --build --preset release && ctest --preset release
#
# Binaries go to build/bin/. Each runner is built only if its dependencies are
# found (AUTO), or can be forced ON/OFF:
#   -DCNN_AOTI=ON|OFF|AUTO   yolo_aoti   AOTInductor runner  (libtorch + CUDA 12.8 from the uv .venv)
#   -DCNN_TRT=ON|OFF|AUTO    yolo_trt    TensorRT runner     (system TensorRT + CUDA runtime)
#   -DCNN_NCNN=ON|OFF|AUTO   yolo_ncnn   ncnn runner         (system ncnn with Vulkan)
#   -DCNN_ENGINE=ON|OFF|AUTO yolo_engine own CPU engine      (no dependency; needs the generated model)
# The configure step prints which runners were enabled and why the others were skipped.
cmake_minimum_required(VERSION 3.24)
project(compiled_cnn LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
enable_testing()

foreach(runner AOTI TRT NCNN ENGINE)
    set(CNN_${runner} AUTO CACHE STRING "Build this runner: ON, OFF or AUTO (if its dependencies are found)")
    set_property(CACHE CNN_${runner} PROPERTY STRINGS AUTO ON OFF)
endforeach()

add_subdirectory(native)  # cnn_native: shared headers used by all runners

set(CNN_SUMMARY "")
# cnn_runner(<NAME> <dir> <found> <reason-if-missing>)
function(cnn_runner name dir found reason)
    if(CNN_${name} STREQUAL "OFF")
        set(line "  ${name}: OFF (disabled)")
    elseif(found)
        add_subdirectory(${dir})
        set(line "  ${name}: ON  -> ${dir}")
    elseif(CNN_${name} STREQUAL "ON")
        message(FATAL_ERROR "CNN_${name}=ON but ${reason}")
    else()
        set(line "  ${name}: skipped (${reason})")
    endif()
    set(CNN_SUMMARY "${CNN_SUMMARY}${line}\n" PARENT_SCOPE)
endfunction()

# AOTInductor: libtorch and the CUDA 12.8 runtime come from the PyTorch wheels in .venv (uv sync).
file(GLOB _torch_cmake "${CMAKE_SOURCE_DIR}/.venv/lib/python3*/site-packages/torch/share/cmake")
set(CNN_TORCH_CMAKE "${_torch_cmake}" CACHE PATH "torch/share/cmake of the PyTorch wheel (default: from .venv)")
set(_aoti_found FALSE)
if(CNN_TORCH_CMAKE AND EXISTS "${CNN_TORCH_CMAKE}/Torch")
    list(APPEND CMAKE_PREFIX_PATH "${CNN_TORCH_CMAKE}")
    set(_aoti_found TRUE)
endif()
cnn_runner(AOTI aoti/cpp "${_aoti_found}" "PyTorch wheel not found in .venv (run uv sync, or set CNN_TORCH_CMAKE)")

# TensorRT: system headers/library plus the system CUDA runtime in /opt/cuda.
find_path(CNN_NVINFER_H NvInfer.h)
find_path(CNN_CUDART_H cuda_runtime_api.h PATHS /opt/cuda/include NO_DEFAULT_PATH)
set(_trt_found FALSE)
if(CNN_NVINFER_H AND CNN_CUDART_H)
    set(_trt_found TRUE)
endif()
cnn_runner(TRT trt/cpp "${_trt_found}" "TensorRT (NvInfer.h) or CUDA toolkit (/opt/cuda) not found")

# ncnn: system package built with Vulkan.
find_package(ncnn CONFIG QUIET PATHS /usr/lib/cmake/ncnn)
cnn_runner(NCNN ncnn/cpp "${ncnn_FOUND}" "ncnn CMake package not found (pacman -S ncnn)")

# Own engine: no external dependency, but model.cpp/weights are generated from the ONNX first.
set(ENGINE_GENERATED_DIR "${CMAKE_SOURCE_DIR}/artifacts/engine/generated" CACHE PATH "Generated model directory")
set(_engine_found FALSE)
if(EXISTS "${ENGINE_GENERATED_DIR}/model.cpp")
    set(_engine_found TRUE)
endif()
cnn_runner(ENGINE engine "${_engine_found}"
    "generated model missing: run engine/export_raw_onnx.py and engine/gen/generate.py first")

message(STATUS "compiled-cnn runners (binaries in ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}):\n${CNN_SUMMARY}")
