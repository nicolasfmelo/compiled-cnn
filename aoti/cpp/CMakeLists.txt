cmake_minimum_required(VERSION 3.24)
project(yolo_aoti LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# This consumer has no .cu sources: importing the wheel's shared libraries
# directly avoids TorchConfig's unnecessary enable_language(CUDA)/nvcc probe,
# which would select the incompatible system CUDA 13.4 toolkit.
find_path(TORCH_WHEEL_ROOT include/torch/csrc/inductor/aoti_package/model_package_loader.h
    HINTS ${CMAKE_PREFIX_PATH} ${Torch_DIR}
    PATH_SUFFIXES ../.. ../../.. REQUIRED NO_DEFAULT_PATH)
get_filename_component(TORCH_WHEEL_ROOT "${TORCH_WHEEL_ROOT}" ABSOLUTE)
get_filename_component(SITE_PACKAGES "${TORCH_WHEEL_ROOT}/.." ABSOLUTE)
set(TORCH_LIB "${TORCH_WHEEL_ROOT}/lib")
set(CUDA_WHEEL "${SITE_PACKAGES}/nvidia/cuda_runtime")
if(NOT EXISTS "${CUDA_WHEEL}/lib/libcudart.so.12")
    message(FATAL_ERROR "The pinned CUDA 12 wheel runtime is required")
endif()
add_executable(yolo_aoti main.cpp)
target_include_directories(yolo_aoti SYSTEM PRIVATE
    "${TORCH_WHEEL_ROOT}/include"
    "${TORCH_WHEEL_ROOT}/include/torch/csrc/api/include"
    "${CUDA_WHEEL}/include"
    "${SITE_PACKAGES}/nvidia/cuda_cccl/include"
    # CUDA 12.8 crt/ headers absent from cuda-runtime wheel, bundled by Triton.
    "${SITE_PACKAGES}/triton/backends/nvidia/include")
target_compile_options(yolo_aoti PRIVATE -Wall -Wextra -Wpedantic)
foreach(lib torch torch_cpu torch_cuda c10 c10_cuda)
    target_link_libraries(yolo_aoti PRIVATE "${TORCH_LIB}/lib${lib}.so")
endforeach()
target_link_libraries(yolo_aoti PRIVATE "${CUDA_WHEEL}/lib/libcudart.so.12" ${CMAKE_DL_LIBS})
# The torch 2.11 Linux wheel uses the modern C++11 libstdc++ ABI.
target_compile_definitions(yolo_aoti PRIVATE _GLIBCXX_USE_CXX11_ABI=1)
file(GLOB NVIDIA_LIB_DIRS LIST_DIRECTORIES true "${SITE_PACKAGES}/nvidia/*/lib")
set_target_properties(yolo_aoti PROPERTIES
    BUILD_RPATH "${TORCH_LIB};${NVIDIA_LIB_DIRS}"
    INSTALL_RPATH "${TORCH_LIB};${NVIDIA_LIB_DIRS}")
message(STATUS "Torch wheel: ${TORCH_WHEEL_ROOT}; CUDA runtime/headers: ${CUDA_WHEEL} (no nvcc)")
