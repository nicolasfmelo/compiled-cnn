# Header-only code shared by every runner: validation against the fixed
# references, letterbox preprocessing, NVML, statistics, YOLOv10 top-300 post-process.
add_library(cnn_native INTERFACE)
target_include_directories(cnn_native INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")
# Third-party header (stb_image): SYSTEM so its warnings do not hide ours.
target_include_directories(cnn_native SYSTEM INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
target_compile_features(cnn_native INTERFACE cxx_std_17)
