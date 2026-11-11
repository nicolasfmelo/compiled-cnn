#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <c10/core/InferenceMode.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <cuda_runtime_api.h>
#include "common.h"

using Loader = torch::inductor::AOTIModelPackageLoader;

void cuda_check(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
void sync_cuda() { cuda_check(cudaDeviceSynchronize()); }

at::Tensor gpu_input(std::vector<float>& data, bool half) {
    return at::from_blob(data.data(), {1, 3, 640, 640}, at::kFloat)
        .to(at::Device(at::kCUDA, 0), half ? at::kHalf : at::kFloat);
}
at::Tensor forward(Loader& loader, const at::Tensor& input) {
    auto outputs = loader.run({input});
    if (outputs.size() != 1 || outputs[0].sizes() != at::IntArrayRef({1, 300, 6}))
        throw std::runtime_error("expected exactly one output with shape [1,300,6]");
    return outputs[0];
}
std::vector<float> cpu_output(Loader& loader, const at::Tensor& input) {
    auto output = forward(loader, input).to(at::kCPU, at::kFloat).contiguous();
    const auto* p = output.const_data_ptr<float>();
    std::vector<float> result(p, p + OUTPUT_SIZE);
    if (!std::all_of(result.begin(), result.end(), [](float v) { return std::isfinite(v); }))
        throw std::runtime_error("non-finite model output");
    return result;
}
void bench(Loader& loader, bool half, double load_ms, const fs::path& package) {
    auto data = read_binary(package.parent_path() / "ref/bus.input.bin", INPUT_SIZE);
    auto input = gpu_input(data, half);
    auto iteration = [&]() {
        sync_cuda();
        auto start = Clock::now();
        auto output = forward(loader, input);
        sync_cuda();
        return milliseconds(start);
    };
    double first = iteration();
    for (int i = 0; i < 50; ++i) iteration();
    std::vector<double> times;
    times.reserve(500);
    for (int i = 0; i < 500; ++i) times.push_back(iteration());
    const auto& context = at::globalContext();
    std::cout << "variant: aoti-cpp " << (half ? "fp16" : "fp32")
              << (context.benchmarkCuDNN() ? " cudnn-bench" : "") << '\n'
              << "settings: cudnn_tf32=" << (context.allowTF32CuDNN() ? "on" : "off")
              << " matmul_tf32=" << (context.allowTF32CuBLAS() ? "on" : "off")
              << " cudnn_benchmark=" << (context.benchmarkCuDNN() ? "on" : "off") << '\n';
    print_bench(std::move(times), load_ms, first);
}

int main(int argc, char** argv) {
    try {
        if (argc < 3) throw std::runtime_error(
            "usage: yolo_aoti validate PACKAGE REF_DIR | bench PACKAGE [--cudnn-benchmark] | detect PACKAGE IMAGE [--check DETS_TXT]");
        std::string mode = argv[1];
        if ((mode == "validate" && argc != 4) ||
            (mode == "bench" && !(argc == 3 || (argc == 4 && std::string(argv[3]) == "--cudnn-benchmark"))) ||
            (mode == "detect" && !(argc == 4 || (argc == 6 && std::string(argv[4]) == "--check"))) ||
            (mode != "validate" && mode != "bench" && mode != "detect"))
            throw std::runtime_error("invalid mode/arguments");
        check_host();
        fs::path package = argv[2];
        // Names are part of the export contract; reject unknown precision rather than guessing.
        std::string filename = package.filename().string();
        bool half = filename == "yolov10s_fp16.pt2";
        if (!half && filename != "yolov10s_fp32.pt2") throw std::runtime_error("expected yolov10s_fp32.pt2 or yolov10s_fp16.pt2");
        c10::InferenceMode guard;
        if (mode == "bench") {
            // Explicitly match the Python benchmark, independently of libtorch defaults.
            auto& context = at::globalContext();
            context.setAllowTF32CuDNN(true);
            context.setAllowTF32CuBLAS(false);
            context.setBenchmarkCuDNN(argc == 4);
        }
        cuda_check(cudaSetDevice(0));
        cuda_check(cudaFree(nullptr)); // Initialize context outside package load timing.
        sync_cuda();
        auto start = Clock::now();
        Loader loader(package.string(), "model", false, 1, 0);
        sync_cuda();
        double load_ms = milliseconds(start);
        auto infer = [&](std::vector<float>& data) { return cpu_output(loader, gpu_input(data, half)); };
        if (mode == "validate") return validate(infer, half, argv[3]);
        if (mode == "bench") { bench(loader, half, load_ms, package); return 0; }
        return detect(infer, half, argv[3], argc == 6 ? std::optional<fs::path>(argv[5]) : std::nullopt);
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }
}
