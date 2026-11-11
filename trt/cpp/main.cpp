#include <NvInferRuntime.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include "common.h"

void cuda_check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
struct Logger : nvinfer1::ILogger {
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kWARNING) std::cerr << "[TRT] " << message << '\n';
    }
};
struct Stream {
    cudaStream_t value{};
    Stream() { cuda_check(cudaStreamCreate(&value)); }
    ~Stream() { if (value) cudaStreamDestroy(value); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    void sync() const { cuda_check(cudaStreamSynchronize(value)); }
};
struct FreeDevice {
    void operator()(void* p) const noexcept { if (p) cudaFree(p); }
};
using DeviceBuffer = std::unique_ptr<void, FreeDevice>;
DeviceBuffer allocate(size_t bytes) {
    void* p = nullptr;
    cuda_check(cudaMalloc(&p, bytes));
    return DeviceBuffer(p);
}

class Runner {
    // Reverse destruction releases context/engine/runtime before stream/logger.
    Logger logger;
    Stream stream;
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    DeviceBuffer input, output;

    void check_tensor(const char* name, nvinfer1::TensorIOMode mode,
                      std::initializer_list<int64_t> shape) {
        auto dims = engine->getTensorShape(name);
        if (engine->getTensorIOMode(name) != mode || dims.nbDims != static_cast<int>(shape.size()) ||
            !std::equal(shape.begin(), shape.end(), dims.d) ||
            engine->getTensorLocation(name) != nvinfer1::TensorLocation::kDEVICE ||
            engine->getTensorFormat(name) != nvinfer1::TensorFormat::kLINEAR)
            throw std::runtime_error(std::string("unexpected shape/mode/location/format: ") + name);
    }
public:
    bool half = false;
    double deserialize_ms = 0;
    explicit Runner(const fs::path& plan) {
        std::ifstream file(plan, std::ios::binary | std::ios::ate);
        if (!file || file.tellg() <= 0) throw std::runtime_error("cannot read plan: " + plan.string());
        std::vector<char> bytes(static_cast<size_t>(file.tellg()));
        file.seekg(0);
        if (!file.read(bytes.data(), bytes.size())) throw std::runtime_error("incomplete plan read");
        runtime.reset(nvinfer1::createInferRuntime(logger));
        if (!runtime) throw std::runtime_error("createInferRuntime failed");
        auto start = Clock::now();
        engine.reset(runtime->deserializeCudaEngine(bytes.data(), bytes.size()));
        stream.sync();
        deserialize_ms = milliseconds(start);
        if (!engine) throw std::runtime_error("deserializeCudaEngine failed");
        if (engine->getNbIOTensors() != 2) throw std::runtime_error("expected two I/O tensors");
        check_tensor("images", nvinfer1::TensorIOMode::kINPUT, {1, 3, 640, 640});
        check_tensor("output0", nvinfer1::TensorIOMode::kOUTPUT, {1, 300, 6});
        auto type = engine->getTensorDataType("images");
        half = type == nvinfer1::DataType::kHALF;
        if ((!half && type != nvinfer1::DataType::kFLOAT) || engine->getTensorDataType("output0") != type)
            throw std::runtime_error("expected matching FP32 or FP16 input/output dtypes");
        context.reset(engine->createExecutionContext());
        if (!context) throw std::runtime_error("createExecutionContext failed");
        size_t element_size = half ? sizeof(__half) : sizeof(float);
        input = allocate(INPUT_SIZE * element_size);
        output = allocate(OUTPUT_SIZE * element_size);
        if (!context->setTensorAddress("images", input.get()) ||
            !context->setTensorAddress("output0", output.get()))
            throw std::runtime_error("setTensorAddress failed");
        stream.sync();
    }
    ~Runner() { cudaStreamSynchronize(stream.value); }
    void set_input(const std::vector<float>& data) {
        if (data.size() != INPUT_SIZE) throw std::runtime_error("wrong input size");
        if (half) {
            std::vector<__half> converted(data.size());
            std::transform(data.begin(), data.end(), converted.begin(), [](float v) { return __float2half_rn(v); });
            cuda_check(cudaMemcpyAsync(input.get(), converted.data(), converted.size() * sizeof(__half),
                                       cudaMemcpyHostToDevice, stream.value));
            stream.sync(); // Keep converted's host storage alive through the transfer.
        } else {
            cuda_check(cudaMemcpyAsync(input.get(), data.data(), data.size() * sizeof(float),
                                       cudaMemcpyHostToDevice, stream.value));
            stream.sync();
        }
    }
    double iteration() {
        stream.sync();
        auto start = Clock::now();
        if (!context->enqueueV3(stream.value)) throw std::runtime_error("enqueueV3 failed");
        stream.sync();
        return milliseconds(start);
    }
    std::vector<float> infer(const std::vector<float>& data) {
        set_input(data);
        iteration();
        std::vector<float> result(OUTPUT_SIZE);
        if (half) {
            std::vector<__half> converted(OUTPUT_SIZE);
            cuda_check(cudaMemcpyAsync(converted.data(), output.get(), converted.size() * sizeof(__half),
                                       cudaMemcpyDeviceToHost, stream.value));
            stream.sync();
            std::transform(converted.begin(), converted.end(), result.begin(), [](const __half& v) { return __half2float(v); });
        } else {
            cuda_check(cudaMemcpyAsync(result.data(), output.get(), result.size() * sizeof(float),
                                       cudaMemcpyDeviceToHost, stream.value));
            stream.sync();
        }
        if (!std::all_of(result.begin(), result.end(), [](float v) { return std::isfinite(v); }))
            throw std::runtime_error("non-finite model output");
        return result;
    }
};

void print_engine(const Runner& runner, const fs::path& plan) {
    std::string precision = runner.half ? "fp16" : "fp32";
    auto prefix = plan.parent_path() / ("trt/build_" + precision);
    std::ifstream seconds(prefix.string() + ".seconds"), tf32_file(prefix.string() + ".tf32");
    double build_s;
    std::string tf32;
    std::cout << std::fixed << std::setprecision(3) << "engine: build s ";
    if (seconds >> build_s && std::isfinite(build_s) && build_s >= 0) std::cout << build_s;
    else std::cout << "n/a";
    std::cout << " | plan MiB " << fs::file_size(plan) / 1048576.0
              << " | deserialize ms " << runner.deserialize_ms << '\n';
    if (!(tf32_file >> tf32) || (tf32 != "on" && tf32 != "off" && tf32 != "n/a")) tf32 = "n/a";
    std::cout << "tf32: " << tf32 << '\n';
}
void bench(Runner& runner, double load_ms, const fs::path& plan) {
    auto data = read_binary(plan.parent_path() / "ref/bus.input.bin", INPUT_SIZE);
    runner.set_input(data); // Transfer once, outside every measured forward.
    double first = runner.iteration();
    for (int i = 0; i < 50; ++i) runner.iteration();
    std::vector<double> times;
    times.reserve(500);
    for (int i = 0; i < 500; ++i) times.push_back(runner.iteration());
    std::cout << "variant: trt-cpp " << (runner.half ? "fp16" : "fp32") << '\n'
              << "settings: strongly_typed=on builder_optimization=default cudnn_benchmark=n/a\n";
    print_bench(std::move(times), load_ms, first);
    print_engine(runner, plan);
}

int main(int argc, char** argv) {
    try {
        if (argc < 3) throw std::runtime_error(
            "usage: yolo_trt validate PLAN REF_DIR | bench PLAN | detect PLAN IMAGE [--check DETS_TXT]");
        std::string mode = argv[1];
        if ((mode == "validate" && argc != 4) || (mode == "bench" && argc != 3) ||
            (mode == "detect" && !(argc == 4 || (argc == 6 && std::string(argv[4]) == "--check"))) ||
            (mode != "validate" && mode != "bench" && mode != "detect"))
            throw std::runtime_error("invalid mode/arguments");
        check_host();
        cuda_check(cudaSetDevice(0));
        cuda_check(cudaFree(nullptr));
        cuda_check(cudaDeviceSynchronize()); // Context initialization outside load timing.
        fs::path plan = argv[2];
        auto start = Clock::now();
        Runner runner(plan);
        double load_ms = milliseconds(start);
        auto infer = [&](const std::vector<float>& data) { return runner.infer(data); };
        if (mode == "bench") { bench(runner, load_ms, plan); return 0; }
        print_engine(runner, plan);
        if (mode == "validate") return validate(infer, runner.half, argv[3]);
        return detect(infer, runner.half, argv[3], argc == 6 ? std::optional<fs::path>(argv[5]) : std::nullopt);
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }
}
