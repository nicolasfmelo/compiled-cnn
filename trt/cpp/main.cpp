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
struct Event {
    cudaEvent_t value{};
    Event() { cuda_check(cudaEventCreate(&value)); }
    ~Event() { if (value) cudaEventDestroy(value); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
};
struct Graph {
    cudaGraph_t value{};
    cudaGraphExec_t executable{};
    ~Graph() {
        if (executable) cudaGraphExecDestroy(executable);
        if (value) cudaGraphDestroy(value);
    }
    Graph() = default;
    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;
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
    Event gpu_start, gpu_stop;
    Graph graph;

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
    double gpu_ms = 0;
    InputShape shape;
    bool graph_enabled() const { return graph.executable != nullptr; }
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
        auto dims = engine->getTensorShape("images");
        if (dims.nbDims != 4 || dims.d[0] != 1 || dims.d[1] != 3 ||
            dims.d[2] <= 0 || dims.d[3] <= 0 || dims.d[2] % 32 || dims.d[3] % 32 ||
            dims.d[2] > std::numeric_limits<int>::max() || dims.d[3] > std::numeric_limits<int>::max())
            throw std::runtime_error("unexpected shape/mode/location/format: images");
        check_tensor("images", nvinfer1::TensorIOMode::kINPUT, {1, 3, dims.d[2], dims.d[3]});
        shape = {static_cast<int>(dims.d[2]), static_cast<int>(dims.d[3])};
        check_tensor("output0", nvinfer1::TensorIOMode::kOUTPUT, {1, 300, 6});
        auto type = engine->getTensorDataType("images");
        half = type == nvinfer1::DataType::kHALF;
        if ((!half && type != nvinfer1::DataType::kFLOAT) || engine->getTensorDataType("output0") != type)
            throw std::runtime_error("expected matching FP32 or FP16 input/output dtypes");
        context.reset(engine->createExecutionContext());
        if (!context) throw std::runtime_error("createExecutionContext failed");
        size_t element_size = half ? sizeof(__half) : sizeof(float);
        input = allocate(shape.size() * element_size);
        output = allocate(OUTPUT_SIZE * element_size);
        if (!context->setTensorAddress("images", input.get()) ||
            !context->setTensorAddress("output0", output.get()))
            throw std::runtime_error("setTensorAddress failed");
        stream.sync();
    }
    ~Runner() { cudaStreamSynchronize(stream.value); }
    void enable_graph() {
        if (graph_enabled()) throw std::runtime_error("CUDA Graph already initialized");
        set_input(std::vector<float>(shape.size(), 0.f));
        if (!context->enqueueV3(stream.value)) throw std::runtime_error("enqueueV3 warmup failed");
        stream.sync();
        cuda_check(cudaStreamBeginCapture(stream.value, cudaStreamCaptureModeGlobal));
        bool enqueued = context->enqueueV3(stream.value);
        auto status = cudaStreamEndCapture(stream.value, &graph.value);
        cuda_check(status);
        if (!enqueued) throw std::runtime_error("enqueueV3 capture failed");
        cuda_check(cudaGraphInstantiate(&graph.executable, graph.value, 0));
    }
    void set_input(const std::vector<float>& data) {
        if (data.size() != shape.size()) throw std::runtime_error("wrong input size");
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
        cuda_check(cudaEventRecord(gpu_start.value, stream.value));
        if (graph_enabled()) cuda_check(cudaGraphLaunch(graph.executable, stream.value));
        else if (!context->enqueueV3(stream.value)) throw std::runtime_error("enqueueV3 failed");
        cuda_check(cudaEventRecord(gpu_stop.value, stream.value));
        stream.sync();
        double elapsed = milliseconds(start);
        float device_ms = 0;
        cuda_check(cudaEventElapsedTime(&device_ms, gpu_start.value, gpu_stop.value));
        gpu_ms = device_ms;
        return elapsed;
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
    auto prefix = plan.parent_path() / ("trt/build_" + plan.stem().string());
    std::ifstream seconds(plan.parent_path() / ("build_" + plan.stem().string() + ".seconds"));
    if (!seconds) seconds.open(prefix.string() + ".seconds");
    std::ifstream tf32_file(prefix.string() + ".tf32");
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
void bench(Runner& runner, double load_ms, const fs::path& plan, const fs::path& ref_dir,
           const std::optional<std::string>& label) {
    auto data = read_binary(ref_dir / "bus.input.bin", runner.shape.size());
    runner.set_input(data); // Transfer once, outside every measured forward.
    double first = runner.iteration();
    for (int i = 0; i < 50; ++i) runner.iteration();
    std::vector<double> times, gpu_times;
    times.reserve(500);
    gpu_times.reserve(500);
    for (int i = 0; i < 500; ++i) {
        times.push_back(runner.iteration());
        gpu_times.push_back(runner.gpu_ms);
    }
    std::cout << "variant: ";
    if (label) std::cout << *label;
    else std::cout << "trt-cpp " << (runner.half ? "fp16" : "fp32")
                   << (plan.stem() == "yolov10s_fp32_notf32" ? " notf32" : "");
    std::cout << '\n' << "settings: strongly_typed=on builder_optimization="
              << (label ? "see-label" : "default") << " cudnn_benchmark=n/a graph="
              << (runner.graph_enabled() ? "on" : "off") << " input="
              << runner.shape.height << 'x' << runner.shape.width << '\n';
    print_bench(std::move(times), load_ms, first);
    print_gpu_latency(std::move(gpu_times));
    print_engine(runner, plan);
}

std::vector<Detection> original_detections(Runner& runner, const fs::path& image) {
    auto input = letterbox(image, runner.shape);
    auto dets = detections(runner.infer(input.data));
    for (auto& d : dets)
        for (int c = 0; c < 4; ++c) {
            double pad = c % 2 ? input.top : input.left;
            double limit = c % 2 ? input.height : input.width;
            d.box[c] = static_cast<float>(std::clamp((d.box[c] - pad) / input.gain, 0.0, limit));
        }
    return dets;
}

int compare(Runner& square, Runner& rect, int count, char** images) {
    bool passed = true;
    for (int i = 0; i < count; ++i) {
        fs::path image = images[i];
        auto square_dets = original_detections(square, image);
        auto rect_dets = original_detections(rect, image);
        auto pairs = match(rect_dets, square_dets, 0.90f);
        if (pairs.scores_max > 0.02f) {
            pairs.ok = false;
            pairs.reason += "score tolerance exceeded; ";
        }
        std::cout << (pairs.ok ? "PASS" : "FAIL") << " compare " << image.stem().string()
                  << " dets=" << rect_dets.size() << '/' << square_dets.size()
                  << std::fixed << std::setprecision(6) << " min_iou=" << pairs.min_iou
                  << " scores_max=" << pairs.scores_max;
        if (!pairs.ok) std::cout << " reason=" << pairs.reason;
        std::cout << '\n';
        if (!pairs.ok) {
            auto print = [](const char* side, const std::vector<Detection>& dets) {
                for (const auto& d : dets)
                    std::cout << side << " det cls=" << d.cls << " score=" << d.score
                              << " box=" << d.box[0] << ',' << d.box[1] << ',' << d.box[2] << ',' << d.box[3] << '\n';
            };
            print("rect", rect_dets);
            print("square", square_dets);
        }
        passed &= pairs.ok;
    }
    return passed ? 0 : 1;
}

int main(int argc, char** argv) {
    try {
        if (argc < 3) throw std::runtime_error(
            "usage: yolo_trt validate PLAN REF_DIR [--graph] | "
            "bench PLAN [--ref-dir DIR] [--graph] [--label NAME] | "
            "detect PLAN IMAGE [--check DETS_TXT] | compare SQUARE_PLAN RECT_PLAN IMAGE...");
        std::string mode = argv[1];
        bool graph = false;
        fs::path plan = argv[2], ref_dir = plan.parent_path() / "ref";
        std::optional<std::string> label;
        std::optional<fs::path> check;
        if (mode == "bench") {
            bool have_ref = false;
            for (int i = 3; i < argc; ++i) {
                std::string option = argv[i];
                if (option == "--graph" && !graph) graph = true;
                else if (option == "--ref-dir" && !have_ref && i + 1 < argc) {
                    ref_dir = argv[++i];
                    have_ref = true;
                } else if (option == "--label" && !label && i + 1 < argc) label = argv[++i];
                else throw std::runtime_error("invalid mode/arguments");
            }
        } else if (mode == "validate") {
            if (argc != 4 && !(argc == 5 && std::string(argv[4]) == "--graph"))
                throw std::runtime_error("invalid mode/arguments");
            ref_dir = argv[3];
            graph = argc == 5;
        } else if (mode == "detect") {
            if (argc != 4 && !(argc == 6 && std::string(argv[4]) == "--check"))
                throw std::runtime_error("invalid mode/arguments");
            if (argc == 6) check = argv[5];
        } else if (mode == "compare") {
            if (argc < 5) throw std::runtime_error("invalid mode/arguments");
            for (int i = 4; i < argc; ++i)
                if (std::string(argv[i]).rfind("--", 0) == 0)
                    throw std::runtime_error("invalid mode/arguments");
        } else throw std::runtime_error("invalid mode/arguments");
        check_host();
        cuda_check(cudaSetDevice(0));
        cuda_check(cudaFree(nullptr));
        cuda_check(cudaDeviceSynchronize()); // Context initialization outside load timing.
        auto start = Clock::now();
        Runner runner(plan);
        if (graph) runner.enable_graph();
        double load_ms = milliseconds(start);
        auto infer = [&](const std::vector<float>& data) { return runner.infer(data); };
        if (mode == "compare") {
            Runner rect(argv[3]);
            return compare(runner, rect, argc - 4, argv + 4);
        }
        if (mode == "bench") { bench(runner, load_ms, plan, ref_dir, label); return 0; }
        print_engine(runner, plan);
        if (mode == "validate") return validate(infer, runner.half, ref_dir, runner.shape);
        return detect(infer, runner.half, argv[3], check, runner.shape);
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }
}
