#include <net.h>
#include <gpu.h>
#include <command.h>
#include "postprocess.h"

void checked(int status, const char* operation) {
    if (status) throw std::runtime_error(std::string(operation) + " failed: " + std::to_string(status));
}

struct Arguments {
    std::string command, device = "nvidia", precision = "fp32", mode = "resident";
    fs::path param, bin, extra;
    std::optional<fs::path> check;
    // Diagnostic overrides, never silently substituted for the requested precision.
    bool storage_only = false, no_winograd = false, no_sgemm = false, no_coop = false;
    Arguments(int argc, char** argv) {
        if (argc < 4) throw std::runtime_error(
            "usage: yolo_ncnn validate PARAM BIN REF_DIR | bench PARAM BIN | detect PARAM BIN IMAGE "
            "[--device nvidia|intel|cpu] [--precision fp32|fp16] [--mode resident|host] [--check DETS_TXT]; "
            "diagnostics: --storage-only --no-winograd --no-sgemm --no-coop; "
            "self-test: yolo_ncnn check-postprocess FIXTURE_DIR");
        command = argv[1]; param = argv[2]; bin = argv[3];
        int i = 4;
        if (command == "validate" || command == "detect") {
            if (i == argc) throw std::runtime_error("missing reference directory/image");
            extra = argv[i++];
        } else if (command != "bench") throw std::runtime_error("unknown command");
        for (; i < argc; ++i) {
            std::string key = argv[i];
            if (key == "--storage-only") { storage_only = true; continue; }
            if (key == "--no-winograd") { no_winograd = true; continue; }
            if (key == "--no-sgemm") { no_sgemm = true; continue; }
            if (key == "--no-coop") { no_coop = true; continue; }
            if (++i == argc) throw std::runtime_error("missing option value");
            if (key == "--device") device = argv[i];
            else if (key == "--precision") precision = argv[i];
            else if (key == "--mode") mode = argv[i];
            else if (key == "--check") check = fs::path(argv[i]);
            else throw std::runtime_error("unknown option " + key);
        }
        if ((device != "nvidia" && device != "intel" && device != "cpu") ||
            (precision != "fp32" && precision != "fp16") || (mode != "resident" && mode != "host") ||
            (device == "cpu" && precision != "fp32") || (storage_only && precision != "fp16"))
            throw std::runtime_error("invalid device/precision/mode combination");
        if (device == "cpu") mode = "host";
    }
    std::string variant() const {
        return (device == "cpu" ? "ncnn-cpu" : "ncnn-vk-" + device) + " " + precision;
    }
};

class Runner {
    ncnn::Net net;
    const ncnn::VulkanDevice* gpu = nullptr;
    ncnn::VkAllocator *blob = nullptr, *staging = nullptr;
    ncnn::VkMat resident;
    ncnn::Option transfer;
    bool vulkan;

    ncnn::Mat host_input(const std::vector<float>& data) const {
        if (data.size() != INPUT_SIZE) throw std::runtime_error("wrong input size");
        return ncnn::Mat(640, 640, 3, const_cast<float*>(data.data()));
    }
    std::vector<float> finish(ncnn::Mat raw) {
        ncnn::Mat unpacked;
        ncnn::convert_packing(raw, unpacked, 1);
        if (unpacked.elemsize == 2) {
            ncnn::Mat fp32;
            ncnn::cast_float16_to_float32(unpacked, fp32);
            unpacked = fp32;
        }
        if (unpacked.dims != 2 || unpacked.w != RAW_WIDTH || unpacked.h != ANCHORS || unpacked.elemsize != 4)
            throw std::runtime_error("unexpected raw output: dims=" + std::to_string(unpacked.dims) +
                " w=" + std::to_string(unpacked.w) + " h=" + std::to_string(unpacked.h) +
                " elemsize=" + std::to_string(unpacked.elemsize));
        return postprocess(static_cast<const float*>(unpacked));
    }
public:
    double model_ms = 0, post_ms = 0;
    Runner(const Arguments& args) : vulkan(args.device != "cpu") {
        int index = -1;
        if (vulkan) {
            const std::string needle = args.device == "nvidia" ? "NVIDIA" : "Intel";
            for (int i = 0; i < ncnn::get_gpu_count(); ++i) {
                const auto& info = ncnn::get_gpu_info(i);
                std::cout << "device candidate: " << i << " " << info.device_name()
                          << " cooperative_matrix_supported=" << info.support_cooperative_matrix() << '\n';
                if (std::string(info.device_name()).find(needle) != std::string::npos) index = i;
            }
            if (index < 0) throw std::runtime_error("requested Vulkan device not found: " + needle);
            net.set_vulkan_device(index);
            gpu = ncnn::get_gpu_device(index);
            const auto& info = ncnn::get_gpu_info(index);
            std::cout << "chosen device: " << info.device_name() << " api_version=" << info.api_version()
                      << " driver_version=" << info.driver_version() << " driver=" << info.driver_name()
                      << " cooperative_matrix_supported=" << info.support_cooperative_matrix() << '\n';
        } else std::cout << "chosen device: CPU (6 threads; external affinity required)\n";
        auto& o = net.opt;
        o.num_threads = 6;
        o.use_vulkan_compute = vulkan;
        o.use_fp16_packed = o.use_fp16_storage = args.precision == "fp16";
        o.use_fp16_arithmetic = args.precision == "fp16" && !args.storage_only;
        o.use_bf16_storage = o.use_bf16_packed = false;
        o.use_cooperative_matrix = args.device == "nvidia" && !args.no_coop;
        if (args.no_winograd) o.use_winograd_convolution = false;
        if (args.no_sgemm) o.use_sgemm_convolution = false;
        checked(net.load_param(args.param.c_str()), "load_param");
        checked(net.load_model(args.bin.c_str()), "load_model/pipeline creation");
        if (vulkan) {
            blob = gpu->acquire_blob_allocator();
            staging = gpu->acquire_staging_allocator();
            transfer = net.opt;
            transfer.blob_vkallocator = transfer.workspace_vkallocator = blob;
            transfer.staging_vkallocator = staging;
        }
        const auto& e = net.opt;
        std::cout << "variant: " << args.variant() << " " << args.mode << '\n'
                  << "settings: vulkan=" << e.use_vulkan_compute << " fp16_packed=" << e.use_fp16_packed
                  << " fp16_storage=" << e.use_fp16_storage << " fp16_arithmetic=" << e.use_fp16_arithmetic
                  << " bf16_storage=" << e.use_bf16_storage << " cooperative_matrix=" << e.use_cooperative_matrix
                  << " packing=" << e.use_packing_layout << " winograd=" << e.use_winograd_convolution
                  << " sgemm=" << e.use_sgemm_convolution << " threads=" << e.num_threads
                  << " pipeline_cache=internal-only-no-persistent-cache\n";
    }
    ~Runner() {
        resident.release();
        net.clear();
        if (gpu) {
            gpu->reclaim_blob_allocator(blob);
            gpu->reclaim_staging_allocator(staging);
        }
    }
    void set_input(const std::vector<float>& data) {
        ncnn::VkCompute cmd(gpu);
        cmd.record_upload(host_input(data), resident, transfer);
        checked(cmd.submit_and_wait(), "resident upload");
    }
    std::vector<float> infer(const std::vector<float>& data, bool use_resident = false) {
        auto start = Clock::now();
        auto ex = net.create_extractor();
        ncnn::Mat raw;
        if (vulkan) {
            ex.set_blob_vkallocator(blob);
            ex.set_workspace_vkallocator(blob);
            ex.set_staging_vkallocator(staging);
        }
        if (use_resident) {
            ncnn::VkCompute cmd(gpu);
            ncnn::VkMat output;
            checked(ex.input("in0", resident), "input VkMat");
            checked(ex.extract("out0", output, cmd), "extract VkMat");
            checked(cmd.submit_and_wait(), "model wait");
            model_ms = milliseconds(start);
            start = Clock::now();
            // All decoded scores/boxes are needed for exact top-k; include this
            // download and unpack/cast in post-process time, not hide a transfer.
            ncnn::VkCompute download(gpu);
            download.record_download(output, raw, transfer);
            checked(download.submit_and_wait(), "postprocess download");
        } else {
            checked(ex.input("in0", host_input(data)), "input Mat");
            checked(ex.extract("out0", raw), "extract Mat");
            model_ms = milliseconds(start);
            start = Clock::now();
        }
        auto result = finish(raw);
        post_ms = milliseconds(start);
        return result;
    }
};

void bench(Runner& runner, const Arguments& args, double load_ms) {
    auto data = read_binary(args.param.parent_path().parent_path() / "ref/bus.input.bin", INPUT_SIZE);
    bool resident = args.device != "cpu" && args.mode == "resident";
    if (resident) runner.set_input(data);
    auto iteration = [&] { auto start = Clock::now(); runner.infer(data, resident); return milliseconds(start); };
    double first = iteration();
    int warmup = args.device == "cpu" ? 20 : 50, measured = args.device == "cpu" ? 200 : 500;
    for (int i = 0; i < warmup; ++i) iteration();
    std::vector<double> times;
    double model = 0, post = 0;
    for (int i = 0; i < measured; ++i) {
        times.push_back(iteration()); model += runner.model_ms; post += runner.post_ms;
    }
    double mean = std::accumulate(times.begin(), times.end(), 0.) / times.size();
    std::sort(times.begin(), times.end());
    auto memory = args.device == "nvidia" ? gpu_memory() : std::nullopt;
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage)) throw std::runtime_error("getrusage failed");
    // Local printer avoids querying NVIDIA NVML for the Intel/CPU variants.
    std::cout << "protocol: warmup=" << warmup << " iterations=" << measured << '\n'
              << std::fixed << std::setprecision(2)
              << "latency ms: mean " << mean << " | p50 " << percentile(times, .5)
              << " | p95 " << percentile(times, .95) << '\n'
              << std::setprecision(1) << "throughput: " << 1000 / mean << " it/s\n"
              << "process GPU MiB (NVML): ";
    if (memory) std::cout << *memory; else std::cout << "n/a";
    std::cout << "\npeak RSS MiB: " << usage.ru_maxrss / 1024.0
              << std::setprecision(2) << "\nload ms: " << load_ms << " | first run ms: " << first
              << "\nmodel mean ms: " << model / measured << " | post-process mean ms: " << post / measured
              << "\nmodel MiB: " << (fs::file_size(args.param) + fs::file_size(args.bin)) / 1048576.0 << '\n';
}

int main(int argc, char** argv) {
    try {
        check_host();
        if (argc == 3 && std::string(argv[1]) == "check-postprocess") return check_postprocess(argv[2]);
        Arguments args(argc, argv);
        // GPU instance discovery is outside load timing; net.load_model creates
        // pipelines (including shader compilation). No disk pipeline cache used.
        struct Instance {
            bool enabled;
            explicit Instance(bool on) : enabled(on) { if (on) checked(ncnn::create_gpu_instance(), "Vulkan init"); }
            ~Instance() { if (enabled) ncnn::destroy_gpu_instance(); }
        } instance(args.device != "cpu");
        auto start = Clock::now();
        Runner runner(args);
        double load_ms = milliseconds(start);
        auto infer = [&](const std::vector<float>& data) {
            bool resident = args.device != "cpu" && args.mode == "resident";
            if (resident) runner.set_input(data);
            return runner.infer(data, resident);
        };
        if (args.command == "validate") return validate(infer, args.precision == "fp16", args.extra);
        if (args.command == "detect") return detect(infer, args.precision == "fp16", args.extra, args.check);
        bench(runner, args, load_ms);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }
}
