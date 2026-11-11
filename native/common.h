#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <unistd.h>
#include <vector>

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#include "third_party/stb_image.h"

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
constexpr size_t INPUT_SIZE = 3 * 640 * 640;
constexpr size_t OUTPUT_SIZE = 300 * 6;
constexpr float CONF = 0.25f;

inline double milliseconds(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

inline std::vector<float> read_binary(const fs::path& path, size_t count) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() != static_cast<std::streamoff>(count * sizeof(float)))
        throw std::runtime_error("missing or wrong-sized reference: " + path.string());
    std::vector<float> data(count);
    file.seekg(0);
    if (!file.read(reinterpret_cast<char*>(data.data()), count * sizeof(float)))
        throw std::runtime_error("cannot read " + path.string());
    if (!std::all_of(data.begin(), data.end(), [](float v) { return std::isfinite(v); }))
        throw std::runtime_error("non-finite reference: " + path.string());
    return data;
}

struct Detection { std::array<float, 4> box; float score; int cls; };
inline std::vector<Detection> detections(const std::vector<float>& output) {
    std::vector<Detection> result;
    for (size_t row = 0; row < 300; ++row) {
        const float* p = output.data() + row * 6;
        if (p[4] >= CONF) result.push_back({{p[0], p[1], p[2], p[3]}, p[4], static_cast<int>(p[5])});
    }
    return result;
}
inline float iou(const Detection& a, const Detection& b) {
    auto area = [](const auto& r) {
        return std::max(0.f, r[2] - r[0]) * std::max(0.f, r[3] - r[1]);
    };
    std::array<float, 4> overlap{std::max(a.box[0], b.box[0]), std::max(a.box[1], b.box[1]),
                               std::min(a.box[2], b.box[2]), std::min(a.box[3], b.box[3])};
    float intersection = area(overlap), total = area(a.box) + area(b.box) - intersection;
    return total > 0 ? intersection / total : 0;
}
struct Match {
    bool ok = true;
    float min_iou = 1;
    std::string reason;
    float boxes_max = 0, scores_max = 0;
};
inline Match match(const std::vector<Detection>& a, const std::vector<Detection>& b, float threshold) {
    Match result;
    if (a.size() != b.size()) return {false, 0, "detection count mismatch"};
    std::vector<bool> used_a(a.size()), used_b(b.size());
    // Greedily take the globally greatest remaining IoU, independently of class.
    for (size_t n = 0; n < a.size(); ++n) {
        float best = -1;
        size_t ai = 0, bi = 0;
        for (size_t i = 0; i < a.size(); ++i) if (!used_a[i])
            for (size_t j = 0; j < b.size(); ++j) if (!used_b[j]) {
                float value = iou(a[i], b[j]);
                if (value > best) { best = value; ai = i; bi = j; }
            }
        used_a[ai] = used_b[bi] = true;
        result.min_iou = std::min(result.min_iou, best);
        for (size_t c = 0; c < 4; ++c)
            result.boxes_max = std::max(result.boxes_max, std::abs(a[ai].box[c] - b[bi].box[c]));
        result.scores_max = std::max(result.scores_max, std::abs(a[ai].score - b[bi].score));
        if (a[ai].cls != b[bi].cls) { result.ok = false; result.reason += "class mismatch; "; }
        if (best < threshold) { result.ok = false; result.reason += "IoU below threshold; "; }
    }
    return result;
}

template<class Infer>
int validate(Infer&& infer, bool half, const fs::path& directory) {
    std::ifstream manifest(directory / "manifest.txt");
    if (!manifest) throw std::runtime_error("cannot open manifest");
    std::string name;
    bool passed = true;
    size_t samples = 0;
    while (std::getline(manifest, name)) {
        if (name.empty()) continue;
        if (name.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-") != std::string::npos)
            throw std::runtime_error("invalid sample name in manifest");
        ++samples;
        auto data = read_binary(directory / (name + ".input.bin"), INPUT_SIZE);
        auto eager = read_binary(directory / (name + (half ? ".eager_fp16.bin" : ".eager_fp32.bin")), OUTPUT_SIZE);
        auto output = infer(data);
        float overall_max = 0;
        for (size_t i = 0; i < OUTPUT_SIZE; ++i)
            overall_max = std::max(overall_max, std::abs(output[i] - eager[i]));
        auto actual = detections(output), expected = detections(eager);
        Match pairs = match(actual, expected, 0.95f);
        float boxes_max = pairs.boxes_max, scores_max = pairs.scores_max;
        bool ok = pairs.ok && boxes_max <= (half ? 2.f : 0.5f) && scores_max <= (half ? 0.005f : 0.001f);
        std::cout << (ok ? "PASS " : "FAIL ") << name << (half ? " fp16" : " fp32")
                  << std::fixed << std::setprecision(6) << " boxes_max=" << boxes_max
                  << " scores_max=" << scores_max << " dets=" << actual.size() << '/' << expected.size()
                  << " min_iou=" << pairs.min_iou << " overall_max=" << overall_max;
        if (!ok) {
            std::cout << " reason=" << pairs.reason;
            if (boxes_max > (half ? 2.f : 0.5f)) std::cout << "box tolerance exceeded; ";
            if (scores_max > (half ? 0.005f : 0.001f)) std::cout << "score tolerance exceeded; ";
        }
        std::cout << '\n';
        passed &= ok;
    }
    if (!samples) throw std::runtime_error("empty manifest");
    return passed ? 0 : 1;
}

// Minimal NVML ABI (v2/v3 process layout), loaded optionally, no NVML headers.
inline std::optional<double> gpu_memory() {
    struct Process { unsigned pid; unsigned long long used; unsigned gpu_id, compute_id; };
    struct Library { void* p; ~Library() { if (p) dlclose(p); } } lib{dlopen("libnvidia-ml.so.1", RTLD_NOW)};
    if (!lib.p) return {};
    auto init = reinterpret_cast<int (*)()>(dlsym(lib.p, "nvmlInit_v2"));
    auto shutdown = reinterpret_cast<int (*)()>(dlsym(lib.p, "nvmlShutdown"));
    auto count = reinterpret_cast<int (*)(unsigned*)>(dlsym(lib.p, "nvmlDeviceGetCount_v2"));
    auto handle = reinterpret_cast<int (*)(unsigned, void**)>(dlsym(lib.p, "nvmlDeviceGetHandleByIndex_v2"));
    auto processes = reinterpret_cast<int (*)(void*, unsigned*, Process*)>(dlsym(lib.p, "nvmlDeviceGetComputeRunningProcesses_v3"));
    if (!init || !shutdown || !count || !handle || !processes || init() != 0) return {};
    struct Shutdown { int (*fn)(); ~Shutdown() { fn(); } } guard{shutdown};
    unsigned devices = 0;
    if (count(&devices)) return {};
    double total = 0;
    bool found = false;
    for (unsigned i = 0; i < devices; ++i) {
        void* device = nullptr;
        if (handle(i, &device)) return {};
        unsigned size = 0;
        int status = processes(device, &size, nullptr);
        if (status == 0 && size == 0) continue;
        if (status != 7) return {}; // NVML_ERROR_INSUFFICIENT_SIZE
        size = size * 2 + 5;
        std::vector<Process> data(size);
        if (processes(device, &size, data.data())) return {};
        for (unsigned j = 0; j < size; ++j)
            if (data[j].pid == static_cast<unsigned>(getpid()) && data[j].used < (1ull << 63)) {
                total += data[j].used / 1048576.0;
                found = true;
            }
    }
    return found ? std::optional<double>(total) : std::nullopt;
}

inline double percentile(const std::vector<double>& sorted, double fraction) {
    double index = (sorted.size() - 1) * fraction;
    size_t lo = static_cast<size_t>(index), hi = std::min(lo + 1, sorted.size() - 1);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (index - lo);
}
inline const std::array<const char*, 80> COCO = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"};

struct ImageInput { std::vector<float> data; int width, height, left, top; double gain; };
inline ImageInput letterbox(const fs::path& path) {
    int width, height, channels;
    std::unique_ptr<unsigned char, decltype(&stbi_image_free)> image(
        stbi_load(path.c_str(), &width, &height, &channels, 3), stbi_image_free);
    if (!image) throw std::runtime_error(std::string("image load failed: ") + stbi_failure_reason());
    double gain = std::min(640.0 / width, 640.0 / height);
    // nearbyint uses round-to-even, matching Python/Ultralytics round.
    int resized_w = static_cast<int>(std::nearbyint(width * gain));
    int resized_h = static_cast<int>(std::nearbyint(height * gain));
    int left = static_cast<int>(std::round((640 - resized_w) / 2.0 - 0.1));
    int top = static_cast<int>(std::round((640 - resized_h) / 2.0 - 0.1));
    ImageInput result{std::vector<float>(INPUT_SIZE, 114.f / 255.f), width, height, left, top, gain};
    // Half-pixel bilinear resize with edge replication, rounded to uint8 before /255.
    for (int y = 0; y < resized_h; ++y) {
        double sy = (y + 0.5) * height / resized_h - 0.5;
        int ybase = static_cast<int>(std::floor(sy));
        double wy = sy - ybase;
        int y0 = std::clamp(ybase, 0, height - 1), y1 = std::clamp(ybase + 1, 0, height - 1);
        for (int x = 0; x < resized_w; ++x) {
            double sx = (x + 0.5) * width / resized_w - 0.5;
            int xbase = static_cast<int>(std::floor(sx));
            double wx = sx - xbase;
            int x0 = std::clamp(xbase, 0, width - 1), x1 = std::clamp(xbase + 1, 0, width - 1);
            for (int c = 0; c < 3; ++c) {
                auto pixel = [&](int py, int px) { return image.get()[(py * width + px) * 3 + c]; };
                double value = (1 - wy) * ((1 - wx) * pixel(y0, x0) + wx * pixel(y0, x1))
                             + wy * ((1 - wx) * pixel(y1, x0) + wx * pixel(y1, x1));
                result.data[c * 640 * 640 + (y + top) * 640 + x + left] =
                    static_cast<float>(std::round(value)) / 255.f;
            }
        }
    }
    return result;
}
inline std::vector<Detection> read_detections(const fs::path& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("cannot open detection reference");
    std::vector<Detection> result;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::istringstream row(line);
        Detection d{};
        if (!(row >> d.cls >> d.score >> d.box[0] >> d.box[1] >> d.box[2] >> d.box[3]))
            throw std::runtime_error("malformed detection reference");
        if (!std::isfinite(d.score) || !std::all_of(d.box.begin(), d.box.end(), [](float v) { return std::isfinite(v); }))
            throw std::runtime_error("non-finite detection reference");
        result.push_back(d);
    }
    return result;
}
template<class Infer>
int detect(Infer&& infer, bool half, const fs::path& image, const std::optional<fs::path>& check) {
    auto input = letterbox(image);
    auto dets = detections(infer(input.data));
    for (auto& d : dets) {
        for (int c = 0; c < 4; ++c) {
            double pad = c % 2 ? input.top : input.left;
            double limit = c % 2 ? input.height : input.width;
            d.box[c] = static_cast<float>(std::clamp((d.box[c] - pad) / input.gain, 0.0, limit));
        }
        std::cout << "class=" << d.cls << " name=" << (d.cls >= 0 && d.cls < 80 ? COCO[d.cls] : "unknown")
                  << std::fixed << std::setprecision(6) << " score=" << d.score
                  << " box=" << d.box[0] << ',' << d.box[1] << ',' << d.box[2] << ',' << d.box[3] << '\n';
    }
    if (!check) return 0;
    auto expected = read_detections(*check);
    auto pairs = match(dets, expected, 0.90f);
    std::cout << (pairs.ok ? "PASS" : "FAIL") << " detect " << image.stem().string()
              << (half ? " fp16" : " fp32") << " dets=" << dets.size() << '/' << expected.size()
              << " min_iou=" << pairs.min_iou;
    if (!pairs.ok) std::cout << " reason=" << pairs.reason;
    std::cout << '\n';
    return pairs.ok ? 0 : 1;
}


inline void print_bench(std::vector<double> times, double load_ms, double first) {
    double mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    std::sort(times.begin(), times.end());
    auto memory = gpu_memory();
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage)) throw std::runtime_error("getrusage failed");
    std::cout << std::fixed << std::setprecision(2)
              << "latency ms: mean " << mean << " | p50 " << percentile(times, 0.5)
              << " | p95 " << percentile(times, 0.95) << '\n'
              << std::setprecision(1) << "throughput: " << 1000 / mean << " it/s\n"
              << "process GPU MiB (NVML): ";
    if (memory) std::cout << *memory; else std::cout << "n/a";
    std::cout << "\npeak RSS MiB: " << usage.ru_maxrss / 1024.0
              << std::setprecision(2) << "\nload ms: " << load_ms << " | first run ms: " << first << '\n';
}

inline void check_host() {
    uint16_t endian = 1;
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
    if (*reinterpret_cast<unsigned char*>(&endian) != 1) throw std::runtime_error("little-endian host required");
}
