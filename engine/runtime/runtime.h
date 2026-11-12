#pragma once
#include "thread_pool.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace engine {
struct Tensor {
    float* data;
    std::vector<int> shape;
    size_t size() const { size_t n = 1; for (int d : shape) n *= d; return n; }
};
class Arena {
    float* p_;
public:
    explicit Arena(size_t floats) {
        p_ = static_cast<float*>(std::aligned_alloc(64, (floats * 4 + 63) / 64 * 64));
        if (!p_) throw std::bad_alloc();
    }
    ~Arena() { std::free(p_); }
    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;
    float* data() { return p_; }
};
struct Context {
    bool optimized, profiling = false, int8 = false;
    std::map<std::string, double> profile;
    ThreadPool pool;
    std::vector<std::vector<float>> scratch;
    Context(bool opt, int threads, const std::vector<int>& cpus) : optimized(opt), pool(threads, cpus),
        scratch(threads, std::vector<float>(512*9*64)) {
        if (!opt && threads != 1) throw std::runtime_error("reference implementation is single-threaded");
    }
    template<class F> void ranges(size_t size, F&& f) {
        if (!optimized) { f(0,size); return; }
        pool.parallel((size+4095)/4096, [&](size_t job, int) {
            size_t start = job*4096;
            f(start,std::min(size,start+4096));
        });
    }
    template<class F> void run(const char* kind, F&& f) {
        if (!profiling) { f(); return; }
        auto t = std::chrono::steady_clock::now();
        f();
        profile[kind] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-t).count();
    }
};
inline constexpr int CONV_MR = 6;
struct ConvParams {
    int kh, kw, sy, sx, py, px, group, silu;
    int spatial_tile = 64, output_chunk = 0;
};
struct QuantConv {
    const int8_t* weights; // [ceil(OC/6), ceil(K/4), 6, 4]
    const float* products; // activation_scale * weight_scale[oc]
    const float* corrected_bias;
    float activation_scale;
    int zero_point;
};
bool has_avx_vnni();
void pack_int8(const Tensor&, const Tensor&, ConvParams, int pos, int count,
               float scale, int zero_point, uint8_t*);
void dot_int8(const int8_t*, const uint8_t*, int k4, int32_t*); // 6 x 16
void dot_int8_ref(const int8_t*, const uint8_t*, int k4, int32_t*);
void conv_int8(Context&, const Tensor&, Tensor&, ConvParams, QuantConv, const Tensor* residual = nullptr);
inline float sigmoid(float x) {
    // Branching avoids overflow and preserves subnormal sigmoid(-100).
    if (x >= 0) return 1.f / (1.f + std::exp(-x));
    float e = std::exp(x); return e / (1.f + e);
}
inline float silu(float x) { return x * sigmoid(x); }
void conv_ref(const Tensor&, const Tensor&, const Tensor&, Tensor&, ConvParams, const Tensor* residual = nullptr);
void conv_depthwise(Context&, const Tensor&, const Tensor&, const Tensor&, Tensor&, ConvParams, const Tensor*);
void conv_dense(Context&, const Tensor&, const Tensor&, Tensor&, ConvParams, const float*, const Tensor*);
void conv_opt(Context&, const Tensor&, const Tensor&, const Tensor&, Tensor&, ConvParams, const float*, const Tensor* residual);
void maxpool_opt(const Tensor&, Tensor&);
void resize_opt(const Tensor&, Tensor&);
void matmul_tile(const Tensor&, const Tensor&, Tensor&, int batch, int row, int rows);
void conv(Context&, const Tensor&, const Tensor&, const Tensor&, Tensor&, ConvParams, const float*, const Tensor* residual = nullptr);
void unary_opt(const Tensor&, Tensor&, int);
void softmax_opt(Context&, const Tensor&, Tensor&, int);
void unary(Context&, const Tensor&, Tensor&, int);
void binary(Context&, const Tensor&, const Tensor&, Tensor&, int);
void maxpool(Context&, const Tensor&, Tensor&);
void resize(Context&, const Tensor&, Tensor&);
void concat(Context&, std::initializer_list<const Tensor*>, Tensor&, int);
void split(Context&, const Tensor&, std::initializer_list<Tensor*>, int, std::initializer_list<std::pair<int,int>>);
bool transpose_opt(Context&, const Tensor&, Tensor&, std::initializer_list<int>);
void transpose(Context&, const Tensor&, Tensor&, std::initializer_list<int>);
void matmul(Context&, const Tensor&, const Tensor&, Tensor&);
void softmax(Context&, const Tensor&, Tensor&, int);
}
