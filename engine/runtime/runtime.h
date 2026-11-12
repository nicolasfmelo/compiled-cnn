#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
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
    bool optimized, profiling = false;
    std::map<std::string, double> profile;
    Context(bool opt, int threads, const std::vector<int>& cpus) : optimized(opt) {
        (void)cpus;
        if (threads != 1) throw std::runtime_error("this milestone supports one thread only");
    }
    template<class F> void run(const char* kind, F&& f) {
        if (!profiling) { f(); return; }
        auto t = std::chrono::steady_clock::now();
        f();
        profile[kind] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-t).count();
    }
};
struct ConvParams { int kh, kw, sy, sx, py, px, group, silu; };
inline float sigmoid(float x) {
    // Branching avoids overflow and preserves subnormal sigmoid(-100).
    if (x >= 0) return 1.f / (1.f + std::exp(-x));
    float e = std::exp(x); return e / (1.f + e);
}
inline float silu(float x) { return x * sigmoid(x); }
void conv_ref(const Tensor&, const Tensor&, const Tensor&, Tensor&, ConvParams);
void conv(Context&, const Tensor&, const Tensor&, const Tensor&, Tensor&, ConvParams, const float*);
void unary(Context&, const Tensor&, Tensor&, int);
void binary(Context&, const Tensor&, const Tensor&, Tensor&, int);
void maxpool(Context&, const Tensor&, Tensor&);
void resize(Context&, const Tensor&, Tensor&);
void concat(Context&, std::initializer_list<const Tensor*>, Tensor&, int);
void split(Context&, const Tensor&, std::initializer_list<Tensor*>, int, std::initializer_list<std::pair<int,int>>);
void transpose(Context&, const Tensor&, Tensor&, std::initializer_list<int>);
void matmul(Context&, const Tensor&, const Tensor&, Tensor&);
void softmax(Context&, const Tensor&, Tensor&, int);
}
