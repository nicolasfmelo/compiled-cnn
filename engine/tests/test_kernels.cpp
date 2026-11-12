#include "runtime.h"
#include <iostream>
#include <random>

namespace {
std::mt19937 rng(97);
struct Buffer {
    std::vector<float> storage;
    engine::Tensor t;
    explicit Buffer(std::vector<int> shape) : t{nullptr, std::move(shape)} {
        storage.resize(t.size()); t.data = storage.data();
        std::uniform_real_distribution<float> dist(-1, 1);
        for (auto& x : storage) x = dist(rng);
    }
};
void equal(const Buffer& a, const Buffer& b, float tolerance = 3e-5f) {
    for (size_t i = 0; i < a.storage.size(); ++i)
        if (!std::isfinite(b.storage[i]) || std::abs(a.storage[i]-b.storage[i]) > tolerance*(1+std::abs(a.storage[i])))
            throw std::runtime_error("kernel mismatch at "+std::to_string(i)+": "+std::to_string(a.storage[i])+" vs "+std::to_string(b.storage[i]));
}
void convolution(int ic, int oc, int h, int width, int kernel, int stride, int group, bool activation) {
    int pad = kernel/2, oh = (h+2*pad-kernel)/stride+1, ow = (width+2*pad-kernel)/stride+1;
    Buffer x({1,ic,h,width}), w({oc,ic/group,kernel,kernel}), b({oc}), ref({1,oc,oh,ow}), opt({1,oc,oh,ow});
    int k = ic/group*kernel*kernel;
    std::vector<float> packed((oc+3)/4*k*4, 0);
    for (int c = 0; c < oc; ++c) for (int z = 0; z < k; ++z) packed[(c/4*k+z)*4+c%4] = w.storage[c*k+z];
    engine::Context ctx(true,1,{0});
    engine::ConvParams p{kernel,kernel,stride,stride,pad,pad,group,activation};
    engine::conv_ref(x.t,w.t,b.t,ref.t,p);
    engine::conv(ctx,x.t,w.t,b.t,opt.t,p,packed.data());
    equal(ref,opt,2e-4f);
}
}
int main() {
    try {
        engine::Context ref(false,1,{0}), opt(true,1,{0});
        for (int kernel : {1,3}) for (int stride : {1,2}) for (bool act : {false,true}) {
            convolution(7,9,9,19,kernel,stride,1,act);
            convolution(17,8,13,23,kernel,stride,1,act);
            convolution(65,5,3,71,kernel,stride,1,act); // K and N panel tails
        }
        for (int kernel : {3,7}) for (int stride : {1,2}) for (bool act : {false,true})
            convolution(5,5,9,37,kernel,stride,5,act);
        convolution(6,10,5,9,3,2,2,false); // grouped fallback
        std::cout << "PASS conv1x1/3x3 GEMM K/M/N tails, stride/pad, depthwise3x3/7x7, epilogues\n";
        for (int width : {1,7,19,32}) {
            Buffer x({1,3,5,width}), r({1,3,5,width}), o({1,3,5,width});
            engine::maxpool(ref,x.t,r.t); engine::maxpool(opt,x.t,o.t); equal(r,o,0);
            Buffer rr({1,3,10,width*2}), oo({1,3,10,width*2});
            engine::resize(ref,x.t,rr.t); engine::resize(opt,x.t,oo.t); equal(rr,oo,0);
        }
        std::cout << "PASS maxpool negative inputs/borders and nearest asymmetric/floor resize\n";
        for (int m : {1,7}) for (int n : {3,17,32}) for (int k : {1,9,33}) {
            Buffer a({1,3,m,k}), b({1,3,k,n}), r({1,3,m,n}), o({1,3,m,n});
            engine::matmul(ref,a.t,b.t,r.t); engine::matmul(opt,a.t,b.t,o.t); equal(r,o);
        }
        for (int axis : {1,3}) {
            Buffer x({1,3,5,17}), r({1,3,5,17}), o({1,3,5,17});
            engine::softmax(ref,x.t,r.t,axis); engine::softmax(opt,x.t,o.t,axis); equal(r,o);
        }
        for (int activation : {0,1}) {
            Buffer x({1,3,5,17}), r({1,3,5,17}), o({1,3,5,17});
            engine::unary(ref,x.t,r.t,activation); engine::unary(opt,x.t,o.t,activation); equal(r,o);
        }
        for (int op = 0; op < 4; ++op) for (bool scalar : {false,true}) {
            Buffer a({1,3,5,17}), b(scalar ? std::vector<int>{1} : a.t.shape), r(a.t.shape), o(a.t.shape);
            for (float& v : b.storage) v = std::abs(v)+0.2f;
            engine::binary(ref,a.t,b.t,r.t,op); engine::binary(opt,a.t,b.t,o.t,op); equal(r,o);
        }
        if (engine::sigmoid(-100.f) <= 0 || !std::isfinite(engine::silu(-100.f)) || !std::signbit(engine::silu(-1000.f)))
            throw std::runtime_error("activation extreme");
        std::cout << "PASS attention MatMul/Softmax, binary, activations and extremes\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
