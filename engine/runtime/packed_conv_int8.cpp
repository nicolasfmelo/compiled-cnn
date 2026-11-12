#include "runtime.h"
#include "exp_avx2.h"
#include <immintrin.h>

namespace engine {
namespace {
__m256i quantize(__m256 v, float inverse, int zp) {
    // Round before adding the integer zero-point. Clamp in float first to
    // avoid cvtps overflow for large, out-of-calibration finite inputs.
    v=_mm256_mul_ps(v,_mm256_set1_ps(inverse));
    v=_mm256_min_ps(_mm256_set1_ps(255-zp),_mm256_max_ps(_mm256_set1_ps(-zp),v));
    return _mm256_add_epi32(_mm256_cvtps_epi32(v),_mm256_set1_epi32(zp));
}
__m256 even8(const float* p) {
    return _mm256_permutevar8x32_ps(_mm256_shuffle_ps(_mm256_loadu_ps(p),_mm256_loadu_ps(p+8),0x88),
                                   _mm256_setr_epi32(0,1,4,5,2,3,6,7));
}
__m256i interleave(__m256i a, __m256i b, __m256i c, __m256i d) {
    return _mm256_or_si256(_mm256_or_si256(a,_mm256_slli_epi32(b,8)),
                          _mm256_or_si256(_mm256_slli_epi32(c,16),_mm256_slli_epi32(d,24)));
}
template<class Finish>
void microkernel(const int8_t* a, const uint8_t* b, int k4, Finish finish) {
    __m256i c00=_mm256_setzero_si256(),c01=c00,c10=c00,c11=c00,c20=c00,c21=c00,
            c30=c00,c31=c00,c40=c00,c41=c00,c50=c00,c51=c00;
    for (int k=0; k<k4; ++k) {
        const __m256i b0=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b+k*64));
        const __m256i b1=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b+k*64+32));
        // memcpy avoids alignment/strict-aliasing assumptions on s8 weights.
        int32_t w;
#define DOT(R) std::memcpy(&w,a+k*24+R*4,4); { const __m256i v=_mm256_set1_epi32(w); \
        c##R##0=_mm256_dpbusd_avx_epi32(c##R##0,b0,v); \
        c##R##1=_mm256_dpbusd_avx_epi32(c##R##1,b1,v); }
        DOT(0) DOT(1) DOT(2) DOT(3) DOT(4) DOT(5)
#undef DOT
    }
    const __m256i accum[6][2]={{c00,c01},{c10,c11},{c20,c21},{c30,c31},{c40,c41},{c50,c51}};
    finish(accum);
}
}
void pack_int8(const Tensor& x, const Tensor& y, ConvParams p, int pos, int count,
               float scale, int zp, uint8_t* dst) {
    const int ih=x.shape[2],iw=x.shape[3],ow=y.shape[3],area=ih*iw;
    const int ks=p.kh*p.kw,k=x.shape[1]*ks,k4=(k+3)/4;
    const float inverse=1.f/scale;
    struct Mapping { int offsets[16]; int fast; };
    std::vector<Mapping> maps(ks);
    for (int micro=0; micro<count; micro+=16) {
        const int size=std::min(16,count-micro);
        for (int ky=0; ky<p.kh; ++ky) for (int kx=0; kx<p.kw; ++kx) {
            auto& m=maps[ky*p.kw+kx];
            for (int j=0; j<16; ++j) {
                int index=pos+micro+j,sy=index/ow*p.sy-p.py+ky,sx=index%ow*p.sx-p.px+kx;
                m.offsets[j]=j<size && sy>=0 && sy<ih && sx>=0 && sx<iw ? sy*iw+sx : -1;
            }
            m.fast=size==16 && m.offsets[0]>=0 && m.offsets[15]==m.offsets[0]+15*p.sx
                   && (pos+micro)/ow==(pos+micro+15)/ow ? p.sx : 0;
            // even8 loads one unused odd lane after the final selected lane.
            if (m.fast==2 && m.offsets[15]+1>=area) m.fast=0;
        }
        uint8_t* panel=dst+(micro/16)*k4*64;
        __m256i qlo[4],qhi[4];
        for (int r=0; r<4; ++r) qlo[r]=qhi[r]=_mm256_set1_epi32(zp);
        int z=0;
        for (int c=0; c<x.shape[1]; ++c) for (int s=0; s<ks; ++s,++z) {
            const auto& m=maps[s];
            const float* channel=x.data+c*area;
            __m256 lo,hi;
            if (m.fast==1) {
                lo=_mm256_loadu_ps(channel+m.offsets[0]); hi=_mm256_loadu_ps(channel+m.offsets[0]+8);
            } else if (m.fast==2) {
                lo=even8(channel+m.offsets[0]); hi=even8(channel+m.offsets[0]+16);
            } else {
                alignas(32) float values[16];
                for (int j=0; j<16; ++j) values[j]=m.offsets[j]>=0 ? channel[m.offsets[j]] : 0.f;
                lo=_mm256_load_ps(values); hi=_mm256_load_ps(values+8);
            }
            qlo[z%4]=quantize(lo,inverse,zp); qhi[z%4]=quantize(hi,inverse,zp);
            if (z%4==3 || z==k-1) {
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(panel+z/4*64),interleave(qlo[0],qlo[1],qlo[2],qlo[3]));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(panel+z/4*64+32),interleave(qhi[0],qhi[1],qhi[2],qhi[3]));
                // Only necessary for K%4 tails in the last group.
                if (z==k-1) break;
                if (k-z<=4) for (int r=0; r<4; ++r) qlo[r]=qhi[r]=_mm256_set1_epi32(zp);
            }
        }
    }
}
void dot_int8(const int8_t* a, const uint8_t* b, int k4, int32_t* output) {
    microkernel(a,b,k4,[&](const __m256i acc[6][2]) {
        for (int r=0; r<6; ++r) for (int h=0; h<2; ++h)
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(output+r*16+h*8),acc[r][h]);
    });
}
void dot_int8_ref(const int8_t* a, const uint8_t* b, int k4, int32_t* output) {
    for (int r=0; r<6; ++r) for (int j=0; j<16; ++j) {
        int32_t acc=0;
        for (int k=0; k<k4; ++k) for (int t=0; t<4; ++t)
            acc+=int32_t(a[(k*6+r)*4+t])*b[(k*16+j)*4+t];
        output[r*16+j]=acc;
    }
}
void conv_int8(Context& ctx, const Tensor& x, Tensor& y, ConvParams p, QuantConv q, const Tensor* residual) {
    if (p.group!=1) throw std::runtime_error("INT8 only supports dense convolution");
    const int n=y.shape[2]*y.shape[3],oc=y.shape[1],k4=(x.shape[1]*p.kh*p.kw+3)/4;
    const int nb=p.spatial_tile,chunk=p.output_chunk ? p.output_chunk : (oc+5)/6*6;
    if (nb<16 || nb%16 || chunk<6 || chunk%6) throw std::runtime_error("invalid INT8 grid");
    if (ctx.scratch[0].size()*sizeof(float)<static_cast<size_t>(k4*4*nb))
        throw std::runtime_error("INT8 scratch too small");
    const int tiles=(n+nb-1)/nb,groups=(oc+chunk-1)/chunk;
    // Per-worker counters avoid races; report sum(worker time)/threads, not an
    // additive wall-clock claim (load imbalance/dispatch remain in conv total).
    std::vector<double> pack_times(ctx.profiling ? ctx.scratch.size() : 0),gemm_times(pack_times.size());
    ctx.pool.parallel(tiles*groups,[&](size_t job,int worker) {
        const int pos=int(job%tiles)*nb,count=std::min(nb,n-pos),first=int(job/tiles)*chunk,end=std::min(oc,first+chunk);
        auto* panel=reinterpret_cast<uint8_t*>(ctx.scratch[worker].data());
        auto start=ctx.profiling ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        pack_int8(x,y,p,pos,count,q.activation_scale,q.zero_point,panel);
        if (ctx.profiling) {
            auto now=std::chrono::steady_clock::now();
            pack_times[worker]+=std::chrono::duration<double,std::milli>(now-start).count(); start=now;
        }
        for (int j=0; j<count; j+=16) for (int o=first; o<end; o+=6) {
            microkernel(q.weights+o/6*k4*24,panel+j/16*k4*64,k4,[&](const __m256i acc[6][2]) {
                for (int r=0; r<std::min(6,oc-o); ++r) for (int h=0; h<2 && h*8<count-j; ++h) {
                    int valid=std::min(8,count-j-h*8);
                    const __m256i mask=_mm256_cmpgt_epi32(_mm256_set1_epi32(valid),_mm256_setr_epi32(0,1,2,3,4,5,6,7));
                    // Deliberately separate multiply/add: reference epilogue
                    // uses the same rounded product and folded correction.
                    __m256 v=_mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(acc[r][h]),_mm256_set1_ps(q.products[o+r])),
                                          _mm256_set1_ps(q.corrected_bias[o+r]));
                    if (p.silu) v=silu8(v);
                    int index=(o+r)*n+pos+j+h*8;
                    if (residual) v=_mm256_add_ps(v,valid==8 ? _mm256_loadu_ps(residual->data+index) : _mm256_maskload_ps(residual->data+index,mask));
                    if (valid==8) _mm256_storeu_ps(y.data+index,v); else _mm256_maskstore_ps(y.data+index,mask,v);
                }
            });
        }
        if (ctx.profiling) gemm_times[worker]+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    });
    if (ctx.profiling) for (size_t w=0; w<pack_times.size(); ++w) {
        ctx.profile["int8_pack_quantize"]+=pack_times[w]/pack_times.size();
        ctx.profile["int8_gemm_epilogue"]+=gemm_times[w]/pack_times.size();
    }
}
}
