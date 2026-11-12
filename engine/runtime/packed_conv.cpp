#include "runtime.h"
#include "exp_avx2.h"
#include <immintrin.h>

namespace engine {
namespace {
constexpr int MR = CONV_MR, NR = 16;
__m256i mask(int count) {
    return _mm256_cmpgt_epi32(_mm256_set1_epi32(count),_mm256_setr_epi32(0,1,2,3,4,5,6,7));
}
// Two contiguous loads, then even lanes in original order. The caller checks
// all sixteen floats are in bounds (the unused last odd lane is still loaded).
__m256 even8(const float* p) {
    __m256 v = _mm256_shuffle_ps(_mm256_loadu_ps(p),_mm256_loadu_ps(p+8),0x88);
    return _mm256_permutevar8x32_ps(v,_mm256_setr_epi32(0,1,4,5,2,3,6,7));
}
struct Segment { int begin, count, y, x; };
struct Run { int begin,left,right,vector_end,row,x; };
void pack_panel(const Tensor& x, const Tensor& y, ConvParams p, int pos, int count, float* dst) {
    const int ih=x.shape[2], iw=x.shape[3], ow=y.shape[3], k=x.shape[1]*p.kh*p.kw;
    const bool direct = p.kh==1 && p.kw==1 && p.sy==1 && p.sx==1 && p.py==0 && p.px==0 && ih==y.shape[2] && iw==ow;
    for (int micro=0; micro<count; micro+=NR) {
        const int size=std::min(NR,count-micro);
        float* panel=dst+micro*k;
        if (direct) {
            for (int c=0; c<x.shape[1]; ++c) {
                const float* src=x.data+c*ih*iw+pos+micro;
                if (size==NR) {
                    _mm256_storeu_ps(panel+c*NR,_mm256_loadu_ps(src));
                    _mm256_storeu_ps(panel+c*NR+8,_mm256_loadu_ps(src+8));
                } else {
                    _mm256_storeu_ps(panel+c*NR,_mm256_maskload_ps(src,mask(std::min(size,8))));
                    _mm256_storeu_ps(panel+c*NR+8,size>8 ? _mm256_maskload_ps(src+8,mask(size-8)) : _mm256_setzero_ps());
                }
            }
            continue;
        }
        Segment segments[NR];
        int ns=0;
        for (int j=0; j<size;) {
            const int index=pos+micro+j, row=index/ow, col=index-row*ow;
            const int length=std::min(size-j,ow-col);
            segments[ns++]={j,length,row*p.sy-p.py,col*p.sx-p.px};
            j+=length;
        }
        Run runs[9][NR];
        bool complete[9];
        // Bounds and divisions depend on the kernel offset, never the input
        // channel. Compute them once, then only copy runs in the channel loop.
        for (int ky=0; ky<p.kh; ++ky)
            for (int kx=0; kx<p.kw; ++kx) {
                int z=ky*p.kw+kx;
                complete[z]=size==NR;
                for (int s=0; s<ns; ++s) {
                    auto seg=segments[s];
                    const int sy=seg.y+ky,sx=seg.x+kx;
                    int left=std::max(0,(-sx+p.sx-1)/p.sx);
                    int right=sx>=iw ? 0 : std::min(seg.count,(iw-1-sx)/p.sx+1);
                    if (sy<0 || sy>=ih) left=right=0;
                    int vector_end=p.sx==2 ? std::min(right,(iw-sx)/2) : right;
                    runs[z][s]={seg.begin,left,right,vector_end,sy*iw,sx};
                    complete[z]=complete[z] && left==0 && right==seg.count;
                }
            }
        for (int c=0; c<x.shape[1]; ++c)
            for (int z=0; z<p.kh*p.kw; ++z) {
                float* out=panel+(c*p.kh*p.kw+z)*NR;
                if (!complete[z]) {
                    _mm256_storeu_ps(out,_mm256_setzero_ps());
                    _mm256_storeu_ps(out+8,_mm256_setzero_ps());
                }
                for (int s=0; s<ns; ++s) {
                    auto run=runs[z][s];
                    if (run.left>=run.right) continue;
                    const float* row=x.data+c*ih*iw+run.row;
                    int j=run.left;
                    for (; j+8<=run.vector_end; j+=8) {
                        const float* src=row+run.x+j*p.sx;
                        if (p.sx==1) _mm256_storeu_ps(out+run.begin+j,_mm256_loadu_ps(src));
                        else if (p.sx==2) _mm256_storeu_ps(out+run.begin+j,even8(src));
                        else break;
                    }
                    for (; j<run.right; ++j) out[run.begin+j]=row[run.x+j*p.sx];
                }
            }
    }
}
template<bool Full>
void microkernel(const float* a, const float* b, float* y, int k, int ys, int rows, int cols,
                 const float* bias, bool activation, const float* residual) {
    __m256 c00=_mm256_setzero_ps(),c01=c00,c10=c00,c11=c00,c20=c00,c21=c00,c30=c00,c31=c00,c40=c00,c41=c00,c50=c00,c51=c00;
    for (int z=0; z<k; ++z) {
        const __m256 b0=_mm256_loadu_ps(b+z*NR),b1=_mm256_loadu_ps(b+z*NR+8);
        __m256 av=_mm256_broadcast_ss(a+z*MR);
        c00=_mm256_fmadd_ps(av,b0,c00); c01=_mm256_fmadd_ps(av,b1,c01);
        av=_mm256_broadcast_ss(a+z*MR+1);
        c10=_mm256_fmadd_ps(av,b0,c10); c11=_mm256_fmadd_ps(av,b1,c11);
        av=_mm256_broadcast_ss(a+z*MR+2);
        c20=_mm256_fmadd_ps(av,b0,c20); c21=_mm256_fmadd_ps(av,b1,c21);
        av=_mm256_broadcast_ss(a+z*MR+3);
        c30=_mm256_fmadd_ps(av,b0,c30); c31=_mm256_fmadd_ps(av,b1,c31);
        av=_mm256_broadcast_ss(a+z*MR+4);
        c40=_mm256_fmadd_ps(av,b0,c40); c41=_mm256_fmadd_ps(av,b1,c41);
        av=_mm256_broadcast_ss(a+z*MR+5);
        c50=_mm256_fmadd_ps(av,b0,c50); c51=_mm256_fmadd_ps(av,b1,c51);
    }
    const __m256 acc[MR][2]={{c00,c01},{c10,c11},{c20,c21},{c30,c31},{c40,c41},{c50,c51}};
    for (int r=0; r<rows; ++r)
        for (int h=0; h<2 && (Full || h*8<cols); ++h) {
            const int valid=std::min(8,cols-h*8);
            const __m256i m=mask(valid);
            __m256 v=_mm256_add_ps(acc[r][h],_mm256_set1_ps(bias[r]));
            if (activation) v=silu8(v);
            if (residual) {
                if constexpr (Full) v=_mm256_add_ps(v,_mm256_loadu_ps(residual+r*ys+h*8));
                else v=_mm256_add_ps(v,_mm256_maskload_ps(residual+r*ys+h*8,m));
            }
            if constexpr (Full) _mm256_storeu_ps(y+r*ys+h*8,v);
            else _mm256_maskstore_ps(y+r*ys+h*8,m,v);
        }
}
}
void conv_dense(Context& ctx, const Tensor& x, const Tensor& b, Tensor& y, ConvParams p,
                const float* packed, const Tensor* residual) {
    const int n=y.shape[2]*y.shape[3],oc=y.shape[1],k=x.shape[1]*p.kh*p.kw;
    const int nb=p.spatial_tile, chunk=p.output_chunk ? p.output_chunk : (oc+MR-1)/MR*MR;
    if (nb<NR || nb%NR || chunk<MR || chunk%MR) throw std::runtime_error("invalid convolution grid");
    if (ctx.scratch[0].size()<static_cast<size_t>(k*nb)) throw std::runtime_error("packed panel scratch too small");
    const int tiles=(n+nb-1)/nb, groups=(oc+chunk-1)/chunk;
    ctx.pool.parallel(tiles*groups,[&](size_t job,int worker) {
        const int pos=static_cast<int>(job%tiles)*nb,count=std::min(nb,n-pos);
        const int first=static_cast<int>(job/tiles)*chunk,end=std::min(oc,first+chunk);
        float* panel=ctx.scratch[worker].data();
        pack_panel(x,y,p,pos,count,panel);
        for (int j=0; j<count; j+=NR)
            for (int o=first; o<end; o+=MR) {
                auto kernel=count-j>=NR ? microkernel<true> : microkernel<false>;
                kernel(packed+o/MR*k*MR,panel+j*k,y.data+o*n+pos+j,k,n,std::min(MR,oc-o),std::min(NR,count-j),
                       b.data+o,p.silu,residual ? residual->data+o*n+pos+j : nullptr);
            }
    });
}
}
