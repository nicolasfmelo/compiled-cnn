#include "runtime.h"
#include "exp_avx2.h"
#include <immintrin.h>

namespace engine {
void conv_depthwise(Context& ctx,const Tensor& x,const Tensor& w,const Tensor& b,Tensor& y,ConvParams p,const Tensor* residual) {
    if (p.sx!=1 && p.sx!=2) { conv_ref(x,w,b,y,p,residual); return; }
    const int ih=x.shape[2],iw=x.shape[3],oh=y.shape[2],ow=y.shape[3],channels=y.shape[1];
    const int block=channels<4*ctx.pool.size() ? 16 : oh, blocks=(oh+block-1)/block;
    // A zero halo makes even tiny 7x7 maps vectorizable at every column.
    // Extra right guard covers unused odd lanes in the stride-2 loads.
    const int pitch=(std::max(iw+p.px,(ow-1)*p.sx+p.kw)+23)/8*8;
    const int rows=(std::min(block,oh)-1)*p.sy+p.kh;
    if (ctx.scratch[0].size()<static_cast<size_t>(pitch*rows)) throw std::runtime_error("depthwise halo scratch too small");
    ctx.pool.parallel(channels*blocks,[&](size_t job,int worker) {
        const int c=static_cast<int>(job)/blocks,first=static_cast<int>(job)%blocks*block,end=std::min(oh,first+block);
        const int height=(end-first-1)*p.sy+p.kh,origin=first*p.sy-p.py;
        float* halo=ctx.scratch[worker].data();
        std::fill(halo,halo+height*pitch,0.f);
        for (int row=0; row<height; ++row) {
            int sy=origin+row;
            if (sy>=0 && sy<ih) std::memcpy(halo+row*pitch+p.px,x.data+(c*ih+sy)*iw,iw*sizeof(float));
        }
        for (int row=first; row<end; ++row)
            for (int col=0; col<ow; col+=8) {
                __m256 sum=_mm256_setzero_ps();
                for (int ky=0; ky<p.kh; ++ky)
                    for (int kx=0; kx<p.kw; ++kx) {
                        const float* src=halo+((row-first)*p.sy+ky)*pitch+col*p.sx+kx;
                        __m256 v;
                        if (p.sx==1) v=_mm256_loadu_ps(src);
                        else {
                            v=_mm256_shuffle_ps(_mm256_loadu_ps(src),_mm256_loadu_ps(src+8),0x88);
                            v=_mm256_permutevar8x32_ps(v,_mm256_setr_epi32(0,1,4,5,2,3,6,7));
                        }
                        sum=_mm256_fmadd_ps(v,_mm256_set1_ps(w.data[(c*p.kh+ky)*p.kw+kx]),sum);
                    }
                sum=_mm256_add_ps(sum,_mm256_set1_ps(b.data[c]));
                if (p.silu) sum=silu8(sum);
                size_t offset=(c*oh+row)*ow+col;
                const __m256i mask=_mm256_cmpgt_epi32(_mm256_set1_epi32(std::min(8,ow-col)),_mm256_setr_epi32(0,1,2,3,4,5,6,7));
                if (residual) sum=_mm256_add_ps(sum,_mm256_maskload_ps(residual->data+offset,mask));
                _mm256_maskstore_ps(y.data+offset,mask,sum);
            }
    });
}
}
