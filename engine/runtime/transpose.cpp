#include "runtime.h"
#include <immintrin.h>

namespace engine {
namespace {
void transpose8(const float* src,float* dst,int is,int os) {
    __m256 r0=_mm256_loadu_ps(src),r1=_mm256_loadu_ps(src+is),r2=_mm256_loadu_ps(src+2*is),r3=_mm256_loadu_ps(src+3*is);
    __m256 r4=_mm256_loadu_ps(src+4*is),r5=_mm256_loadu_ps(src+5*is),r6=_mm256_loadu_ps(src+6*is),r7=_mm256_loadu_ps(src+7*is);
    __m256 t0=_mm256_unpacklo_ps(r0,r1),t1=_mm256_unpackhi_ps(r0,r1),t2=_mm256_unpacklo_ps(r2,r3),t3=_mm256_unpackhi_ps(r2,r3);
    __m256 t4=_mm256_unpacklo_ps(r4,r5),t5=_mm256_unpackhi_ps(r4,r5),t6=_mm256_unpacklo_ps(r6,r7),t7=_mm256_unpackhi_ps(r6,r7);
    __m256 s0=_mm256_shuffle_ps(t0,t2,0x44),s1=_mm256_shuffle_ps(t0,t2,0xee),s2=_mm256_shuffle_ps(t1,t3,0x44),s3=_mm256_shuffle_ps(t1,t3,0xee);
    __m256 s4=_mm256_shuffle_ps(t4,t6,0x44),s5=_mm256_shuffle_ps(t4,t6,0xee),s6=_mm256_shuffle_ps(t5,t7,0x44),s7=_mm256_shuffle_ps(t5,t7,0xee);
    _mm256_storeu_ps(dst,_mm256_permute2f128_ps(s0,s4,0x20));
    _mm256_storeu_ps(dst+os,_mm256_permute2f128_ps(s1,s5,0x20));
    _mm256_storeu_ps(dst+2*os,_mm256_permute2f128_ps(s2,s6,0x20));
    _mm256_storeu_ps(dst+3*os,_mm256_permute2f128_ps(s3,s7,0x20));
    _mm256_storeu_ps(dst+4*os,_mm256_permute2f128_ps(s0,s4,0x31));
    _mm256_storeu_ps(dst+5*os,_mm256_permute2f128_ps(s1,s5,0x31));
    _mm256_storeu_ps(dst+6*os,_mm256_permute2f128_ps(s2,s6,0x31));
    _mm256_storeu_ps(dst+7*os,_mm256_permute2f128_ps(s3,s7,0x31));
}
}
bool transpose_opt(Context& ctx,const Tensor& x,Tensor& y,std::initializer_list<int> permutation) {
    const std::vector<int> perm(permutation);
    int rank=static_cast<int>(perm.size());
    bool swap=rank>=2 && perm[rank-2]==rank-1 && perm[rank-1]==rank-2;
    for (int i=0; i<rank-2; ++i) swap=swap && perm[i]==i;
    if (swap) {
        const int m=x.shape[rank-2],n=x.shape[rank-1];
        const int mt=(m+31)/32,nt=(n+31)/32,batches=static_cast<int>(x.size()/(m*n));
        ctx.pool.parallel(batches*mt*nt,[&](size_t job,int) {
            int batch=static_cast<int>(job)/(mt*nt),ri=static_cast<int>(job)/nt%mt*32,cj=static_cast<int>(job)%nt*32;
            int re=std::min(m,ri+32),ce=std::min(n,cj+32);
            const float* src=x.data+batch*m*n;
            float* dst=y.data+batch*m*n;
            for (int i=ri; i<re; i+=8)
                for (int j=cj; j<ce; j+=8) {
                    if (i+8<=re && j+8<=ce) transpose8(src+i*n+j,dst+j*m+i,n,m);
                    else for (int r=i; r<std::min(re,i+8); ++r)
                        for (int c=j; c<std::min(ce,j+8); ++c) dst[c*m+r]=src[r*n+c];
                }
        });
        return true;
    }
    if (rank>0 && perm.back()==rank-1) {
        std::vector<size_t> xs(rank,1),ys(rank,1);
        for (int i=rank-2; i>=0; --i) { xs[i]=xs[i+1]*x.shape[i+1]; ys[i]=ys[i+1]*y.shape[i+1]; }
        const size_t width=y.shape.back();
        ctx.pool.parallel(y.size()/width,[&](size_t row,int) {
            size_t offset=0,index=row*width;
            for (int d=0; d<rank-1; ++d) offset+=(index/ys[d]%y.shape[d])*xs[perm[d]];
            std::memcpy(y.data+index,x.data+offset,width*sizeof(float));
        });
        return true;
    }
    return false;
}
}
