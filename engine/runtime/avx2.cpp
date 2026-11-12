#include "runtime.h"
#include "exp_avx2.h"
#include <immintrin.h>
#include <limits>

namespace engine {
void conv_opt(Context& ctx, const Tensor& x, const Tensor& w, const Tensor& b, Tensor& y, ConvParams p, const float* packed, const Tensor* residual) {
    if (p.group == x.shape[1] && y.shape[1] == x.shape[1]) {
        conv_depthwise(ctx,x,w,b,y,p,residual);
        return;
    }
    if (p.group != 1 || p.kh*p.kw>9) { conv_ref(x,w,b,y,p,residual); return; }
    conv_dense(ctx,x,b,y,p,packed,residual);
}
void maxpool_opt(const Tensor& x, Tensor& y) {
    int h = x.shape[2], w = x.shape[3];
    for (int c = 0; c < x.shape[1]; ++c)
        for (int row = 0; row < h; ++row) {
            int col = 0;
            while (col < w) {
                if (col >= 2 && col+9 < w) {
                    __m256 sum = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
                    for (int dy = -2; dy <= 2; ++dy) if (row+dy >= 0 && row+dy < h)
                        for (int dx = -2; dx <= 2; ++dx)
                            sum = _mm256_max_ps(sum, _mm256_loadu_ps(x.data+(c*h+row+dy)*w+col+dx));
                    _mm256_storeu_ps(y.data+(c*h+row)*w+col, sum); col += 8;
                } else {
                    float sum = -std::numeric_limits<float>::infinity();
                    for (int dy = -2; dy <= 2; ++dy) if (row+dy >= 0 && row+dy < h)
                        for (int dx = -2; dx <= 2; ++dx) if (col+dx >= 0 && col+dx < w)
                            sum = std::max(sum, x.data[(c*h+row+dy)*w+col+dx]);
                    y.data[(c*h+row)*w+col++] = sum;
                }
            }
        }
}
void resize_opt(const Tensor& x, Tensor& y) {
    int h = x.shape[2], w = x.shape[3];
    for (int c = 0; c < x.shape[1]; ++c)
        for (int row = 0; row < h; ++row) {
            const float* src = x.data+(c*h+row)*w;
            float* dst = y.data+(c*2*h+row*2)*w*2;
            int col = 0;
            for (; col+8 <= w; col += 8) {
                __m256 v = _mm256_loadu_ps(src+col);
                __m256 lo = _mm256_unpacklo_ps(v,v), hi = _mm256_unpackhi_ps(v,v);
                _mm256_storeu_ps(dst+2*col, _mm256_permute2f128_ps(lo,hi,0x20));
                _mm256_storeu_ps(dst+2*col+8, _mm256_permute2f128_ps(lo,hi,0x31));
            }
            for (; col < w; ++col) dst[2*col] = dst[2*col+1] = src[col];
            std::memcpy(dst+w*2, dst, w*2*sizeof(float));
        }
}
void unary_opt(const Tensor& x, Tensor& y, int activation) {
    const size_t size = x.size();
    size_t i = 0;
    for (; i+8 <= size; i += 8) {
        __m256 v = _mm256_loadu_ps(x.data+i);
        _mm256_storeu_ps(y.data+i, activation ? silu8(v) : sigmoid8(v));
    }
    for (; i < size; ++i) y.data[i] = activation ? silu(x.data[i]) : sigmoid(x.data[i]);
}
void softmax_opt(Context& ctx, const Tensor& x, Tensor& y, int axis) {
    size_t inner = 1, outer = 1;
    for (int i = 0; i < axis; ++i) outer *= x.shape[i];
    for (size_t i = axis+1; i < x.shape.size(); ++i) inner *= x.shape[i];
    const int n = x.shape[axis];
    const size_t tiles = (inner+127)/128;
    ctx.pool.parallel(outer*tiles, [&](size_t job, int) {
        const size_t o = job/tiles, start = (job%tiles)*128, end = std::min(inner,start+128);
        size_t i = start;
        // DFL: eight independent reductions in the spatial dimension.
        for (; i+8 <= end; i += 8) {
            size_t base = o*n*inner+i;
            __m256 maximum = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
            for (int c = 0; c < n; ++c) maximum = _mm256_max_ps(maximum,_mm256_loadu_ps(x.data+base+c*inner));
            __m256 sum = _mm256_setzero_ps();
            for (int c = 0; c < n; ++c) {
                __m256 e = exp8(_mm256_sub_ps(_mm256_loadu_ps(x.data+base+c*inner),maximum));
                _mm256_storeu_ps(y.data+base+c*inner,e); sum = _mm256_add_ps(sum,e);
            }
            for (int c = 0; c < n; ++c) _mm256_storeu_ps(y.data+base+c*inner,_mm256_div_ps(_mm256_loadu_ps(y.data+base+c*inner),sum));
        }
        for (; i < end; ++i) {
            size_t base = o*n*inner+i;
            float maximum = -std::numeric_limits<float>::infinity(), sum = 0;
            for (int c = 0; c < n; ++c) maximum = std::max(maximum,x.data[base+c*inner]);
            int c = 0;
            if (inner == 1) {
                for (; c+8 <= n; c += 8) {
                    __m256 e = exp8(_mm256_sub_ps(_mm256_loadu_ps(x.data+base+c),_mm256_set1_ps(maximum)));
                    _mm256_storeu_ps(y.data+base+c,e);
                }
                // Keep the reference reduction order (only exp is vectorized).
                for (int z = 0; z < c; ++z) sum += y.data[base+z];
            }
            for (; c < n; ++c) { float e = std::exp(x.data[base+c*inner]-maximum); y.data[base+c*inner] = e; sum += e; }
            for (c = 0; c < n; ++c) y.data[base+c*inner] /= sum;
        }
    });
}
void matmul_tile(const Tensor& a,const Tensor& b,Tensor& y,int batch,int row,int rows) {
    const int m=a.shape[2],k=a.shape[3],n=b.shape[3];
    const float* aa=a.data+(batch*m+row)*k;
    const float* bb=b.data+batch*k*n;
    float* out=y.data+(batch*m+row)*n;
    int j=0;
    if (rows==4) for (; j+16<=n; j+=16) {
        __m256 c00=_mm256_setzero_ps(),c01=c00,c10=c00,c11=c00,c20=c00,c21=c00,c30=c00,c31=c00;
        for (int z=0; z<k; ++z) {
            __m256 b0=_mm256_loadu_ps(bb+z*n+j),b1=_mm256_loadu_ps(bb+z*n+j+8);
            __m256 av=_mm256_set1_ps(aa[z]);
            c00=_mm256_fmadd_ps(av,b0,c00); c01=_mm256_fmadd_ps(av,b1,c01);
            av=_mm256_set1_ps(aa[k+z]);
            c10=_mm256_fmadd_ps(av,b0,c10); c11=_mm256_fmadd_ps(av,b1,c11);
            av=_mm256_set1_ps(aa[2*k+z]);
            c20=_mm256_fmadd_ps(av,b0,c20); c21=_mm256_fmadd_ps(av,b1,c21);
            av=_mm256_set1_ps(aa[3*k+z]);
            c30=_mm256_fmadd_ps(av,b0,c30); c31=_mm256_fmadd_ps(av,b1,c31);
        }
        _mm256_storeu_ps(out+j,c00); _mm256_storeu_ps(out+j+8,c01);
        _mm256_storeu_ps(out+n+j,c10); _mm256_storeu_ps(out+n+j+8,c11);
        _mm256_storeu_ps(out+2*n+j,c20); _mm256_storeu_ps(out+2*n+j+8,c21);
        _mm256_storeu_ps(out+3*n+j,c30); _mm256_storeu_ps(out+3*n+j+8,c31);
    }
    for (int r=0; r<rows; ++r) {
        int col=j;
        for (; col+8<=n; col+=8) {
            __m256 sum=_mm256_setzero_ps();
            for (int z=0; z<k; ++z) sum=_mm256_fmadd_ps(_mm256_set1_ps(aa[r*k+z]),_mm256_loadu_ps(bb+z*n+col),sum);
            _mm256_storeu_ps(out+r*n+col,sum);
        }
        for (; col<n; ++col) {
            float sum=0;
            for (int z=0; z<k; ++z) sum=std::fma(aa[r*k+z],bb[z*n+col],sum);
            out[r*n+col]=sum;
        }
    }
}
}
