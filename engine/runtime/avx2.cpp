#include "runtime.h"
#include "exp_avx2.h"
#include <immintrin.h>
#include <limits>

namespace engine {
// Four output channels x sixteen spatial positions, eight independent FMA
// chains. Packed A is [oc/4][K][4], B is a spatial row-major panel. KB=128,
// NB=64 keep the hot B panel (32 KiB) plus weights/output below 48 KiB L1d;
// the complete im2col tile is at most 1.125 MiB for this graph, below the
// P-core's 1.25 MiB L2.
static void gemm4x16(const float* a, const float* b, float* y, int k,
                     int bs, int ys, bool first, bool last, const float* bias, bool activation, const float* residual) {
    __m256 c00 = first ? _mm256_setzero_ps() : _mm256_loadu_ps(y);
    __m256 c01 = first ? _mm256_setzero_ps() : _mm256_loadu_ps(y+8);
    __m256 c10 = first ? _mm256_setzero_ps() : _mm256_loadu_ps(y+ys);
    __m256 c11 = first ? _mm256_setzero_ps() : _mm256_loadu_ps(y+ys+8);
    __m256 c20 = first ? _mm256_setzero_ps() : _mm256_loadu_ps(y+2*ys);
    __m256 c21 = first ? _mm256_setzero_ps() : _mm256_loadu_ps(y+2*ys+8);
    __m256 c30 = first ? _mm256_setzero_ps() : _mm256_loadu_ps(y+3*ys);
    __m256 c31 = first ? _mm256_setzero_ps() : _mm256_loadu_ps(y+3*ys+8);
    for (int z = 0; z < k; ++z) {
        __m256 b0 = _mm256_loadu_ps(b+z*bs), b1 = _mm256_loadu_ps(b+z*bs+8);
        __m256 a0 = _mm256_broadcast_ss(a+z*4), a1 = _mm256_broadcast_ss(a+z*4+1);
        __m256 a2 = _mm256_broadcast_ss(a+z*4+2), a3 = _mm256_broadcast_ss(a+z*4+3);
        c00 = _mm256_fmadd_ps(a0, b0, c00); c01 = _mm256_fmadd_ps(a0, b1, c01);
        c10 = _mm256_fmadd_ps(a1, b0, c10); c11 = _mm256_fmadd_ps(a1, b1, c11);
        c20 = _mm256_fmadd_ps(a2, b0, c20); c21 = _mm256_fmadd_ps(a2, b1, c21);
        c30 = _mm256_fmadd_ps(a3, b0, c30); c31 = _mm256_fmadd_ps(a3, b1, c31);
    }
    __m256 rows[4][2] = {{c00,c01},{c10,c11},{c20,c21},{c30,c31}};
    for (int r = 0; r < 4; ++r) {
        for (int j = 0; j < 2; ++j) {
            __m256 v = rows[r][j];
            if (last) {
                v = _mm256_add_ps(v, _mm256_set1_ps(bias[r]));
                if (activation) v = silu8(v);
                if (residual) v = _mm256_add_ps(v, _mm256_loadu_ps(residual+r*ys+j*8));
            }
            _mm256_storeu_ps(y+r*ys+j*8, v);
        }
    }
}

static void depthwise(const Tensor& x, const Tensor& w, const Tensor& b, Tensor& y, ConvParams p, const Tensor* residual) {
    int ih = x.shape[2], iw = x.shape[3], oh = y.shape[2], ow = y.shape[3];
    const __m256i indices = _mm256_setr_epi32(0,1,2,3,4,5,6,7);
    for (int c = 0; c < y.shape[1]; ++c)
        for (int row = 0; row < oh; ++row) {
            int col = 0;
            while (col < ow) {
                bool vector = col* p.sx >= p.px && (col+7)*p.sx+p.kw-p.px <= iw && col+8 <= ow;
                if (vector) {
                    __m256 sum = _mm256_setzero_ps();
                    for (int ky = 0; ky < p.kh; ++ky) {
                        int sy = row*p.sy+ky-p.py;
                        if (sy < 0 || sy >= ih) continue;
                        for (int kx = 0; kx < p.kw; ++kx) {
                            const float* src = x.data+(c*ih+sy)*iw+col*p.sx+kx-p.px;
                            __m256 v = p.sx == 1 ? _mm256_loadu_ps(src) : _mm256_i32gather_ps(src, _mm256_mullo_epi32(indices, _mm256_set1_epi32(p.sx)), 4);
                            sum = _mm256_fmadd_ps(v, _mm256_set1_ps(w.data[(c*p.kh+ky)*p.kw+kx]), sum);
                        }
                    }
                    float* dst = y.data+(c*oh+row)*ow+col;
                    sum = _mm256_add_ps(sum, _mm256_set1_ps(b.data[c]));
                    if (p.silu) sum = silu8(sum);
                    if (residual) sum = _mm256_add_ps(sum, _mm256_loadu_ps(residual->data+(dst-y.data)));
                    _mm256_storeu_ps(dst, sum);
                    col += 8;
                } else {
                    float sum = 0;
                    for (int ky = 0; ky < p.kh; ++ky)
                        for (int kx = 0; kx < p.kw; ++kx) {
                            int sy = row*p.sy+ky-p.py, sx = col*p.sx+kx-p.px;
                            if (sy >= 0 && sy < ih && sx >= 0 && sx < iw)
                                sum = std::fma(x.data[(c*ih+sy)*iw+sx], w.data[(c*p.kh+ky)*p.kw+kx], sum);
                        }
                    sum += b.data[c];
                    size_t dst = (c*oh+row)*ow+col++;
                    sum = p.silu ? silu(sum) : sum;
                    y.data[dst] = residual ? sum+residual->data[dst] : sum;
                }
            }
        }
}

void conv_opt(Context& ctx, const Tensor& x, const Tensor& w, const Tensor& b, Tensor& y, ConvParams p, const float* packed, const Tensor* residual) {
    if (p.group == x.shape[1] && y.shape[1] == x.shape[1]) {
        int ih = x.shape[2], iw = x.shape[3], oh = y.shape[2], ow = y.shape[3];
        ctx.pool.parallel(y.shape[1], [&](size_t c, int) {
            Tensor xx{x.data+c*ih*iw,{1,1,ih,iw}}, ww{w.data+c*p.kh*p.kw,{1,1,p.kh,p.kw}}, bb{b.data+c,{1}}, yy{y.data+c*oh*ow,{1,1,oh,ow}};
            Tensor rr{residual ? residual->data+c*oh*ow : nullptr, yy.shape};
            auto q = p; q.group = 1;
            depthwise(xx,ww,bb,yy,q,residual ? &rr : nullptr);
        });
        return;
    }
    if (p.group != 1) { conv_ref(x,w,b,y,p,residual); return; }
    int ic = x.shape[1], ih = x.shape[2], iw = x.shape[3];
    int oc = y.shape[1], oh = y.shape[2], ow = y.shape[3], n = oh*ow, k = ic*p.kh*p.kw;
    constexpr int nb = 64, kb = 128;
    if (ctx.scratch[0].size() < static_cast<size_t>(k*nb)) throw std::runtime_error("im2col scratch too small");
    bool direct = p.kh == 1 && p.kw == 1 && p.sy == 1 && p.sx == 1 && p.py == 0 && p.px == 0 && ih == oh && iw == ow;
    ctx.pool.parallel((n+nb-1)/nb, [&](size_t job, int worker) {
        int pos = static_cast<int>(job)*nb;
        float* scratch = ctx.scratch[worker].data();
        int count = std::min(nb, n-pos), bs = direct ? n : nb;
        const float* panel = direct ? x.data+pos : scratch;
        if (!direct) {
            int base_y[nb], base_x[nb];
            for (int j = 0; j < count; ++j) { base_y[j] = (pos+j)/ow*p.sy-p.py; base_x[j] = (pos+j)%ow*p.sx-p.px; }
            for (int c = 0; c < ic; ++c)
                for (int ky = 0; ky < p.kh; ++ky)
                    for (int kx = 0; kx < p.kw; ++kx) {
                        float* row = scratch+((c*p.kh+ky)*p.kw+kx)*nb;
                        for (int j = 0; j < count; ++j) {
                            int sy = base_y[j]+ky, sx = base_x[j]+kx;
                            row[j] = sy >= 0 && sy < ih && sx >= 0 && sx < iw ? x.data[(c*ih+sy)*iw+sx] : 0;
                        }
                    }
        }
        for (int z = 0; z < k; z += kb) {
            int kc = std::min(kb, k-z);
            for (int o = 0; o < oc; o += 4) {
                int j = 0;
                if (o+4 <= oc) for (; j+16 <= count; j += 16)
                    gemm4x16(packed+(o/4*k+z)*4, panel+z*bs+j, y.data+o*n+pos+j,
                              kc, bs, n, z == 0, z+kc == k, b.data+o, p.silu, residual ? residual->data+o*n+pos+j : nullptr);
                for (int r = o; r < std::min(o+4, oc); ++r)
                    for (int col = j; col < count; ++col) {
                        float sum = z == 0 ? 0 : y.data[r*n+pos+col];
                        for (int t = 0; t < kc; ++t)
                            sum = std::fma(packed[(o/4*k+z+t)*4+r-o], panel[(z+t)*bs+col], sum);
                        if (z+kc == k) {
                            sum += b.data[r]; if (p.silu) sum = silu(sum);
                            if (residual) sum += residual->data[r*n+pos+col];
                        }
                        y.data[r*n+pos+col] = sum;
                    }
            }
        }
    });
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
void matmul_opt(const Tensor& a, const Tensor& b, Tensor& y) {
    int m = a.shape[2], k = a.shape[3], n = b.shape[3];
    for (int batch = 0; batch < a.shape[0]*a.shape[1]; ++batch)
        for (int i = 0; i < m; ++i) {
            int j = 0;
            for (; j+8 <= n; j += 8) {
                __m256 sum = _mm256_setzero_ps();
                for (int z = 0; z < k; ++z)
                    sum = _mm256_fmadd_ps(_mm256_set1_ps(a.data[(batch*m+i)*k+z]), _mm256_loadu_ps(b.data+(batch*k+z)*n+j), sum);
                _mm256_storeu_ps(y.data+(batch*m+i)*n+j, sum);
            }
            for (; j < n; ++j) {
                float sum = 0;
                for (int z = 0; z < k; ++z) sum = std::fma(a.data[(batch*m+i)*k+z], b.data[(batch*k+z)*n+j], sum);
                y.data[(batch*m+i)*n+j] = sum;
            }
        }
}
}
