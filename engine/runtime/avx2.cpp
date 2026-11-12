#include "runtime.h"
#include <immintrin.h>
#include <limits>

namespace engine {
// Four output channels x sixteen spatial positions, eight independent FMA
// chains. Packed A is [oc/4][K][4], B is a spatial row-major panel. KB=128,
// NB=64 keep the hot B panel (32 KiB) plus weights/output below 48 KiB L1d;
// the complete im2col tile is at most 576 KiB, below the P-core's 1.25 MiB L2.
static void gemm4x16(const float* a, const float* b, float* y, int k,
                     int bs, int ys, bool first, bool last, const float* bias, bool activation) {
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
            if (last) v = _mm256_add_ps(v, _mm256_set1_ps(bias[r]));
            _mm256_storeu_ps(y+r*ys+j*8, v);
        }
        if (last && activation) for (int j = 0; j < 16; ++j) y[r*ys+j] = silu(y[r*ys+j]);
    }
}

static void depthwise(const Tensor& x, const Tensor& w, const Tensor& b, Tensor& y, ConvParams p) {
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
                    _mm256_storeu_ps(dst, _mm256_add_ps(sum, _mm256_set1_ps(b.data[c])));
                    if (p.silu) for (int j = 0; j < 8; ++j) dst[j] = silu(dst[j]);
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
                    y.data[(c*oh+row)*ow+col++] = p.silu ? silu(sum) : sum;
                }
            }
        }
}

void conv_opt(Context& ctx, const Tensor& x, const Tensor& w, const Tensor& b, Tensor& y, ConvParams p, const float* packed) {
    if (p.group == x.shape[1] && y.shape[1] == x.shape[1]) { depthwise(x,w,b,y,p); return; }
    if (p.group != 1) { conv_ref(x,w,b,y,p); return; }
    int ic = x.shape[1], ih = x.shape[2], iw = x.shape[3];
    int oc = y.shape[1], oh = y.shape[2], ow = y.shape[3], n = oh*ow, k = ic*p.kh*p.kw;
    constexpr int nb = 64, kb = 128;
    if (ctx.scratch.size() < static_cast<size_t>(k*nb)) throw std::runtime_error("im2col scratch too small");
    bool direct = p.kh == 1 && p.kw == 1 && p.sy == 1 && p.sx == 1 && p.py == 0 && p.px == 0;
    for (int pos = 0; pos < n; pos += nb) {
        int count = std::min(nb, n-pos), bs = direct ? n : nb;
        const float* panel = direct ? x.data+pos : ctx.scratch.data();
        if (!direct) {
            for (int c = 0; c < ic; ++c)
                for (int ky = 0; ky < p.kh; ++ky)
                    for (int kx = 0; kx < p.kw; ++kx) {
                        float* row = ctx.scratch.data()+((c*p.kh+ky)*p.kw+kx)*nb;
                        for (int j = 0; j < count; ++j) {
                            int sy = (pos+j)/ow*p.sy+ky-p.py, sx = (pos+j)%ow*p.sx+kx-p.px;
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
                              kc, bs, n, z == 0, z+kc == k, b.data+o, p.silu);
                for (int r = o; r < std::min(o+4, oc); ++r)
                    for (int col = j; col < count; ++col) {
                        float sum = z == 0 ? 0 : y.data[r*n+pos+col];
                        for (int t = 0; t < kc; ++t)
                            sum = std::fma(packed[(o/4*k+z+t)*4+r-o], panel[(z+t)*bs+col], sum);
                        if (z+kc == k) { sum += b.data[r]; if (p.silu) sum = silu(sum); }
                        y.data[r*n+pos+col] = sum;
                    }
            }
        }
    }
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
