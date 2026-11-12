#include "runtime.h"
#include <limits>

namespace engine {
void conv_ref(const Tensor& x, const Tensor& w, const Tensor& b, Tensor& y, ConvParams p, const Tensor* residual) {
    const int ic = x.shape[1], ih = x.shape[2], iw = x.shape[3];
    const int oc = y.shape[1], oh = y.shape[2], ow = y.shape[3];
    const int cg = ic/p.group, og = oc/p.group;
    for (int o = 0; o < oc; ++o)
        for (int h = 0; h < oh; ++h)
            for (int col = 0; col < ow; ++col) {
                float sum = 0;
                for (int c = 0; c < cg; ++c)
                    for (int ky = 0; ky < p.kh; ++ky)
                        for (int kx = 0; kx < p.kw; ++kx) {
                            int sy = h*p.sy+ky-p.py, sx = col*p.sx+kx-p.px;
                            if (sy >= 0 && sy < ih && sx >= 0 && sx < iw)
                                sum += x.data[((o/og*cg+c)*ih+sy)*iw+sx] * w.data[((o*cg+c)*p.kh+ky)*p.kw+kx];
                        }
                sum += b.data[o];
                const size_t dst = (o*oh+h)*ow+col;
                sum = p.silu ? silu(sum) : sum;
                y.data[dst] = residual ? sum+residual->data[dst] : sum;
            }
}
void conv(Context& ctx, const Tensor& x, const Tensor& w, const Tensor& b, Tensor& y, ConvParams p, const float* packed, const Tensor* residual) {
    if (ctx.optimized) conv_opt(ctx, x, w, b, y, p, packed, residual);
    else conv_ref(x, w, b, y, p, residual);
}
void unary(Context& ctx, const Tensor& x, Tensor& y, int activation) {
    if (ctx.optimized) {
        ctx.ranges(x.size(), [&](size_t begin, size_t end) {
            Tensor xx{x.data+begin,{static_cast<int>(end-begin)}}, yy{y.data+begin,xx.shape};
            unary_opt(xx,yy,activation);
        });
        return;
    }
    const size_t size = x.size();
    for (size_t i = 0; i < size; ++i) y.data[i] = activation ? silu(x.data[i]) : sigmoid(x.data[i]);
}
static std::vector<size_t> strides(const std::vector<int>& shape) {
    std::vector<size_t> result(shape.size(), 1);
    for (int i = static_cast<int>(shape.size())-2; i >= 0; --i) result[i] = result[i+1]*shape[i+1];
    return result;
}
void binary(Context& ctx, const Tensor& a, const Tensor& b, Tensor& y, int op) {
    if (ctx.optimized && (a.size() == y.size() || a.size() == 1) && (b.size() == y.size() || b.size() == 1)) {
        const bool sa = a.size() == 1, sb = b.size() == 1;
        ctx.ranges(y.size(), [&](size_t begin, size_t end) {
            for (size_t i = begin; i < end; ++i) {
                float l = a.data[sa ? 0 : i], r = b.data[sb ? 0 : i];
                y.data[i] = op == 0 ? l+r : op == 1 ? l-r : op == 2 ? l*r : l/r;
            }
        });
        return;
    }
    auto as = strides(a.shape), bs = strides(b.shape), ys = strides(y.shape);
    size_t rank = y.shape.size(), da = rank-a.shape.size(), db = rank-b.shape.size();
    if (ctx.optimized) {
        const size_t width=y.shape.back();
        const bool sa=a.shape.back()==1, sb=b.shape.back()==1;
        // Coalesce the last axis. Decode broadcasting once per contiguous run,
        // not once per float per dimension (notably head strides [1,8400]).
        ctx.ranges(y.size(),[&](size_t begin,size_t end) {
            for (size_t i=begin; i<end;) {
                size_t ai=0,bi=0;
                for (size_t d=0; d+1<rank; ++d) {
                    size_t coordinate=i/ys[d]%y.shape[d];
                    if (d>=da && a.shape[d-da]!=1) ai+=coordinate*as[d-da];
                    if (d>=db && b.shape[d-db]!=1) bi+=coordinate*bs[d-db];
                }
                size_t col=i%width, count=std::min(end-i,width-col);
                for (size_t j=0; j<count; ++j) {
                    float l=a.data[ai+(sa ? 0 : col+j)], r=b.data[bi+(sb ? 0 : col+j)];
                    y.data[i+j]=op==0 ? l+r : op==1 ? l-r : op==2 ? l*r : l/r;
                }
                i+=count;
            }
        });
        return;
    }
    ctx.ranges(y.size(), [&](size_t begin, size_t end) {
      for (size_t i = begin; i < end; ++i) {
        size_t ai = 0, bi = 0;
        for (size_t d = 0; d < rank; ++d) {
            size_t coord = i/ys[d] % y.shape[d];
            if (d >= da && a.shape[d-da] != 1) ai += coord*as[d-da];
            if (d >= db && b.shape[d-db] != 1) bi += coord*bs[d-db];
        }
        float l = a.data[ai], r = b.data[bi];
        y.data[i] = op == 0 ? l+r : op == 1 ? l-r : op == 2 ? l*r : l/r;
      }
    });
}
void maxpool(Context& ctx, const Tensor& x, Tensor& y) {
    if (ctx.optimized) {
        int h = x.shape[2], w = x.shape[3];
        ctx.pool.parallel(x.shape[1], [&](size_t c, int) {
            Tensor xx{x.data+c*h*w,{1,1,h,w}}, yy{y.data+c*h*w,xx.shape};
            maxpool_opt(xx,yy);
        });
        return;
    }
    int c = x.shape[1], h = x.shape[2], w = x.shape[3];
    for (int ch = 0; ch < c; ++ch)
        for (int row = 0; row < h; ++row)
            for (int col = 0; col < w; ++col) {
                float v = -std::numeric_limits<float>::infinity();
                for (int dy = -2; dy <= 2; ++dy)
                    for (int dx = -2; dx <= 2; ++dx)
                        if (row+dy >= 0 && row+dy < h && col+dx >= 0 && col+dx < w)
                            v = std::max(v, x.data[(ch*h+row+dy)*w+col+dx]);
                y.data[(ch*h+row)*w+col] = v;
            }
}
void resize(Context& ctx, const Tensor& x, Tensor& y) {
    if (ctx.optimized) {
        int h = x.shape[2], w = x.shape[3];
        ctx.pool.parallel(x.shape[1], [&](size_t c, int) {
            Tensor xx{x.data+c*h*w,{1,1,h,w}}, yy{y.data+c*h*w*4,{1,1,h*2,w*2}};
            resize_opt(xx,yy);
        });
        return;
    }
    int h = x.shape[2], w = x.shape[3];
    for (int c = 0; c < x.shape[1]; ++c)
        for (int row = 0; row < h*2; ++row)
            for (int col = 0; col < w*2; ++col)
                y.data[(c*h*2+row)*w*2+col] = x.data[(c*h+row/2)*w+col/2];
}
void concat(Context&, std::initializer_list<const Tensor*> inputs, Tensor& y, int axis) {
    size_t outer = 1, inner = 1;
    for (int i = 0; i < axis; ++i) outer *= y.shape[i];
    for (size_t i = axis+1; i < y.shape.size(); ++i) inner *= y.shape[i];
    for (size_t o = 0; o < outer; ++o) {
        size_t offset = o*y.shape[axis]*inner;
        for (auto x : inputs) {
            size_t n = x->shape[axis]*inner;
            std::memcpy(y.data+offset, x->data+o*n, n*sizeof(float));
            offset += n;
        }
    }
}
void split(Context&, const Tensor& x, std::initializer_list<Tensor*> outputs, int axis,
           std::initializer_list<std::pair<int,int>> segments) {
    size_t outer = 1, inner = 1;
    for (int i = 0; i < axis; ++i) outer *= x.shape[i];
    for (size_t i = axis+1; i < x.shape.size(); ++i) inner *= x.shape[i];
    auto seg = segments.begin();
    for (auto y : outputs) {
        auto [start, n] = *seg++;
        for (size_t o = 0; o < outer; ++o)
            std::memcpy(y->data+o*n*inner, x.data+(o*x.shape[axis]+start)*inner, n*inner*sizeof(float));
    }
}
void transpose(Context& ctx, const Tensor& x, Tensor& y, std::initializer_list<int> permutation) {
    if (ctx.optimized && transpose_opt(ctx,x,y,permutation)) return;
    auto xs = strides(x.shape), ys = strides(y.shape);
    ctx.ranges(y.size(), [&](size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i) {
            size_t src = 0, d = 0;
            for (int p : permutation) { src += (i/ys[d] % y.shape[d])*xs[p]; ++d; }
            y.data[i] = x.data[src];
        }
    });
}
void matmul(Context& ctx, const Tensor& a, const Tensor& b, Tensor& y) {
    if (ctx.optimized) {
        const int m=a.shape[2],tiles=(m+3)/4;
        ctx.pool.parallel(a.shape[0]*a.shape[1]*tiles,[&](size_t job,int) {
            int row=static_cast<int>(job%tiles)*4,batch=static_cast<int>(job/tiles);
            matmul_tile(a,b,y,batch,row,std::min(4,m-row));
        });
        return;
    }
    int m = a.shape[2], k = a.shape[3], n = b.shape[3];
    int batches = a.shape[0]*a.shape[1];
    for (int batch = 0; batch < batches; ++batch)
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j) {
                float v = 0;
                for (int z = 0; z < k; ++z) v += a.data[(batch*m+i)*k+z]*b.data[(batch*k+z)*n+j];
                y.data[(batch*m+i)*n+j] = v;
            }
}
void softmax(Context& ctx, const Tensor& x, Tensor& y, int axis) {
    if (ctx.optimized) { softmax_opt(ctx,x,y,axis); return; }
    size_t inner = 1, outer = 1;
    for (int i = 0; i < axis; ++i) outer *= x.shape[i];
    for (size_t i = axis+1; i < x.shape.size(); ++i) inner *= x.shape[i];
    int n = x.shape[axis];
    for (size_t o = 0; o < outer; ++o)
        for (size_t i = 0; i < inner; ++i) {
            size_t base = o*n*inner+i;
            float maximum = -std::numeric_limits<float>::infinity(), sum = 0;
            for (int c = 0; c < n; ++c) maximum = std::max(maximum, x.data[base+c*inner]);
            for (int c = 0; c < n; ++c) { float e = std::exp(x.data[base+c*inner]-maximum); y.data[base+c*inner] = e; sum += e; }
            for (int c = 0; c < n; ++c) y.data[base+c*inner] /= sum;
        }
}
}
