#include "runtime.h"
#include <iostream>
#include <random>
#include <cfenv>

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
int quant(float v,float scale,int zp) {
    return std::clamp(int(std::nearbyint(std::clamp(v*(1.f/scale),float(-zp),float(255-zp))))+zp,0,255);
}
int main(int argc,char** argv) {
    try {
        if (!engine::has_avx_vnni()) { std::cout << "SKIP: AVX-VNNI unavailable\n"; return 77; }
        require(std::fegetround()==FE_TONEAREST,"rounding environment is not nearest-even");
        int threads=argc>1 ? std::stoi(argv[1]) : 1;
        std::vector<int> cpus;
        for (int i=0; i<threads; ++i) cpus.push_back(i*2);
        engine::Context ctx(true,threads,cpus);
        std::mt19937 rng(98);
        std::uniform_real_distribution<float> values(-40,40);
        // Dyadic scale gives exact halfway inputs and tests both tie parities.
        const float scale=.25f;
        const int zp=117;
        for (int ic : {1,3,4,5,7}) for (int oc : {1,5,6,7,13})
        for (int kw : {1,3}) for (int stride : {1,2}) for (int width : {1,7,16,19}) {
            const int h=3,pad=kw/2,oh=(h+2*pad-kw)/stride+1,ow=(width+2*pad-kw)/stride+1;
            const int n=oh*ow,k=ic*kw*kw,k4=(k+3)/4;
            std::vector<float> xs(ic*h*width),ys(oc*n),products(oc),bias(oc),residual(oc*n);
            for (float& v:xs) v=values(rng);
            const float special[]={.125f,.375f,-.125f,-.375f,1e30f,-1e30f,0.f};
            for (size_t i=0;i<xs.size() && i<7;++i) xs[i]=special[i];
            engine::Tensor x{xs.data(),{1,ic,h,width}},y{ys.data(),{1,oc,oh,ow}},res{residual.data(),y.shape};
            engine::ConvParams p{kw,kw,stride,stride,pad,pad,1,0,16,6};
            std::vector<int8_t> weights((oc+5)/6*k4*24,0);
            for (int o=0;o<oc;++o) {
                products[o]=.0001f*(o+1); bias[o]=.23f*(o-3);
                for (int z=0;z<k;++z) weights[(o/6*k4+z/4)*24+o%6*4+z%4]=int(rng()%255)-127;
            }
            // Keep the SiLU+residual reference away from cancellation at zero,
            // so the requested purely relative tolerance remains meaningful.
            for (float& v:residual) v=1.f+std::abs(values(rng))*.01f;
            for (int pos=0;pos<n;pos+=16) {
                int count=std::min(16,n-pos);
                std::vector<uint8_t> panel(k4*64,42);
                engine::pack_int8(x,y,p,pos,count,scale,zp,panel.data());
                for (int z=0;z<k4*4;++z) for (int j=0;j<16;++j) {
                    int expected=zp;
                    if (z<k && j<count) {
                        int c=z/(kw*kw),ky=z/kw%kw,kx=z%kw;
                        int sy=(pos+j)/ow*stride-pad+ky,sx=(pos+j)%ow*stride-pad+kx;
                        if (sy>=0 && sy<h && sx>=0 && sx<width) expected=quant(xs[(c*h+sy)*width+sx],scale,zp);
                    }
                    require(panel[(z/4*16+j)*4+z%4]==expected,"pack rounding/clamp/border/K tail mismatch");
                }
                for (int o=0;o<oc;o+=6) {
                    int32_t actual[96],expected[96];
                    engine::dot_int8(weights.data()+o/6*k4*24,panel.data(),k4,actual);
                    engine::dot_int8_ref(weights.data()+o/6*k4*24,panel.data(),k4,expected);
                    for (int i=0;i<96;++i) require(actual[i]==expected[i],"VNNI integer mismatch");
                }
            }
            engine::QuantConv q{weights.data(),products.data(),bias.data(),scale,zp};
            for (int act : {0,1}) {
                p.silu=act;
                engine::conv_int8(ctx,x,y,p,q,&res);
                for (int pos=0;pos<n;pos+=16) {
                    int count=std::min(16,n-pos);
                    std::vector<uint8_t> panel(k4*64);
                    engine::pack_int8(x,y,p,pos,count,scale,zp,panel.data());
                    for (int o=0;o<oc;o+=6) {
                        int32_t acc[96];
                        engine::dot_int8_ref(weights.data()+o/6*k4*24,panel.data(),k4,acc);
                        for (int r=0;r<std::min(6,oc-o);++r) for (int j=0;j<count;++j) {
                            float v=float(acc[r*16+j])*products[o+r];
                            v+=bias[o+r];
                            if (act) v=engine::silu(v);
                            int index=(o+r)*n+pos+j;
                            v+=residual[index];
                            if (!std::isfinite(ys[index]) || std::abs(v-ys[index])>1e-5f*std::abs(v)) {
                                std::cerr << "epilogue expected=" << v << " actual=" << ys[index] << " relative="
                                          << std::abs(v-ys[index])/std::abs(v) << " act=" << act << '\n';
                                throw std::runtime_error("epilogue mismatch");
                            }
                        }
                    }
                }
            }
        }
        // Large K with worst-case products catches accidental pmaddubsw's
        // saturating intermediate (two 255*127 products exceed int16).
        {
            const int k4=1153;
            std::vector<int8_t> a(k4*24,127);
            std::vector<uint8_t> b(k4*64,255);
            int32_t actual[96],expected[96];
            engine::dot_int8(a.data(),b.data(),k4,actual);
            engine::dot_int8_ref(a.data(),b.data(),k4,expected);
            for (int i=0;i<96;++i) require(actual[i]==expected[i] && actual[i]==k4*4*255*127,"large-K saturation mismatch");
        }
        // Profile counters are worker-private until the pool barrier.
        {
            std::vector<float> xs(3*19,1),ys(7*19),products(7,.001f),bias(7,0);
            std::vector<int8_t> weights(2*24,1);
            engine::Tensor x{xs.data(),{1,3,1,19}},y{ys.data(),{1,7,1,19}};
            engine::ConvParams p{1,1,1,1,0,0,1,1,16,6};
            ctx.profiling=true;
            engine::conv_int8(ctx,x,y,p,{weights.data(),products.data(),bias.data(),1.f,13});
            require(ctx.profile["int8_pack_quantize"]>0 && ctx.profile["int8_gemm_epilogue"]>0,"missing INT8 profile breakdown");
        }
        std::cout << "PASS INT8 nearest-even/clamp, nonzero zp borders, stride 2, K%4, OC%6, column tails; exact int32 VNNI/reference; epilogue 1e-5\n";
    } catch (const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
