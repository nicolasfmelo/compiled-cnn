#include "model.h"
#include "postprocess.h"
#include <sched.h>

struct Arguments {
    std::string command, impl = "opt", precision = "fp32", cpus;
    fs::path extra;
    std::optional<fs::path> check;
    int threads = 1, warmup = 20, iters = 200;
    bool profile = false;
    Arguments(int argc, char** argv) {
        if (argc < 2) throw std::runtime_error("usage: yolo_engine validate REF_DIR | bench | detect IMAGE | serve [--precision fp32|int8] [--impl ref|opt] [--threads N] [--cpus LIST] [--warmup 20] [--iters 200] [--profile] [--check DETS_TXT]");
        command = argv[1];
        int i = 2;
        if (command == "validate" || command == "detect") {
            if (i == argc) throw std::runtime_error("missing path");
            extra = argv[i++];
        } else if (command != "bench" && command != "serve") throw std::runtime_error("unknown command");
        for (; i < argc; ++i) {
            std::string key = argv[i];
            if (key == "--profile") { profile = true; continue; }
            if (++i == argc) throw std::runtime_error("missing option value");
            if (key == "--impl") impl = argv[i];
            else if (key == "--precision") precision = argv[i];
            else if (key == "--threads") threads = std::stoi(argv[i]);
            else if (key == "--cpus") cpus = argv[i];
            else if (key == "--warmup") warmup = std::stoi(argv[i]);
            else if (key == "--iters") iters = std::stoi(argv[i]);
            else if (key == "--check") check = fs::path(argv[i]);
            else throw std::runtime_error("unknown option " + key);
        }
        if ((impl != "ref" && impl != "opt") || threads < 1 || threads > 16 || warmup < 0 || iters < 1 || (impl == "ref" && threads != 1))
            throw std::runtime_error("invalid implementation/threads/iteration count");
        if (precision != "fp32" && precision != "int8") throw std::runtime_error("precision must be fp32 or int8");
        if (precision == "int8") {
#ifndef ENGINE_HAS_INT8
            throw std::runtime_error("model has no INT8 weights: regenerate with --calibration");
#endif
            if (!engine::has_avx_vnni()) throw std::runtime_error("INT8 requires AVX-VNNI (AVX-512 is not required)");
            if (impl != "opt") throw std::runtime_error("INT8 requires --impl opt; integer reference is available in unit tests");
            if (command == "validate") throw std::runtime_error("INT8 validation: run uv run python engine/validate_int8.py (round-3 criterion)");
        }
        if (cpus.empty()) {
            for (int t = 0; t < threads; ++t) cpus += (t ? "," : "") + std::to_string(t*2);
        }
    }
    std::vector<int> cpu_list() const {
        std::vector<int> list;
        std::istringstream stream(cpus);
        std::string token;
        while (std::getline(stream, token, ',')) {
            size_t end = 0;
            int cpu = std::stoi(token, &end);
            if (end != token.size() || cpu < 0 || cpu >= CPU_SETSIZE || std::find(list.begin(), list.end(), cpu) != list.end())
                throw std::runtime_error("invalid CPU list");
            list.push_back(cpu);
        }
        if (list.size() != static_cast<size_t>(threads)) throw std::runtime_error("CPU list length must equal threads");
        return list;
    }
};
int main(int argc, char** argv) {
    try {
        check_host();
        Arguments args(argc, argv);
        auto start = Clock::now();
        engine::Model model(args.impl == "opt", args.threads, args.cpu_list());
        model.ctx.int8 = args.precision == "int8";
        double load = milliseconds(start), model_ms = 0, post_ms = 0;
        auto infer = [&](const std::vector<float>& input) {
            auto t = Clock::now();
            const float* raw = model.forward(input.data());
            model_ms = milliseconds(t);
            t = Clock::now();
            auto output = postprocess(raw);
            post_ms = milliseconds(t);
            return output;
        };
        if (args.command == "serve") {
            // No text on stdout: exactly one little-endian [300,6] record per
            // complete little-endian [3,640,640] input; truncated EOF is an error.
            std::ios::sync_with_stdio(false);
            std::vector<float> input(INPUT_SIZE);
            const auto bytes=static_cast<std::streamsize>(input.size()*sizeof(float));
            for (;;) {
                std::cin.read(reinterpret_cast<char*>(input.data()),bytes);
                if (std::cin.gcount()==0 && std::cin.eof()) return 0;
                if (std::cin.gcount()!=bytes) throw std::runtime_error("truncated serve input");
                if (!std::all_of(input.begin(),input.end(),[](float v) { return std::isfinite(v); }))
                    throw std::runtime_error("non-finite serve input");
                auto output=infer(input);
                std::cout.write(reinterpret_cast<const char*>(output.data()),output.size()*sizeof(float));
                std::cout.flush();
                if (!std::cout) throw std::runtime_error("serve output pipe failed");
            }
        }
        if (args.command == "validate") {
            auto bus = args.extra / "bus.input.bin";
            if (fs::exists(bus) && fs::exists("artifacts/ncnn/bus.raw.bin")) {
                auto input = read_binary(bus, INPUT_SIZE);
                auto expected = read_binary("artifacts/ncnn/bus.raw.bin", ANCHORS*RAW_WIDTH);
                const float* raw = model.forward(input.data());
                float diff = 0;
                for (size_t i = 0; i < expected.size(); ++i) diff = std::max(diff, std::abs(raw[i]-expected[i]));
                std::cout << "raw bus max_abs_diff=" << diff << '\n';
            }
            return validate(infer, false, args.extra);
        }
        if (args.command == "detect") return detect(infer, false, args.extra, args.check);
        std::cout << "variant: engine " << args.precision << " impl=" << args.impl << " threads=" << args.threads << " cpus=" << args.cpus << '\n';
        auto input = read_binary("artifacts/ref/bus.input.bin", INPUT_SIZE);
        auto iteration = [&] { auto t = Clock::now(); infer(input); return milliseconds(t); };
        double first = iteration();
        for (int i = 0; i < args.warmup; ++i) iteration();
        model.ctx.profiling = args.profile;
        std::vector<double> times;
        double model_total = 0, post_total = 0;
        for (int i = 0; i < args.iters; ++i) {
            times.push_back(iteration()); model_total += model_ms; post_total += post_ms;
        }
        print_bench(times, load, first);
        {
            auto sorted = times;
            std::sort(sorted.begin(), sorted.end());
            std::cout << std::fixed << std::setprecision(2) << "best-case ms: p5 " << percentile(sorted, 0.05)
                      << " | min " << sorted.front() << '\n';
        }
        double mean = std::accumulate(times.begin(), times.end(), 0.0)/times.size();
        double gflops = engine::Model::flops()/1e6/mean;
        std::cout << "model ms: " << model_total/args.iters << " | post ms: " << post_total/args.iters
                  << "\neffective GFLOPS: " << gflops << " (" << gflops/8 << "% of 800 GFLOPS peak)\n";
        if (args.profile) {
            std::cout << "profile kind | ms | % model\n";
            for (const char* kind : {"conv3x3", "conv3x3s2", "conv1x1", "dwconv", "maxpool", "resize", "attention", "concat/copy", "head", "other", "int8_pack_quantize", "int8_gemm_epilogue"}) {
                double ms = model.ctx.profile[kind]/args.iters;
                std::cout << kind << " | " << ms << " | " << ms/(model_total/args.iters)*100 << '\n';
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n'; return 1;
    }
}
