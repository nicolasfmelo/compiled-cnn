#include "thread_pool.h"
#include <iostream>
#include <algorithm>
#include <numeric>

int main() {
    try {
        for (int threads : {1,2,6}) {
            std::vector<int> cpus;
            for (int i = 0; i < threads; ++i) cpus.push_back(i*2);
            engine::ThreadPool pool(threads,cpus);
            std::vector<int> values(131,0), actual_cpu(threads,-1);
            for (int inference = 0; inference < 100; ++inference) {
                engine::ThreadPool::Session session(pool);
                for (int op = 0; op < 10; ++op) {
                    pool.parallel(values.size(), [&](size_t j, int id) { values[j]++; actual_cpu[id] = sched_getcpu(); });
                    if (!std::all_of(values.begin(),values.end(),[&](int v) { return v == inference*10+op+1; }))
                        throw std::runtime_error("pool completion/order failure");
                }
            }
            for (int id = 0; id < threads; ++id) if (actual_cpu[id] != cpus[id]) throw std::runtime_error("pool affinity failure");
            pool.parallel(0, [](size_t,int) {});
            pool.parallel(1, [&](size_t,int) { values[0]++; });
        }
        std::cout << "PASS thread pool 1/2/6 threads: affinity, exactly-once tiles, barriers, sleep/wake and empty work\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
