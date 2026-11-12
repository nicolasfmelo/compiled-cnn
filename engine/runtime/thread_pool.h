#pragma once
#include <atomic>
#include <condition_variable>
#include <functional>
#include <immintrin.h>
#include <mutex>
#include <pthread.h>
#include <sched.h>
#include <stdexcept>
#include <thread>
#include <vector>

namespace engine {
// A single caller owns dispatch. Each published job is immutable until every
// worker has released its completion counter; publication uses release/acquire
// on epoch, completion uses release/acquire on done. No relaxed pointer handoff.
class ThreadPool {
    int count_;
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool stopping_ = false;
    std::atomic<bool> running_{false};
    alignas(64) std::atomic<unsigned long long> epoch_{0};
    alignas(64) std::atomic<int> done_{0};
    std::function<void(int)> task_;
    int depth_ = 0; // only accessed by the main/calling thread
    void worker(int id) {
        unsigned long long seen = 0;
        for (;;) {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [&] { return stopping_ || running_.load(std::memory_order_acquire); });
            if (stopping_) return;
            lock.unlock();
            while (running_.load(std::memory_order_acquire)) {
                auto epoch = epoch_.load(std::memory_order_acquire);
                if (epoch != seen) {
                    seen = epoch;
                    task_(id);
                    done_.fetch_add(1, std::memory_order_release);
                } else _mm_pause();
            }
        }
    }
    static void pin(pthread_t thread, int cpu) {
        if (cpu < 0 || cpu >= CPU_SETSIZE) throw std::runtime_error("CPU out of range");
        cpu_set_t set;
        CPU_ZERO(&set); CPU_SET(cpu, &set);
        int error = pthread_setaffinity_np(thread, sizeof(set), &set);
        if (error) throw std::runtime_error("pthread_setaffinity_np failed: "+std::to_string(error));
    }
    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
            running_.store(false, std::memory_order_release);
        }
        cv_.notify_all();
        for (auto& thread : workers_) if (thread.joinable()) thread.join();
    }
public:
    ThreadPool(int count, const std::vector<int>& cpus) : count_(count) {
        if (count < 1 || cpus.size() != static_cast<size_t>(count)) throw std::runtime_error("bad thread/CPU count");
        try {
            pin(pthread_self(), cpus[0]);
            for (int id = 1; id < count; ++id) {
                workers_.emplace_back([this,id] { worker(id); });
                pin(workers_.back().native_handle(), cpus[id]);
            }
        } catch (...) { stop(); throw; }
    }
    ~ThreadPool() { stop(); }
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    int size() const { return count_; }
    void begin() {
        if (depth_++ != 0 || count_ == 1) return;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            running_.store(true, std::memory_order_release);
        }
        cv_.notify_all();
    }
    void end() {
        if (--depth_ != 0 || count_ == 1) return;
        std::lock_guard<std::mutex> lock(mutex_);
        running_.store(false, std::memory_order_release);
    }
    struct Session {
        ThreadPool& pool;
        explicit Session(ThreadPool& p) : pool(p) { pool.begin(); }
        ~Session() { pool.end(); }
        Session(const Session&) = delete;
        Session& operator=(const Session&) = delete;
    };
    template<class F> void parallel(size_t jobs, F&& f) {
        if (count_ == 1) { for (size_t j = 0; j < jobs; ++j) f(j,0); return; }
        Session session(*this);
        task_ = [&](int id) { for (size_t j = id; j < jobs; j += count_) f(j,id); };
        done_.store(0, std::memory_order_relaxed);
        epoch_.fetch_add(1, std::memory_order_release);
        task_(0);
        while (done_.load(std::memory_order_acquire) != count_-1) _mm_pause();
    }
};
}
