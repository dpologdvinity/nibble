#pragma once
// A small fork-join thread pool for data-parallel loops (the matmul kernels split rows across it). The
// caller participates as a worker, so a pool of N threads owns N-1 background threads.
#include <atomic>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <mutex>
#include <thread>
#include <vector>

#include "util/function_ref.h"

namespace nibble {

// Fixed-size pool; n_threads counts the calling thread. If fn throws, parallel_for waits for every thread to finish,
// skips the chunks not yet started, and rethrows the first exception on the caller.
// Workers spin briefly on a generation counter after each call, then sleep (futex via atomic::wait),
// so back-to-back calls skip the kernel and an idle pool burns no CPU.
class ThreadPool {
 public:
  // Starts n_threads - 1 workers. Throws Error if n_threads <= 0 or a thread cannot be created.
  explicit ThreadPool(int n_threads);
  ~ThreadPool();  // stops and joins the workers
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  int size() const { return n_threads_; }  // total threads, caller included
  // Splits [0, n) into chunks of `chunk` indices and runs fn(begin, end) on each, distributing chunks across
  // the pool dynamically (an atomic cursor). Returns when all chunks are done. Calls must not overlap or
  // nest (one parallel_for at a time per pool). n <= 0 does nothing; chunk < 1 is treated as 1.
  void parallel_for(int64_t n, int64_t chunk, FunctionRef<void(int64_t begin, int64_t end)> fn);
  // Workers that passed the spin bound and entered the sleep path; some may not be in the futex yet
  // (test hook).
  int sleeping_workers() const { return sleeping_.load(); }

 private:
  void worker_loop();  // body of each background thread: wait for a new generation, run chunks, ack
  void run_chunks();   // claim and run chunks until the cursor passes n (used by workers and the caller)
  void publish();      // bump generation_ and wake sleepers

  int n_threads_;
  std::vector<std::thread> workers_;
  // 32-bit so atomic::wait maps straight onto a futex.
  std::atomic<uint32_t> generation_{0};
  std::atomic<int> pending_{0};
  std::atomic<int> sleeping_{0};
  std::atomic<bool> stop_{false};
  // Written by the caller before the release bump of generation_; read by workers after acquiring it.
  const FunctionRef<void(int64_t, int64_t)>* fn_ = nullptr;
  int64_t n_ = 0, chunk_ = 1;
  std::atomic<int64_t> next_{0};
  std::mutex error_mu_;
  std::exception_ptr error_;  // first exception thrown by fn in the current call
};

// Thread count for n_threads = 0. cpu_dir is /sys/devices/system/cpu (injectable for tests); hw is
// std::thread::hardware_concurrency(). If every cpuN/cpu_capacity is readable and they differ (hybrid
// CPU), returns the number of CPUs above the midpoint of the lowest and highest capacity, clamped
// to [1, hw]. Otherwise (missing, or uniform as WSL2
// reports even on hybrid hosts) returns 2/3 of hw, at least 1: the M6 sweep on the i7-1265U picked
// 8 of 12 for tg128 (README, Performance).
int default_thread_count(const std::filesystem::path& cpu_dir, unsigned hw);

}  // namespace nibble
