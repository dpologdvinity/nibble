#pragma once
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>
#include <atomic>

namespace nibble {

// n_threads counts the calling thread. fn must not throw.
class ThreadPool {
 public:
  explicit ThreadPool(int n_threads);
  ~ThreadPool();
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  int size() const { return n_threads_; }
  void parallel_for(int64_t n, int64_t chunk,
                    const std::function<void(int64_t begin, int64_t end)>& fn);

 private:
  void worker_loop();
  void run_chunks();

  int n_threads_;
  std::vector<std::thread> workers_;
  std::mutex mu_;
  std::condition_variable cv_work_, cv_done_;
  uint64_t generation_ = 0;
  int pending_ = 0;
  bool stop_ = false;
  const std::function<void(int64_t, int64_t)>* fn_ = nullptr;
  int64_t n_ = 0, chunk_ = 1;
  std::atomic<int64_t> next_{0};
};

}  // namespace nibble
