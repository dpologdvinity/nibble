#include "thread_pool.h"

#include <algorithm>

#include "nibble/error.h"

namespace nibble {

ThreadPool::ThreadPool(int n_threads) : n_threads_(n_threads) {
  if (n_threads <= 0) throw Error("ThreadPool: n_threads must be positive, got " + std::to_string(n_threads));
  // If creating a worker throws, the destructor will not run: stop and join the ones
  // already started (destroying a joinable std::thread would call std::terminate).
  try {
    workers_.reserve(static_cast<size_t>(n_threads - 1));
    for (int i = 1; i < n_threads; ++i) workers_.emplace_back([this] { worker_loop(); });
  } catch (const std::exception& e) {
    {
      std::lock_guard<std::mutex> lk(mu_);
      stop_ = true;
    }
    cv_work_.notify_all();
    for (auto& t : workers_) t.join();
    throw Error("ThreadPool: cannot create " + std::to_string(n_threads) + " threads: " + e.what());
  }
}

ThreadPool::~ThreadPool() {
  {
    std::lock_guard<std::mutex> lk(mu_);
    stop_ = true;
  }
  cv_work_.notify_all();
  for (auto& t : workers_) t.join();
}

void ThreadPool::run_chunks() {
  for (;;) {
    int64_t b = next_.fetch_add(chunk_);
    if (b >= n_) break;
    (*fn_)(b, std::min(b + chunk_, n_));
  }
}

void ThreadPool::worker_loop() {
  uint64_t seen = 0;
  for (;;) {
    {
      std::unique_lock<std::mutex> lk(mu_);
      cv_work_.wait(lk, [&] { return stop_ || generation_ != seen; });
      if (stop_) return;
      seen = generation_;
    }
    run_chunks();
    {
      std::lock_guard<std::mutex> lk(mu_);
      if (--pending_ == 0) cv_done_.notify_one();
    }
  }
}

void ThreadPool::parallel_for(int64_t n, int64_t chunk,
                              const std::function<void(int64_t, int64_t)>& fn) {
  if (n <= 0) return;
  if (chunk < 1) chunk = 1;
  if (workers_.empty()) {
    for (int64_t b = 0; b < n; b += chunk) fn(b, std::min(b + chunk, n));
    return;
  }
  {
    std::lock_guard<std::mutex> lk(mu_);
    fn_ = &fn;
    n_ = n;
    chunk_ = chunk;
    next_.store(0);
    pending_ = static_cast<int>(workers_.size());
    ++generation_;
  }
  cv_work_.notify_all();
  run_chunks();
  std::unique_lock<std::mutex> lk(mu_);
  cv_done_.wait(lk, [&] { return pending_ == 0; });
}

}  // namespace nibble
