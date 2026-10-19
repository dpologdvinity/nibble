#include "thread_pool.h"

#include <algorithm>
#include <chrono>
#include <exception>
#include <fstream>
#include <string>
#include <system_error>
#include <utility>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

#include "nibble/error.h"

namespace nibble {

namespace {

// How long a worker polls for the next call before sleeping. A decode step issues ~200 calls with
// short serial gaps between them; this covers the gaps but not the time between tokens.
constexpr auto kSpin = std::chrono::microseconds(50);

inline void cpu_relax() {
#if defined(__x86_64__) || defined(__i386__)
  _mm_pause();
#else
  std::this_thread::yield();
#endif
}

}  // namespace

ThreadPool::ThreadPool(int n_threads) : n_threads_(n_threads) {
  if (n_threads <= 0) throw Error("ThreadPool: n_threads must be positive, got " + std::to_string(n_threads));
  // If creating a worker throws, the destructor will not run: stop and join the ones
  // already started (destroying a joinable std::thread would call std::terminate).
  try {
    workers_.reserve(static_cast<size_t>(n_threads - 1));
    for (int i = 1; i < n_threads; ++i) workers_.emplace_back([this] { worker_loop(); });
  } catch (const std::exception& e) {
    stop_.store(true);
    publish();
    for (auto& t : workers_) t.join();
    throw Error("ThreadPool: cannot create " + std::to_string(n_threads) + " threads: " + e.what());
  }
}

ThreadPool::~ThreadPool() {
  stop_.store(true);
  publish();
  for (auto& t : workers_) t.join();
}

void ThreadPool::publish() {
  generation_.fetch_add(1);  // seq_cst: pairs with the sleeper's sleeping_ increment (Dekker)
  if (sleeping_.load() > 0) generation_.notify_all();
}

void ThreadPool::run_chunks() {
  for (;;) {
    int64_t b = next_.fetch_add(chunk_, std::memory_order_relaxed);
    if (b >= n_) break;
    try {
      (*fn_)(b, std::min(b + chunk_, n_));
    } catch (...) {
      std::lock_guard<std::mutex> lock(error_mu_);
      if (!error_) error_ = std::current_exception();
      next_.store(n_, std::memory_order_relaxed);  // skip the chunks nobody has claimed yet
      return;
    }
  }
}

void ThreadPool::worker_loop() {
  uint32_t seen = 0;  // generation_ starts at 0; a worker that starts late must still ack the first call
  for (;;) {
    uint32_t g = generation_.load(std::memory_order_acquire);
    if (g == seen) {
      const auto deadline = std::chrono::steady_clock::now() + kSpin;
      for (unsigned i = 1; g == seen; ++i) {
        cpu_relax();
        g = generation_.load(std::memory_order_acquire);
        if (g == seen && i % 64 == 0 && std::chrono::steady_clock::now() > deadline) {
          // Either publish() sees our increment and notifies, or we see its bump here.
          sleeping_.fetch_add(1);
          while ((g = generation_.load()) == seen) generation_.wait(seen);
          sleeping_.fetch_sub(1);
        }
      }
    }
    seen = g;
    if (stop_.load(std::memory_order_acquire)) return;
    run_chunks();
    pending_.fetch_sub(1, std::memory_order_acq_rel);
  }
}

void ThreadPool::parallel_for(int64_t n, int64_t chunk,
                              FunctionRef<void(int64_t, int64_t)> fn) {
  if (n <= 0) return;
  if (chunk < 1) chunk = 1;
  if (workers_.empty()) {
    for (int64_t b = 0; b < n; b += chunk) fn(b, std::min(b + chunk, n));
    return;
  }
  fn_ = &fn;  // dangles after return; workers read it only between a new generation and their pending_ ack
  n_ = n;
  chunk_ = chunk;
  next_.store(0, std::memory_order_relaxed);
  pending_.store(static_cast<int>(workers_.size()), std::memory_order_relaxed);
  publish();  // release: the stores above are visible to any worker that acquires the new generation
  run_chunks();
  // Every worker acks every generation, so none can still touch fn_ or the fields once this returns.
  for (unsigned i = 1; pending_.load(std::memory_order_acquire) != 0; ++i) {  // unsigned: wraps, no UB
    if (i % 1024 == 0) std::this_thread::yield(); else cpu_relax();
  }
  // The acquire above orders every worker's error_ write before this read.
  if (error_) std::rethrow_exception(std::exchange(error_, nullptr));
}

int default_thread_count(const std::filesystem::path& cpu_dir, unsigned hw) {
  const int fallback = std::max(1, static_cast<int>(hw) * 2 / 3);
  std::vector<long> caps;
  std::error_code ec;
  for (std::filesystem::directory_iterator it(cpu_dir, ec), end; !ec && it != end; it.increment(ec)) {
    const std::string name = it->path().filename().string();
    if (name.size() < 4 || name.compare(0, 3, "cpu") != 0 ||
        name.find_first_not_of("0123456789", 3) != std::string::npos)
      continue;
    long c = -1;
    std::ifstream(it->path() / "cpu_capacity") >> c;
    if (c <= 0) return fallback;
    caps.push_back(c);
  }
  if (ec || caps.empty()) return fallback;
  const auto [lo, hi] = std::minmax_element(caps.begin(), caps.end());
  if (*lo == *hi) return fallback;
  // Above the midpoint rather than at the max: prime cores and ITMT-favoured cores sit a little above the big tier.
  const long mid = (*lo + *hi) / 2;
  const auto big = std::count_if(caps.begin(), caps.end(), [mid](long c) { return c > mid; });
  return std::clamp(static_cast<int>(big), 1, std::max(1, static_cast<int>(hw)));  // offline cpuN dirs count too
}

}  // namespace nibble
