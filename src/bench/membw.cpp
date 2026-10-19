#include "bench/membw.h"

#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>

namespace nibble::bench {

size_t membw_default_bytes() {
  const long l3 = sysconf(_SC_LEVEL3_CACHE_SIZE);
  return std::max<size_t>(size_t{1} << 30, l3 > 0 ? 4 * static_cast<size_t>(l3) : 0);
}

std::vector<double> measure_membw(ThreadPool& pool, size_t bytes, int reps) {
  const int64_t n = static_cast<int64_t>(bytes / sizeof(uint64_t));
  std::unique_ptr<uint64_t[]> buf(new uint64_t[static_cast<size_t>(n)]);
  // Touch every page in parallel so first-touch cost is outside the timing.
  pool.parallel_for(n, 1 << 16, [&](int64_t b, int64_t e) {
    for (int64_t i = b; i < e; ++i) buf[static_cast<size_t>(i)] = static_cast<uint64_t>(i);
  });
  std::atomic<uint64_t> sink{0};
  auto once = [&] {
    pool.parallel_for(n, 1 << 16, [&](int64_t b, int64_t e) {
      uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
      int64_t i = b;
      for (; i + 4 <= e; i += 4) {
        s0 += buf[static_cast<size_t>(i)];
        s1 += buf[static_cast<size_t>(i) + 1];
        s2 += buf[static_cast<size_t>(i) + 2];
        s3 += buf[static_cast<size_t>(i) + 3];
      }
      for (; i < e; ++i) s0 += buf[static_cast<size_t>(i)];
      sink.fetch_add(s0 + s1 + s2 + s3, std::memory_order_relaxed);  // keeps the loads live
    });
  };
  once();  // warm-up
  std::vector<double> out;
  for (int r = 0; r < reps; ++r) {
    const auto t0 = std::chrono::steady_clock::now();
    once();
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    out.push_back(static_cast<double>(n) * sizeof(uint64_t) / s / 1e9);
  }
  volatile uint64_t keep = sink.load();  // read after the loop so the sums cannot be elided
  (void)keep;
  return out;
}

}  // namespace nibble::bench
