// Tests for ThreadPool (src/thread_pool.cpp): work distribution, exception propagation, the spin-then-sleep
// wake-up protocol under stress, and the default thread-count policy driven by a fake sysfs.
#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

#include "nibble/error.h"
#include "thread_pool.h"

// Every index in [0, n) is processed exactly once even though chunks are claimed dynamically.
TEST_CASE("parallel_for visits every index exactly once") {
  nibble::ThreadPool pool(4);
  std::vector<std::atomic<int>> hits(10007);
  pool.parallel_for(10007, 64, [&](int64_t b, int64_t e) {
    for (auto i = b; i < e; ++i) hits[i]++;
  });
  for (auto& h : hits) REQUIRE(h.load() == 1);
}

// An empty range is a no-op.
TEST_CASE("parallel_for with n == 0 never calls fn") {
  nibble::ThreadPool pool(4);
  bool flag = false;
  pool.parallel_for(0, 8, [&](int64_t, int64_t) { flag = true; });
  REQUIRE(!flag);
}

// With one thread the work runs on the calling thread.
TEST_CASE("single-thread pool runs inline") {
  nibble::ThreadPool pool(1);
  REQUIRE(pool.size() == 1);
  std::thread::id seen;
  pool.parallel_for(10, 3, [&](int64_t, int64_t) { seen = std::this_thread::get_id(); });
  REQUIRE(seen == std::this_thread::get_id());
}

// Many consecutive calls (the decode loop's usage pattern) all complete with the right result.
TEST_CASE("1000 back-to-back calls complete") {
  nibble::ThreadPool pool(4);
  std::atomic<int64_t> sum{0};
  for (int k = 0; k < 1000; ++k) {
    pool.parallel_for(100, 7, [&](int64_t b, int64_t e) {
      int64_t s = 0;
      for (auto i = b; i < e; ++i) s += i;
      sum += s;
    });
  }
  REQUIRE(sum.load() == 1000 * 4950);
}

// Constructing a pool with zero threads is an Error.
TEST_CASE("non-positive thread count throws") { REQUIRE_THROWS_AS(nibble::ThreadPool(0), nibble::Error); }

// With 12 threads and 2 chunks the idle workers still acknowledge each call, and nothing runs twice.
TEST_CASE("more threads than chunks") {
  nibble::ThreadPool pool(12);
  for (int k = 0; k < 1000; ++k) {
    std::atomic<int> hits[2] = {0, 0};
    pool.parallel_for(2, 16, [&](int64_t b, int64_t e) {
      for (auto i = b; i < e; ++i) hits[i]++;
    });
    REQUIRE(hits[0].load() == 1);
    REQUIRE(hits[1].load() == 1);
  }
}

// An exception in one chunk (or in every chunk at once) is rethrown on the caller only after all threads have
// stopped touching the job, and the pool remains usable afterwards.
TEST_CASE("an exception from fn reaches the caller after every thread finishes") {
  nibble::ThreadPool pool(4);
  for (int k = 0; k < 200; ++k) {
    // Throw on one chunk (it may land on any thread), or on every chunk (all threads throw at once).
    const bool all = k % 2 == 1;
    std::atomic<int> running{0};
    REQUIRE_THROWS_AS(pool.parallel_for(256, 1,
                                        [&](int64_t b, int64_t) {
                                          running++;
                                          std::this_thread::yield();
                                          running--;
                                          if (all || b == k) throw nibble::Error("chunk " + std::to_string(b));
                                        }),
                      nibble::Error);
    REQUIRE(running.load() == 0);
    std::atomic<int64_t> sum{0};  // the pool is still usable and the error does not leak into the next call
    pool.parallel_for(100, 7, [&](int64_t b, int64_t e) { sum += e - b; });
    REQUIRE(sum.load() == 100);
  }
}

// After a call, all background workers eventually enter the futex sleep (so an idle pool burns no CPU) and
// still wake for the next call.
TEST_CASE("idle workers go to sleep within the spin bound") {
  using clock = std::chrono::steady_clock;
  nibble::ThreadPool pool(4);
  std::atomic<int64_t> sum{0};
  auto add = [&](int64_t b, int64_t e) { sum += e - b; };
  for (int round = 0; round < 3; ++round) {
    pool.parallel_for(64, 4, add);
    // Loose: the spin bound is tens of microseconds; 2 s absorbs scheduler noise, ctest -j and sanitizers.
    const auto t0 = clock::now();
    while (pool.sleeping_workers() < 3 && clock::now() - t0 < std::chrono::seconds(2)) std::this_thread::yield();
    REQUIRE(pool.sleeping_workers() == 3);
  }
  REQUIRE(sum.load() == 3 * 64);  // sleeping workers still wake for each call
}

// Stress for the sleep/wake race: gaps of 0-100 us between calls land on both sides of the spin bound.
TEST_CASE("calls straddling the spin bound all complete") {
  // Gaps of 0-100 us land on both sides of the spin bound, so workers wake from both the spin and the
  // sleep path, and some calls race a worker that is just entering sleep.
  nibble::ThreadPool pool(4);
  for (int k = 0; k < 2000; ++k) {
    std::atomic<int64_t> sum{0};
    pool.parallel_for(100, 7, [&](int64_t b, int64_t e) {
      for (auto i = b; i < e; ++i) sum += i;
    });
    REQUIRE(sum.load() == 4950);
    std::this_thread::sleep_for(std::chrono::microseconds((k * 37) % 101));
  }
}

// Informational benchmark of the per-call dispatch cost for 2-12 threads, hidden by default (tag
// [.dispatch]).
// Hidden: run with `nibble_tests "[.dispatch]"`. Reports per-call latency of an empty parallel_for.
TEST_CASE("dispatch latency", "[.dispatch]") {
  using clock = std::chrono::steady_clock;
  for (const int t : {2, 4, 8, 12}) {
    nibble::ThreadPool pool(t);
    std::atomic<int64_t> sink{0};
    auto fn = [&](int64_t b, int64_t e) { sink.fetch_add(e - b, std::memory_order_relaxed); };
    for (int k = 0; k < 1000; ++k) pool.parallel_for(t, 1, fn);  // warm up
    constexpr int kCalls = 20000;
    const auto t0 = clock::now();
    for (int k = 0; k < kCalls; ++k) pool.parallel_for(t, 1, fn);
    const double ns = std::chrono::duration<double, std::nano>(clock::now() - t0).count() / kCalls;
    std::printf("dispatch t=%d: %.0f ns/call\n", t, ns);
    REQUIRE(sink.load() == int64_t{kCalls + 1000} * t);
  }
}

namespace {

// Fake /sys/devices/system/cpu: one cpuN/cpu_capacity per entry (-1 = no capacity file).
// Builds a throw-away directory tree that looks like /sys/devices/system/cpu: one cpuN/cpu_capacity file per
// entry (-1 means no capacity file), removed on destruction.
struct FakeSysfs {
  std::filesystem::path root;
  explicit FakeSysfs(const std::vector<int>& capacity) {
    static std::atomic<int> n{0};
    root = std::filesystem::temp_directory_path() /
           ("nibble_sysfs_" + std::to_string(::getpid()) + "_" + std::to_string(n++));
    std::filesystem::create_directories(root / "cpufreq");  // non-cpuN entries are ignored
    for (size_t i = 0; i < capacity.size(); ++i) {
      const auto dir = root / ("cpu" + std::to_string(i));
      std::filesystem::create_directories(dir);
      if (capacity[i] >= 0) std::ofstream(dir / "cpu_capacity") << capacity[i] << "\n";
    }
  }
  ~FakeSysfs() { std::filesystem::remove_all(root); }
};

}  // namespace

// A hybrid CPU (4 fast + 8 slow) yields the number of fast CPUs.
TEST_CASE("default thread count: hybrid capacity picks the fastest CPUs") {
  FakeSysfs fs({1024, 1024, 1024, 1024, 512, 512, 512, 512, 512, 512, 512, 512});
  REQUIRE(nibble::default_thread_count(fs.root, 12) == 4);
}

// With prime, big and little tiers, every CPU above the midpoint of the extremes counts.
TEST_CASE("default thread count: three tiers count every CPU above the midpoint") {
  FakeSysfs fs({1024, 800, 800, 800, 300, 300, 300, 300});  // 1 prime + 3 big + 4 little
  REQUIRE(nibble::default_thread_count(fs.root, 8) == 4);
}

// Slightly higher capacities on favoured cores do not shrink the big tier.
TEST_CASE("default thread count: ITMT-favoured cores do not shrink the big tier") {
  FakeSysfs fs({1024, 1024, 1010, 1010, 400, 400, 400, 400, 400, 400, 400, 400});
  REQUIRE(nibble::default_thread_count(fs.root, 12) == 4);
}

// The result is clamped to the reported hardware thread count.
TEST_CASE("default thread count: never more than hardware_concurrency") {
  FakeSysfs fs({1024, 1024, 1024, 1024, 512, 512, 512, 512});  // offline cpuN dirs beyond hw
  REQUIRE(nibble::default_thread_count(fs.root, 2) == 2);
}

// Uniform capacities (as WSL2 reports) or missing data use two thirds of the hardware threads, at least 1.
TEST_CASE("default thread count: uniform or missing capacity falls back to the fraction") {
  FakeSysfs uniform(std::vector<int>(12, 1024));  // what WSL2 reports on a hybrid host
  REQUIRE(nibble::default_thread_count(uniform.root, 12) == 8);
  FakeSysfs missing(std::vector<int>(12, -1));
  REQUIRE(nibble::default_thread_count(missing.root, 12) == 8);
  REQUIRE(nibble::default_thread_count("/nonexistent/sysfs", 12) == 8);
  REQUIRE(nibble::default_thread_count("/nonexistent/sysfs", 1) == 1);
  REQUIRE(nibble::default_thread_count("/nonexistent/sysfs", 0) == 1);  // hardware_concurrency unknown
}

// One unreadable capacity file makes the whole topology untrusted, so the 2/3 rule applies.
TEST_CASE("default thread count: unreadable capacity on one CPU falls back") {
  FakeSysfs partial({1024, 1024, -1, 512});
  REQUIRE(nibble::default_thread_count(partial.root, 4) == 2);
}
