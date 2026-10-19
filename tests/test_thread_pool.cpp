#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <thread>
#include <vector>

#include "nibble/error.h"
#include "thread_pool.h"

TEST_CASE("parallel_for visits every index exactly once") {
  nibble::ThreadPool pool(4);
  std::vector<std::atomic<int>> hits(10007);
  pool.parallel_for(10007, 64, [&](int64_t b, int64_t e) {
    for (auto i = b; i < e; ++i) hits[i]++;
  });
  for (auto& h : hits) REQUIRE(h.load() == 1);
}

TEST_CASE("parallel_for with n == 0 never calls fn") {
  nibble::ThreadPool pool(4);
  bool flag = false;
  pool.parallel_for(0, 8, [&](int64_t, int64_t) { flag = true; });
  REQUIRE(!flag);
}

TEST_CASE("single-thread pool runs inline") {
  nibble::ThreadPool pool(1);
  REQUIRE(pool.size() == 1);
  std::thread::id seen;
  pool.parallel_for(10, 3, [&](int64_t, int64_t) { seen = std::this_thread::get_id(); });
  REQUIRE(seen == std::this_thread::get_id());
}

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

TEST_CASE("non-positive thread count throws") {
  REQUIRE_THROWS_AS(nibble::ThreadPool(0), nibble::Error);
}

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

TEST_CASE("idle workers go to sleep within the spin bound") {
  using clock = std::chrono::steady_clock;
  nibble::ThreadPool pool(4);
  std::atomic<int64_t> sum{0};
  auto add = [&](int64_t b, int64_t e) { sum += e - b; };
  for (int round = 0; round < 3; ++round) {
    pool.parallel_for(64, 4, add);
    // Loose: the spin bound is tens of microseconds; 200 ms absorbs scheduler noise and sanitizers.
    const auto t0 = clock::now();
    while (pool.sleeping_workers() < 3 && clock::now() - t0 < std::chrono::milliseconds(200)) std::this_thread::yield();
    REQUIRE(pool.sleeping_workers() == 3);
  }
  REQUIRE(sum.load() == 3 * 64);  // sleeping workers still wake for each call
}

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
