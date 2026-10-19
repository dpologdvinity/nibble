#include <catch2/catch_test_macros.hpp>

#include <atomic>
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
