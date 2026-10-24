#pragma once
// A memory-bandwidth probe used by `nibble bench --mem`: token generation reads every weight once per token, so
// sustained DRAM read bandwidth bounds decode speed (the roofline in the README).
#include <cstddef>
#include <vector>

#include "thread_pool.h"

namespace nibble::bench {

// Streaming read of a buffer of `bytes` (summing uint64 words) split across the pool,
// repeated `reps` times after one warm-up. Returns GB/s (1e9 bytes) per rep.
std::vector<double> measure_membw(ThreadPool& pool, size_t bytes, int reps);

// max(1 GiB, 4 x L3), the buffer size for the probe.
size_t membw_default_bytes();

}  // namespace nibble::bench
