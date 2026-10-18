#pragma once
// Deterministic integer-only generator for the raw Q4_0/Q8_0 block bytes used by the golden-vector
// tests. Shared by tests/test_quant.cpp and tools/quant_vectors (which runs llama.cpp's reference code
// on the same bytes). Integer-only so the bytes are identical on every compiler and flag set; the
// generated .inc also records an FNV-1a hash of the bytes so drift is caught.
#include <cstddef>
#include <cstdint>

namespace golden {

inline constexpr int kQ4Bytes = 18;  // fp16 d + 16 nibble bytes
inline constexpr int kQ8Bytes = 34;  // fp16 d + 32 int8

struct Rng {
  uint64_t s;
  uint64_t next() {  // splitmix64
    uint64_t z = (s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }
};

// fp16 scale bits: random sign and mantissa; exponent field 8..15 (|d| in [2^-7, 2^1), the range real
// weight/activation scales occupy), or 0 (zero/subnormal) one time in 16.
inline uint16_t rand_d(Rng& r) {
  const uint64_t v = r.next();
  const uint16_t sign = static_cast<uint16_t>((v & 1u) << 15);
  const uint16_t exp = static_cast<uint16_t>(((v >> 1) & 15u) == 0 ? 0u : 8u + ((v >> 5) & 7u));
  const uint16_t man = static_cast<uint16_t>((v >> 8) & 0x3FFu);
  return static_cast<uint16_t>(sign | (exp << 10) | man);
}

inline void put_d(uint8_t* p, uint16_t d) {
  p[0] = static_cast<uint8_t>(d & 0xFF);
  p[1] = static_cast<uint8_t>(d >> 8);
}

// Fills nb consecutive blocks of each type (raw little-endian block bytes).
inline void fill_q8_0(Rng& r, uint8_t* out, int64_t nb) {
  for (int64_t b = 0; b < nb; ++b) {
    uint8_t* p = out + b * kQ8Bytes;
    put_d(p, rand_d(r));
    for (int j = 0; j < 32; ++j) p[2 + j] = static_cast<uint8_t>(r.next());
  }
}
inline void fill_q4_0(Rng& r, uint8_t* out, int64_t nb) {
  for (int64_t b = 0; b < nb; ++b) {
    uint8_t* p = out + b * kQ4Bytes;
    put_d(p, rand_d(r));
    for (int j = 0; j < 16; ++j) p[2 + j] = static_cast<uint8_t>(r.next());
  }
}

inline uint64_t fnv1a(const uint8_t* p, size_t n, uint64_t h = 0xCBF29CE484222325ull) {
  for (size_t i = 0; i < n; ++i) h = (h ^ p[i]) * 0x100000001B3ull;
  return h;
}

}  // namespace golden
