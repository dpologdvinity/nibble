#pragma once
// Deterministic integer-only generator for the raw Q4_0/Q8_0/Q5_0/Q5_1/Q8_1/Q4_K/Q5_K/Q6_K/Q8_K block bytes used by the golden-vector
// tests. Shared by tests/test_quant.cpp and tools/quant_vectors (which runs llama.cpp's reference code
// on the same bytes). Integer-only so the bytes are identical on every compiler and flag set; the
// generated .inc also records an FNV-1a hash of the bytes so drift is caught.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace golden {

inline constexpr int kQ4Bytes = 18;  // fp16 d + 16 nibble bytes
inline constexpr int kQ8Bytes = 34;  // fp16 d + 32 int8
inline constexpr int kQ5_0Bytes = 22;  // fp16 d + 4 qh + 16 nibble bytes
inline constexpr int kQ5_1Bytes = 24;  // fp16 d, m + 4 qh + 16 nibble bytes
inline constexpr int kQ8_1Bytes = 36;  // fp16 d, s + 32 int8
inline constexpr int kQ4_KBytes = 144;  // fp16 d, dmin + 12 scales + 128 qs
inline constexpr int kQ5_KBytes = 176;  // fp16 d, dmin + 12 scales + 32 qh + 128 qs
inline constexpr int kQ6_KBytes = 210;  // 128 ql + 64 qh + 16 int8 scales + fp16 d
inline constexpr int kQ8_KBytes = 292;  // float d + 256 int8 + 16 int16 bsums
inline constexpr int kCraftedQ8K = 4;   // crafted_q8k_row ids 0..3

// splitmix64: tiny, fast, platform-independent random numbers so tests and the generator tool derive
// identical inputs from a seed.
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
// Random fp16 scale bits.
inline uint16_t rand_d(Rng& r) {
  const uint64_t v = r.next();
  const uint16_t sign = static_cast<uint16_t>((v & 1u) << 15);
  const uint16_t exp = static_cast<uint16_t>(((v >> 1) & 15u) == 0 ? 0u : 8u + ((v >> 5) & 7u));
  const uint16_t man = static_cast<uint16_t>((v >> 8) & 0x3FFu);
  return static_cast<uint16_t>(sign | (exp << 10) | man);
}

// Stores a 16-bit value little-endian.
inline void put_d(uint8_t* p, uint16_t d) {
  p[0] = static_cast<uint8_t>(d & 0xFF);
  p[1] = static_cast<uint8_t>(d >> 8);
}

// Fills nb consecutive blocks of each type (raw little-endian block bytes).
// Fillers: nb consecutive raw blocks of each type from the generator (random quant bits, random scales).
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

inline void fill_q5_0(Rng& r, uint8_t* out, int64_t nb) {
  for (int64_t b = 0; b < nb; ++b) {
    uint8_t* p = out + b * kQ5_0Bytes;
    put_d(p, rand_d(r));
    for (int j = 0; j < 20; ++j) p[2 + j] = static_cast<uint8_t>(r.next());  // qh then qs
  }
}
inline void fill_q5_1(Rng& r, uint8_t* out, int64_t nb) {
  for (int64_t b = 0; b < nb; ++b) {
    uint8_t* p = out + b * kQ5_1Bytes;
    put_d(p, rand_d(r));
    put_d(p + 2, rand_d(r));  // m
    for (int j = 0; j < 20; ++j) p[4 + j] = static_cast<uint8_t>(r.next());
  }
}

// fp16 bits of sum * fp16(d), truncated toward zero: within 1 ulp of fp16(sum * d). Integer-only; d must be
// finite and |sum * d| < 65504 (true for rand_d and |sum| <= 32 * 127).
// Integer-only product of an integer and an fp16 scale, truncated toward zero (so the generator needs no
// floating point and is identical on every platform).
inline uint16_t fp16_mul_int(int sum, uint16_t d) {
  const uint16_t sign = static_cast<uint16_t>((d & 0x8000u) ^ (sum < 0 ? 0x8000u : 0u));
  const int e = (d >> 10) & 31;
  uint64_t p = static_cast<uint64_t>(sum < 0 ? -static_cast<int64_t>(sum) : sum) *
               (e ? (0x400u | (d & 0x3FFu)) : (d & 0x3FFu));
  int ex = e ? e : 1;  // value = p * 2^(ex - 25) for normal and subnormal d alike
  if (p == 0) return sign;
  while (p >= 2048) {
    p >>= 1;
    ++ex;
  }
  while (p < 1024 && ex > 1) {
    p <<= 1;
    --ex;
  }
  return static_cast<uint16_t>(sign | (p >= 1024 ? (ex << 10) | (p & 0x3FFu) : p));
}

// Q8_1 activations: qs in [-127, 127] like a real quantizer's output, s = d * sum(qs).
// Q8_1 activations with quants in [-127, 127] and a consistent block sum s.
inline void fill_q8_1(Rng& r, uint8_t* out, int64_t nb) {
  for (int64_t b = 0; b < nb; ++b) {
    uint8_t* p = out + b * kQ8_1Bytes;
    const uint16_t d = rand_d(r);
    int sum = 0;
    for (int j = 0; j < 32; ++j) {
      const int q = static_cast<int>(r.next() % 255) - 127;
      p[4 + j] = static_cast<uint8_t>(q);
      sum += q;
    }
    put_d(p, d);
    put_d(p + 2, fp16_mul_int(sum, d));
  }
}

// K-quant weights: random bytes everywhere except the fp16 d (and dmin), which come from rand_d.
// K-quant weights: random bytes everywhere except the fp16 scale fields, which come from rand_d.
inline void fill_q4_K(Rng& r, uint8_t* out, int64_t nb) {
  for (int64_t b = 0; b < nb; ++b) {
    uint8_t* p = out + b * kQ4_KBytes;
    put_d(p, rand_d(r));
    put_d(p + 2, rand_d(r));
    for (int j = 4; j < kQ4_KBytes; ++j) p[j] = static_cast<uint8_t>(r.next());
  }
}
inline void fill_q5_K(Rng& r, uint8_t* out, int64_t nb) {
  for (int64_t b = 0; b < nb; ++b) {
    uint8_t* p = out + b * kQ5_KBytes;
    put_d(p, rand_d(r));
    put_d(p + 2, rand_d(r));
    for (int j = 4; j < kQ5_KBytes; ++j) p[j] = static_cast<uint8_t>(r.next());
  }
}
inline void fill_q6_K(Rng& r, uint8_t* out, int64_t nb) {
  for (int64_t b = 0; b < nb; ++b) {
    uint8_t* p = out + b * kQ6_KBytes;
    for (int j = 0; j < kQ6_KBytes - 2; ++j) p[j] = static_cast<uint8_t>(r.next());
    put_d(p + kQ6_KBytes - 2, rand_d(r));
  }
}

// Stores a float's bytes.
inline void put_f32(uint8_t* p, float f) { std::memcpy(p, &f, 4); }

// Q8_K activations: qs in [-127, 127], consistent bsums, random finite float d (exact: built from bits).
// Q8_K activations: quants in [-127, 127], consistent bsums and a random finite float scale.
inline void fill_q8_K(Rng& r, uint8_t* out, int64_t nb) {
  for (int64_t b = 0; b < nb; ++b) {
    uint8_t* p = out + b * kQ8_KBytes;
    const uint64_t v = r.next();
    const uint32_t bits = static_cast<uint32_t>(((v & 1u) << 31) | ((120u + ((v >> 1) & 15u)) << 23) | ((v >> 8) & 0x7FFFFFu));
    float d;
    std::memcpy(&d, &bits, 4);
    put_f32(p, d);
    for (int j = 0; j < 256; ++j) p[4 + j] = static_cast<uint8_t>(static_cast<int>(r.next() % 255) - 127);
    for (int g = 0; g < 16; ++g) {
      int sum = 0;
      for (int i = 0; i < 16; ++i) sum += static_cast<int8_t>(p[4 + g * 16 + i]);
      const uint16_t s = static_cast<uint16_t>(static_cast<int16_t>(sum));
      put_d(p + 260 + g * 2, s);
    }
  }
}

// Float rows for the Q8_K quantizer: per block a power-of-two scale, elements are 21-bit integers times it
// (exact in float, no rounding or FMA contraction involved), with an occasional 8x outlier.
// Float rows for the Q8_K quantizer tests: integers times a power-of-two scale, so every input is exactly
// representable and the quantizer's arithmetic involves no hidden rounding of the inputs.
inline void q8k_rows(Rng& r, float* x, int64_t n) {
  for (int64_t b = 0; b < n / 256; ++b) {
    const int e = static_cast<int>(r.next() % 20) - 10 - 20;
    for (int j = 0; j < 256; ++j) {
      const uint64_t v = r.next();
      int k = static_cast<int>(v % (1u << 21)) - (1 << 20);
      if (((v >> 32) & 63u) == 0) k *= 8;
      x[b * 256 + j] = std::ldexp(static_cast<float>(k), e);
    }
  }
}

// Crafted 256-float rows: 0 zero; 1 sign tie (-5 first, +5 later); 2 half-way values with iscale = 1; 3 positive max.
// Hand-made 256-float rows exercising Q8_K corner cases (id selects the case).
inline void crafted_q8k_row(int id, float* x) {
  for (int j = 0; j < 256; ++j) x[j] = 0.f;
  switch (id) {
    case 1:
      x[3] = -5.f;
      x[10] = 5.f;
      break;
    case 2:
      x[0] = -127.f;
      for (int j = 1; j < 256; ++j) x[j] = static_cast<float>((j % 61) - 30) + 0.5f;
      break;
    case 3:
      x[0] = 127.f;
      for (int j = 1; j < 256; ++j) x[j] = static_cast<float>((j % 5) - 2) * 3.f;
      break;
    default:
      break;
  }
}

// FNV-1a hash, used to pin large generated inputs and outputs in the golden file.
inline uint64_t fnv1a(const uint8_t* p, size_t n, uint64_t h = 0xCBF29CE484222325ull) {
  for (size_t i = 0; i < n; ++i) h = (h ^ p[i]) * 0x100000001B3ull;
  return h;
}

}  // namespace golden
