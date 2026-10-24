#pragma once
// IEEE 754 binary16 ("half") <-> binary32 conversion, in plain C++ with no hardware dependence.
//
// Quantized blocks store their scales as raw 16-bit patterns (uint16_t), and the scalar reference code
// converts them with these functions. The AVX2 kernels use the F16C instructions instead
// (_cvtsh_ss / _mm_cvtph_ps), which give the same float for every finite value, so scalar and SIMD results
// agree. Both functions are exact where the format allows (fp16 -> fp32 is always exact).
#include <cstdint>
#include <cstring>

namespace nibble {

// Widens the half-precision bit pattern h to a float. Handles zeros, subnormals (normalized into float
// normals), infinities and NaNs (payload bits preserved, shifted up).
inline float fp16_to_fp32(uint16_t h) {
  uint32_t sign = (static_cast<uint32_t>(h) & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1Fu;
  uint32_t man = h & 0x3FFu;
  uint32_t bits;
  if (exp == 0x1F) {
    bits = sign | 0x7F800000u | (man << 13);
  } else if (exp != 0) {
    bits = sign | ((exp + 112u) << 23) | (man << 13);
  } else if (man == 0) {
    bits = sign;
  } else {
    // subnormal half: normalize
    int shift = 0;
    while ((man & 0x400u) == 0) {
      man <<= 1;
      ++shift;
    }
    man &= 0x3FFu;
    bits = sign | (static_cast<uint32_t>(113 - shift) << 23) | (man << 13);
  }
  float f;
  std::memcpy(&f, &bits, sizeof f);
  return f;
}

// Narrows f to the nearest half, rounding ties to even (the IEEE default). Overflow gives +-infinity,
// values below half's smallest subnormal give signed zero, NaN stays NaN (quiet bit set).
inline uint16_t fp32_to_fp16(float f) {
  uint32_t x;
  std::memcpy(&x, &f, sizeof x);
  uint32_t sign = (x >> 16) & 0x8000u;
  uint32_t e = (x >> 23) & 0xFFu;
  uint32_t m = x & 0x7FFFFFu;
  if (e == 0xFF) {
    return static_cast<uint16_t>(sign | 0x7C00u | (m ? (0x200u | (m >> 13)) : 0u));
  }
  int exp = static_cast<int>(e) - 127 + 15;
  if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
  if (exp <= 0) {
    if (exp < -10) return static_cast<uint16_t>(sign);
    m |= 0x800000u;
    int shift = 14 - exp;
    uint32_t half = m >> shift;
    uint32_t rem = m & ((1u << shift) - 1u);
    uint32_t mid = 1u << (shift - 1);
    if (rem > mid || (rem == mid && (half & 1u))) ++half;
    return static_cast<uint16_t>(sign | half);
  }
  uint32_t half = (static_cast<uint32_t>(exp) << 10) | (m >> 13);
  uint32_t rem = m & 0x1FFFu;
  if (rem > 0x1000u || (rem == 0x1000u && (half & 1u))) ++half;  // carry may reach inf
  return static_cast<uint16_t>(sign | half);
}

}  // namespace nibble
