// Runtime kernel selection (see dispatch.h): CPU feature detection, the NIBBLE_KERNELS override, the cached
// choice, and the error helper the tier sources throw through. Compiled with the build's target flags, never with
// a tier's, so it runs on any x86-64 CPU.
#include "backend/cpu/simd/dispatch.h"

#include <cstdlib>
#include <cstring>
#include <string>

#include <nibble/error.h>

#ifdef NIBBLE_SIMD_TIERS
#include <cpuid.h>
#endif

// NIBBLE_SIMD_TIERS: any tier was built; NIBBLE_SIMD_TIER_AVX2 / NIBBLE_SIMD_TIER_VNNI: that tier was built
// (CMakeLists.txt). The vnni tier can be missing on its own (compiler without -mavxvnni, NIBBLE_SIMD_VNNI=OFF).
#ifdef NIBBLE_SIMD_TIER_AVX2
namespace nibble::simd::avx2 {
extern const Kernels kTable;
}
#endif
#ifdef NIBBLE_SIMD_TIER_VNNI
namespace nibble::simd::vnni {
extern const Kernels kTable;
}
#endif

namespace nibble::simd {
namespace {

#ifdef NIBBLE_SIMD_TIERS
// The vnni tier's table, or nullptr when this build lacks it.
const Kernels* vnni_table() {
#ifdef NIBBLE_SIMD_TIER_VNNI
  return &vnni::kTable;
#else
  return nullptr;
#endif
}

bool allows_avx2(const CpuFeatures& f) { return f.avx2 && f.fma && f.f16c; }
// The CPU can run AVX-VNNI and this build has the tier.
bool allows_vnni(const CpuFeatures& f) { return allows_avx2(f) && f.avxvnni && vnni_table() != nullptr; }

// "scalar, avx2, avx2+vnni" limited to what f and this build allow: for error messages.
std::string supported_list(const CpuFeatures& f) {
  std::string s = "scalar";
  if (allows_avx2(f)) s += ", avx2";
  if (allows_vnni(f)) s += ", avx2+vnni";
  return s;
}
#endif

// The value as the user wrote it, for messages.
std::string quoted(const char* env) { return std::string("NIBBLE_KERNELS=\"") + env + "\""; }

// Appended to the list of valid NIBBLE_KERNELS values: which of them this build cannot honour ("" when none).
const char* unbuilt_note() {
#ifdef NIBBLE_SIMD_TIERS
  return vnni_table() == nullptr ? " (this build has no AVX-VNNI kernels)" : "";
#else
  return " (this build has no SIMD kernels)";
#endif
}

// The cached outcome of select(): a table, or the error to rethrow.
struct Choice {
  const Kernels* kernels = nullptr;
  std::string error;
};

}  // namespace

CpuFeatures cpu_features() {
  CpuFeatures f;
#ifdef NIBBLE_SIMD_TIERS
  __builtin_cpu_init();
  f.avx2 = __builtin_cpu_supports("avx2");
  f.fma = __builtin_cpu_supports("fma");
  f.f16c = __builtin_cpu_supports("f16c");
  // AVX-VNNI is CPUID leaf 7, sub-leaf 1, EAX bit 4. Read it directly: clang 18's __builtin_cpu_supports does not
  // know "avxvnni". It uses the same YMM register state as AVX2, whose OS support the avx2 check already covers.
  // Leaf 7 is queried only when the CPU has it (old CPUs and some VMs stop below leaf 7).
  const unsigned max_leaf = __get_cpuid_max(0, nullptr);
  unsigned eax = 0, ebx = 0, ecx = 0, edx = 0;
  if (max_leaf >= 7) __cpuid_count(7, 1, eax, ebx, ecx, edx);
  f.avxvnni = avxvnni_from_cpuid(max_leaf, eax);
#endif
  return f;
}

bool avxvnni_from_cpuid(unsigned max_leaf, unsigned leaf7_1_eax) {
  return max_leaf >= 7 && (leaf7_1_eax & (1u << 4)) != 0;
}

const Kernels* select_for(const char* env, const CpuFeatures& f) {
  const bool automatic = env == nullptr || *env == '\0';
  if (!automatic && std::strcmp(env, "scalar") == 0) return nullptr;
  const bool want_avx2 = !automatic && (std::strcmp(env, "avx2") == 0 || std::strcmp(env, "avx2+fma+f16c") == 0);
  const bool want_vnni =
      !automatic && (std::strcmp(env, "avx2+vnni") == 0 || std::strcmp(env, "avx2+fma+f16c+avxvnni") == 0);
  if (!automatic && !want_avx2 && !want_vnni) {
    throw Error(quoted(env) + " is not one of: scalar, avx2, avx2+vnni" + unbuilt_note());
  }
#ifdef NIBBLE_SIMD_TIERS
  if (automatic) return allows_vnni(f) ? vnni_table() : allows_avx2(f) ? &avx2::kTable : nullptr;
  if (want_avx2 && !allows_avx2(f))
    throw Error(quoted(env) + " needs a CPU with AVX2, FMA and F16C; this CPU supports: " + supported_list(f));
  if (want_vnni && vnni_table() == nullptr)
    throw Error(quoted(env) +
                ": this build has no AVX-VNNI kernels (compiler without -mavxvnni or NIBBLE_SIMD_VNNI=OFF)");
  if (want_vnni && !allows_vnni(f))
    throw Error(quoted(env) + " needs a CPU with AVX-VNNI; this CPU supports: " + supported_list(f));
  return want_vnni ? vnni_table() : &avx2::kTable;
#else
  (void)f;
  if (automatic) return nullptr;
  throw Error(quoted(env) + ": this build has no SIMD kernels (NIBBLE_SIMD=OFF or not x86-64)");
#endif
}

const Kernels* select() {
  static const Choice choice = [] {
    Choice c;
    try {
      // getenv runs once, inside this thread-safe static initialisation; nibble never calls setenv.
      c.kernels = select_for(std::getenv("NIBBLE_KERNELS"), cpu_features());  // NOLINT(concurrency-mt-unsafe)
    } catch (const Error& e) {
      c.error = e.what();
    }
    return c;
  }();
  if (!choice.error.empty()) throw Error(choice.error);
  return choice.kernels;
}

std::vector<const Kernels*> available() {
  std::vector<const Kernels*> out;
#ifdef NIBBLE_SIMD_TIERS
  const CpuFeatures f = cpu_features();
  if (allows_avx2(f)) out.push_back(&avx2::kTable);
  if (allows_vnni(f)) out.push_back(vnni_table());
#endif
  return out;
}

const char* kernel_set() {
  const Kernels* k = select();
  return k ? k->name : "scalar";
}

void throw_error(const char* message) { throw Error(message); }

}  // namespace nibble::simd
