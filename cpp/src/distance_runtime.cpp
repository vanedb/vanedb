// VaneDB - Copyright (c) 2025 Anton Tsvetkov - MIT License
#include "core/cpu_features.h"
#include "core/distance_kernels.h"
#include "core/distance_runtime.h"

#if defined(__AVX__) || defined(__AVX2__) || defined(__FMA__)
#error "The runtime dispatcher must be compiled for baseline x86-64"
#endif

#ifdef VANEDB_HAS_AVX2_OBJECT
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

namespace vanedb::detail {

#ifdef VANEDB_HAS_AVX2_OBJECT
// Defined in an AVX2/FMA object that is never compiled with LTO. Calling this
// function is itself forbidden until the CPU/OS check has succeeded.
const DistanceKernels& avx2_kernels() noexcept;

static bool cpu_has_avx2_fma() noexcept {
  uint32_t leaf1_ecx, leaf7_ebx;
  uint64_t xcr0;
#if defined(_MSC_VER)
  int regs[4];
  __cpuid(regs, 0);
  if (regs[0] < 7) return false;
  __cpuidex(regs, 1, 0);
  leaf1_ecx = static_cast<uint32_t>(regs[2]);
  if (!can_read_avx_state(leaf1_ecx)) return false;
  __cpuidex(regs, 7, 0);
  leaf7_ebx = static_cast<uint32_t>(regs[1]);
  xcr0 = _xgetbv(0);
#else
  if (__get_cpuid_max(0, nullptr) < 7) return false;
  unsigned int eax, ebx, ecx, edx;
  __cpuid_count(1, 0, eax, ebx, ecx, edx);
  leaf1_ecx = ecx;
  if (!can_read_avx_state(leaf1_ecx)) return false;
  __cpuid_count(7, 0, eax, ebx, ecx, edx);
  leaf7_ebx = ebx;
  // XGETBV is safe only after checking XSAVE + OSXSAVE above; unlike the
  // intrinsic, inline assembly does not require -mxsave on the baseline TU.
  __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
  xcr0 = (static_cast<uint64_t>(edx) << 32) | eax;
#endif
  return supports_avx2_fma(leaf1_ecx, leaf7_ebx, xcr0);
}
#endif

const DistanceKernels& select_kernels(bool allow_avx2) noexcept {
  static const DistanceKernels baseline = {
      compiled::l2_sq, compiled::dot_product, compiled::cosine_distance,
#ifdef VANE_ARM_NEON
      "neon"
#else
      "scalar"
#endif
  };
#ifdef VANEDB_HAS_AVX2_OBJECT
  if (allow_avx2 && cpu_has_avx2_fma()) return avx2_kernels();
#else
  (void)allow_avx2;
#endif
  return baseline;
}

const DistanceKernels& runtime_kernels() noexcept {
  // C++ guarantees thread-safe initialization. No CPUID on the hot path.
  static const DistanceKernels& kernels = select_kernels(true);
  return kernels;
}

} // namespace vanedb::detail
