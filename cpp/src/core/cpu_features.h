// VaneDB - Copyright (c) 2025 Anton Tsvetkov - MIT License
#pragma once
#include <cstdint>

namespace vanedb::detail {

// CPUID.1:ECX: FMA, XSAVE, OSXSAVE, AVX. CPUID.7.0:EBX: AVX2.
// OSXSAVE alone is insufficient: XCR0 must enable both XMM and YMM state.
inline constexpr uint32_t avx_prerequisites =
    (1u << 12) | (1u << 26) | (1u << 27) | (1u << 28);

[[nodiscard]] constexpr bool can_read_avx_state(uint32_t leaf1_ecx) noexcept {
  return (leaf1_ecx & avx_prerequisites) == avx_prerequisites;
}

[[nodiscard]] constexpr bool supports_avx2_fma(uint32_t leaf1_ecx,
                                             uint32_t leaf7_ebx,
                                             uint64_t xcr0) noexcept {
  return can_read_avx_state(leaf1_ecx) && (leaf7_ebx & (1u << 5)) != 0 &&
         (xcr0 & 0x6) == 0x6;
}

} // namespace vanedb::detail
