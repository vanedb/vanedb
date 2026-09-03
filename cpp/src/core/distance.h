// VaneDB - Copyright (c) 2025 Anton Tsvetkov - MIT License
#pragma once

#ifdef VANEDB_RUNTIME_DISPATCH
#include "distance_runtime.h"
#else
#include "distance_kernels.h"
#endif

namespace vanedb {

// Header-only consumers retain compile-time specialization and inlining.
// Distributable Python wheels instead select an isolated kernel object once,
// after checking CPU features AND the OS's saved AVX register state.
[[nodiscard]] inline float l2_sq(const float* a, const float* b, size_t n) noexcept {
#ifdef VANEDB_RUNTIME_DISPATCH
  return detail::runtime_kernels().l2(a, b, n);
#else
  return detail::compiled::l2_sq(a, b, n);
#endif
}

[[nodiscard]] inline float dot_product(const float* a, const float* b, size_t n) noexcept {
#ifdef VANEDB_RUNTIME_DISPATCH
  return detail::runtime_kernels().dot(a, b, n);
#else
  return detail::compiled::dot_product(a, b, n);
#endif
}

[[nodiscard]] inline float cosine_distance(const float* a, const float* b, size_t n) noexcept {
#ifdef VANEDB_RUNTIME_DISPATCH
  return detail::runtime_kernels().cosine(a, b, n);
#else
  return detail::compiled::cosine_distance(a, b, n);
#endif
}

} // namespace vanedb
