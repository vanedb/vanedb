// VaneDB - Copyright (c) 2025 Anton Tsvetkov - MIT License
#pragma once
#include <cstddef>

namespace vanedb::detail {

using DistanceKernel = float (*)(const float*, const float*, size_t) noexcept;

struct DistanceKernels {
  DistanceKernel l2;
  DistanceKernel dot;
  DistanceKernel cosine;
  const char* name;
};

// The false branch lets tests compare the baseline with the selected backend.
// Passing true never forces an unsupported ISA.
[[nodiscard]] const DistanceKernels& select_kernels(bool allow_avx2) noexcept;
[[nodiscard]] const DistanceKernels& runtime_kernels() noexcept;

} // namespace vanedb::detail
