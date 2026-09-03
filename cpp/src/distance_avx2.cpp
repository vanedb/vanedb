// VaneDB - Copyright (c) 2025 Anton Tsvetkov - MIT License
#include "core/distance_kernels.h"
#include "core/distance_runtime.h"

#ifndef VANE_AVX2
#error "Compile only this object with AVX2 and FMA enabled"
#endif

namespace vanedb::detail {

const DistanceKernels& avx2_kernels() noexcept {
  static const DistanceKernels kernels = {
      avx2_fma::l2_sq, avx2_fma::dot_product, avx2_fma::cosine_distance, "avx2_fma"};
  return kernels;
}

} // namespace vanedb::detail
