// VaneDB - Copyright (c) 2025 Anton Tsvetkov - MIT License
#pragma once
#include "distance.h"
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace vanedb {

enum class Metric { L2 = 0, COSINE = 1, DOT = 2 };

// Default-constructed instances are invalid; operator() returns infinity.
// Default-constructibility is required for DiskStore which assigns
// dist_ in its constructor body (after parsing metric/dim from the file).
//
// Header-only operator() dispatches via switch so the SIMD distance functions
// inline into the hot loop. Wheels use the same metric switch with a separate
// runtime ISA dispatcher; do not impose that indirection on header-only users.
class DistanceComputer {
public:
  DistanceComputer() noexcept = default;

  DistanceComputer(Metric metric, size_t dimension)
      : dim_(dimension), metric_(metric), valid_(true) {
    switch (metric) {
      case Metric::L2:
      case Metric::COSINE:
      case Metric::DOT:
        return;
    }
    throw std::invalid_argument("DistanceComputer: invalid Metric value");
  }

  [[nodiscard]] float operator()(const float* a, const float* b) const noexcept {
    if (!valid_) [[unlikely]] return std::numeric_limits<float>::infinity();
    switch (metric_) {
      case Metric::L2:     return l2_sq(a, b, dim_);
      case Metric::COSINE: return cosine_distance(a, b, dim_);
      case Metric::DOT:    return -dot_product(a, b, dim_);
    }
    return std::numeric_limits<float>::infinity();
  }

  [[nodiscard]] size_t dimension() const noexcept { return dim_; }

private:
  size_t dim_ = 0;
  Metric metric_ = Metric::L2;
  bool valid_ = false;
};

} // namespace vanedb
