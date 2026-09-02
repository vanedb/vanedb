// VaneDB - Copyright (c) 2025 Anton Tsvetkov - MIT License
#pragma once

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace vanedb::detail {

inline bool distance_less(float left, float right) noexcept {
  constexpr float negative_infinity = -std::numeric_limits<float>::infinity();
  if (left < right) return left != negative_infinity;
  if (right < left) return right == negative_infinity;
  if (std::isnan(left) != std::isnan(right)) return !std::isnan(left);
  return false;
}

inline void require_finite(const float* values, size_t count, const char* input) {
  for (size_t i = 0; i < count; ++i) {
    if (!std::isfinite(values[i]))
      throw std::invalid_argument(std::string(input) + " must contain only finite values");
  }
}

struct DistanceIdLess {
  bool operator()(const std::pair<float, size_t>& left,
                  const std::pair<float, size_t>& right) const noexcept {
    if (distance_less(left.first, right.first)) return true;
    if (distance_less(right.first, left.first)) return false;
    return left.second < right.second;
  }
};

struct DistanceIdGreater {
  bool operator()(const std::pair<float, size_t>& left,
                  const std::pair<float, size_t>& right) const noexcept {
    return DistanceIdLess{}(right, left);
  }
};

}  // namespace vanedb::detail
