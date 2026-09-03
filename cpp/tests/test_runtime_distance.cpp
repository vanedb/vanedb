#include "core/cpu_features.h"
#include "core/distance.h"
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <string_view>
#include <vector>

using Catch::Approx;

TEST_CASE("runtime ISA selection requires CPU and OS support", "[distance][runtime]") {
  using namespace vanedb::detail;
  constexpr uint32_t avx2 = 1u << 5;
  REQUIRE(supports_avx2_fma(avx_prerequisites, avx2, 0x6));
  for (unsigned bit : {12u, 26u, 27u, 28u}) {
    INFO("missing CPUID.1:ECX bit " << bit);
    const auto missing = avx_prerequisites & ~(1u << bit);
    REQUIRE_FALSE(can_read_avx_state(missing));
    REQUIRE_FALSE(supports_avx2_fma(missing, avx2, 0x6));
  }
  REQUIRE_FALSE(supports_avx2_fma(avx_prerequisites, 0, 0x6));
  for (uint64_t state : {0u, 0x2u, 0x4u}) {
    INFO("XCR0=" << state);
    REQUIRE_FALSE(supports_avx2_fma(avx_prerequisites, avx2, state));
  }
}

TEST_CASE("runtime and baseline kernels match independent references", "[distance][runtime]") {
  using namespace vanedb::detail;
  const auto& baseline = select_kernels(false);
  const auto& selected = runtime_kernels();
  REQUIRE(std::string_view(baseline.name) != "avx2_fma");
  REQUIRE(&selected == &runtime_kernels());
  REQUIRE(&selected == &select_kernels(true));

  for (size_t n : {1u, 7u, 8u, 9u, 16u, 31u, 32u, 33u, 127u, 768u, 773u, 1536u}) {
    INFO("dimension=" << n << " backend=" << selected.name);
    // Non-aligned pointers and tails exercise both vector and remainder loops.
    std::vector<float> a(n + 1), b(n + 1);
    double l2 = 0, dot = 0, na = 0, nb = 0;
    for (size_t i = 1; i <= n; ++i) {
      a[i] = static_cast<float>(std::sin(static_cast<double>(i)));
      b[i] = static_cast<float>(std::cos(static_cast<double>(i)));
      const double x = a[i], y = b[i];
      l2 += (x - y) * (x - y);
      dot += x * y;
      na += x * x;
      nb += y * y;
    }
    for (const auto* kernels : {&baseline, &selected}) {
      REQUIRE(kernels->l2(a.data() + 1, b.data() + 1, n) == Approx(l2).epsilon(2e-5));
      REQUIRE(kernels->dot(a.data() + 1, b.data() + 1, n) == Approx(dot).margin(2e-5));
      REQUIRE(kernels->cosine(a.data() + 1, b.data() + 1, n) ==
              Approx(1 - dot / (std::sqrt(na) * std::sqrt(nb))).margin(2e-5));
    }
    REQUIRE(vanedb::l2_sq(a.data() + 1, b.data() + 1, n) == Approx(l2).epsilon(2e-5));
  }
}

TEST_CASE("runtime vector loops preserve cosine scaling", "[distance][runtime]") {
  for (float scale : {1e-18f, 1e-9f, 1.0f, 1e9f, 1e18f}) {
    std::vector<float> values(64, scale);
    for (const auto* kernels : {&vanedb::detail::select_kernels(false),
                               &vanedb::detail::runtime_kernels()}) {
      INFO("scale=" << scale << " backend=" << kernels->name);
      REQUIRE(kernels->cosine(values.data(), values.data(), values.size()) ==
              Approx(0.0f).margin(2e-6));
    }
  }
}
