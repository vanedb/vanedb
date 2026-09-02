// Cross-engine cosine cases from conformance/cosine_scale_invariance.tsv.
//
// The zero-vector guard compared `na * nb` against a fixed epsilon. That
// product grows with the fourth power of magnitude, so ordinary small vectors
// were classified as zero and large ones overflowed the product to infinity —
// both returned 1.0 for identical inputs (vanedb-cpp#36 / vanedb#40).

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "core/distance.h"

namespace {

constexpr float TOLERANCE = 1e-5f;

struct Case {
  float scale;
  std::string relation;
  float expected;
};

std::vector<Case> load_cases() {
  const std::string path = std::string(VANEDB_CONFORMANCE_DIR) + "/cosine_scale_invariance.tsv";
  std::ifstream in(path);
  REQUIRE(in.is_open());

  std::vector<Case> cases;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream fields(line);
    std::string scale, relation, expected;
    std::getline(fields, scale, '\t');
    std::getline(fields, relation, '\t');
    std::getline(fields, expected, '\t');
    cases.push_back({std::stof(scale), relation, std::stof(expected)});
  }
  REQUIRE_FALSE(cases.empty());
  return cases;
}

std::pair<std::vector<float>, std::vector<float>> vectors(float s, const std::string& relation) {
  if (relation == "identical") return {{s, 2.0f * s}, {s, 2.0f * s}};
  if (relation == "opposite") return {{s, 2.0f * s}, {-s, -2.0f * s}};
  if (relation == "orthogonal") return {{s, 0.0f}, {0.0f, s}};
  if (relation == "zero") return {{0.0f, 0.0f}, {s, 2.0f * s}};
  FAIL("unknown relation: " << relation);
  return {};
}

}  // namespace

TEST_CASE("cosine distance matches the shared cases", "[conformance][distance]") {
  for (const auto& c : load_cases()) {
    auto [a, b] = vectors(c.scale, c.relation);
    const float got = vanedb::cosine_distance(a.data(), b.data(), a.size());
    INFO("scale=" << c.scale << " relation=" << c.relation);
    REQUIRE(std::fabs(got - c.expected) <= TOLERANCE);
  }
}

TEST_CASE("cosine is scale invariant across SIMD-width vectors", "[conformance][distance]") {
  for (float scale : {1e-18f, 1e-4f, 1.0f, 1e4f, 1e15f}) {
    std::vector<float> a(128);
    for (size_t i = 0; i < a.size(); ++i) a[i] = (static_cast<float>(i) + 1.0f) * scale;
    INFO("scale=" << scale);
    REQUIRE(std::fabs(vanedb::cosine_distance(a.data(), a.data(), a.size())) <= TOLERANCE);
  }
}

TEST_CASE("cosine returns one when norms overflow rather than NaN", "[conformance][distance]") {
  const std::vector<float> a(128, 1e20f);
  const float got = vanedb::cosine_distance(a.data(), a.data(), a.size());
  REQUIRE(std::isfinite(got));
  REQUIRE(std::fabs(got - 1.0f) <= TOLERANCE);
}
