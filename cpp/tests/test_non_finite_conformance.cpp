#include "core/index.h"
#include "core/disk_store.h"
#include "core/store.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

std::vector<std::pair<std::string, float>> cases() {
  std::ifstream fixture(std::string(VANEDB_CONFORMANCE_DIR) + "/non_finite_vectors.tsv");
  REQUIRE(fixture.is_open());

  std::vector<std::pair<std::string, float>> result;
  for (std::string line; std::getline(fixture, line);) {
    if (line.empty() || line[0] == '#') continue;
    const size_t tab = line.find('\t');
    REQUIRE(tab != std::string::npos);
    result.emplace_back(line.substr(0, tab), std::stof(line.substr(tab + 1)));
  }
  return result;
}

}  // namespace

TEST_CASE("Shared non-finite fixture parses as non-finite", "[conformance][non-finite]") {
  const auto fixture_cases = cases();
  REQUIRE(fixture_cases.size() == 3);
  for (const auto& [name, value] : fixture_cases) {
    CAPTURE(name);
    REQUIRE_FALSE(std::isfinite(value));
  }
}

TEST_CASE("Store rejects non-finite public inputs", "[conformance][non-finite]") {
  for (const auto& [name, value] : cases()) {
    CAPTURE(name);
    vanedb::Store store(2);
    const float invalid[2] = {value, 0.0f};
    REQUIRE_THROWS_AS(store.add(1, invalid), std::invalid_argument);
    REQUIRE(store.size() == 0);

    const float finite[2] = {0.0f, 0.0f};
    store.add(2, finite);
    REQUIRE_THROWS_AS(store.search(invalid, 1), std::invalid_argument);
    REQUIRE_THROWS_AS(store.update(2, invalid), std::invalid_argument);
    REQUIRE(store.get_copy(2) == std::vector<float>{0.0f, 0.0f});
  }
}

TEST_CASE("Index rejects non-finite public inputs", "[conformance][non-finite]") {
  for (const auto& [name, value] : cases()) {
    CAPTURE(name);
    vanedb::Index index(2, vanedb::Metric::L2, 4);
    const float invalid[2] = {value, 0.0f};
    REQUIRE_THROWS_AS(index.add(1, invalid), std::invalid_argument);
    REQUIRE(index.size() == 0);

    const float finite[2] = {0.0f, 0.0f};
    index.add(2, finite);
    REQUIRE_THROWS_AS(index.search(invalid, 1), std::invalid_argument);
  }
}

TEST_CASE("MMap builder rejects non-finite vectors", "[conformance][non-finite]") {
  for (const auto& [name, value] : cases()) {
    CAPTURE(name);
    vanedb::DiskStoreBuilder builder(2);
    const float invalid[2] = {value, 0.0f};
    REQUIRE_THROWS_AS(builder.add(1, invalid), std::invalid_argument);
    REQUIRE(builder.size() == 0);
  }
}

TEST_CASE("Finite results sort before non-finite results", "[conformance][non-finite]") {
  for (const auto& [name, value] : cases()) {
    CAPTURE(name);
    std::vector<vanedb::SearchResult> results = {{1, value}, {2, 0.0f}};
    std::sort(results.begin(), results.end());
    REQUIRE(results.front().id == 2);
  }
}

TEST_CASE("Finite exact match outranks overflowed distance", "[conformance][non-finite]") {
  vanedb::Store store(2);
  const float huge[2] = {std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
  const float exact[2] = {0.0f, 0.0f};
  store.add(1, huge);
  store.add(2, exact);

  const auto results = store.search(exact, 2);
  REQUIRE(results[0].id == 2);
  REQUIRE(results[0].distance == 0.0f);
  REQUIRE_FALSE(std::isfinite(results[1].distance));
}
