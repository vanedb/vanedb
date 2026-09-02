// Cross-engine HNSW size cases from conformance/hnsw_derived_sizes.tsv.
#include "core/hnsw_index.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>

#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Case {
  std::string name;
  size_t dimension;
  size_t max_elements;
  size_t M;
  std::string overflow;
};

size_t parse_size(const std::string& value) {
  if (value == "SIZE_MAX") return std::numeric_limits<size_t>::max();
  if (value == "HALF_SIZE_MAX_PLUS_ONE")
    return std::numeric_limits<size_t>::max() / 2 + 1;
  return static_cast<size_t>(std::stoull(value));
}

std::vector<Case> cases() {
  std::ifstream fixture(std::string(VANEDB_CONFORMANCE_DIR) + "/hnsw_derived_sizes.tsv");
  REQUIRE(fixture.is_open());

  std::vector<Case> result;
  for (std::string line; std::getline(fixture, line);) {
    if (line.empty() || line[0] == '#') continue;
    std::vector<std::string> fields;
    size_t start = 0;
    for (size_t tab = line.find('\t');; tab = line.find('\t', start)) {
      fields.push_back(line.substr(start, tab - start));
      if (tab == std::string::npos) break;
      start = tab + 1;
    }
    REQUIRE(fields.size() == 5);
    result.push_back(
        {fields[0], parse_size(fields[1]), parse_size(fields[2]), parse_size(fields[3]), fields[4]});
  }
  return result;
}

std::string direct_error(const Case& test_case) {
  if (test_case.overflow == "capacity_times_dimension")
    return "max_elements * dimension overflow";
  if (test_case.overflow == "m_times_two") return "M * 2 overflow";
  FAIL("Unknown overflow kind: " << test_case.overflow);
}

std::string load_error(const Case& test_case) {
  return "Corrupted file: " + direct_error(test_case);
}

void write_overflowing_header(const std::filesystem::path& path, const Case& test_case) {
  std::ofstream file(path, std::ios::binary);
  REQUIRE(file.is_open());
  vanedb::detail::write_bin(file, vanedb::HNSWIndex::MAGIC);
  vanedb::detail::write_bin(file, vanedb::HNSWIndex::VERSION);
  vanedb::detail::write_bin(file, test_case.dimension);
  vanedb::detail::write_bin(file, uint32_t{0});
  vanedb::detail::write_bin(file, test_case.max_elements);
  vanedb::detail::write_bin(file, test_case.M);
  vanedb::detail::write_bin(file, size_t{200});
  vanedb::detail::write_bin(file, size_t{50});
  vanedb::detail::write_bin(file, double{1.0});
}

}  // namespace

TEST_CASE("HNSW construction rejects shared derived-size overflows",
          "[conformance][hnsw][sizes]") {
  const auto fixture_cases = cases();
  REQUIRE(fixture_cases.size() == 2);
  for (const auto& test_case : fixture_cases) {
    CAPTURE(test_case.name);
    REQUIRE_THROWS_AS(
        vanedb::HNSWIndex(test_case.dimension, vanedb::DistanceMetric::L2,
                         test_case.max_elements, test_case.M),
        std::invalid_argument);
    REQUIRE_THROWS_WITH(
        vanedb::HNSWIndex(test_case.dimension, vanedb::DistanceMetric::L2,
                         test_case.max_elements, test_case.M),
        direct_error(test_case));
  }
}

TEST_CASE("HNSW load rejects derived-size overflows before allocation",
          "[conformance][hnsw][sizes][persistence]") {
  for (const auto& test_case : cases()) {
    CAPTURE(test_case.name);
    const auto path = std::filesystem::path("test_hnsw_" + test_case.name + ".bin");
    write_overflowing_header(path, test_case);
    REQUIRE_THROWS_AS(vanedb::HNSWIndex::load(path.string()), std::runtime_error);
    REQUIRE_THROWS_WITH(vanedb::HNSWIndex::load(path.string()), load_error(test_case));
    std::filesystem::remove(path);
  }
}
