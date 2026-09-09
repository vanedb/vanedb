// Cross-engine HNSW size cases from vanedb/tests/fixtures/conformance/index_derived_sizes.tsv.
#include "core/approx_index.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>

#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
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
  std::ifstream fixture(std::string(VANEDB_CONFORMANCE_DIR) + "/index_derived_sizes.tsv");
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

void write_header(const std::filesystem::path& path, const Case& test_case) {
  std::ofstream file(path, std::ios::binary);
  REQUIRE(file.is_open());
  vanedb::detail::write_bin(file, vanedb::ApproxIndex::MAGIC);
  vanedb::detail::write_bin(file, vanedb::ApproxIndex::VERSION);
  vanedb::detail::write_bin(file, test_case.dimension);
  vanedb::detail::write_bin(file, uint32_t{0});
  vanedb::detail::write_bin(file, test_case.max_elements);
  vanedb::detail::write_bin(file, test_case.M);
  vanedb::detail::write_bin(file, size_t{200});
  vanedb::detail::write_bin(file, size_t{50});
  vanedb::detail::write_bin(file, double{1.0});
}

void append_count(const std::filesystem::path& path, size_t count) {
  std::ofstream file(path, std::ios::binary | std::ios::app);
  REQUIRE(file.is_open());
  vanedb::detail::write_bin(file, count);
}

}  // namespace

TEST_CASE("HNSW construction rejects shared derived-size overflows",
          "[conformance][index][sizes]") {
  const auto fixture_cases = cases();
  REQUIRE(fixture_cases.size() == 2);
  for (const auto& test_case : fixture_cases) {
    CAPTURE(test_case.name);
    REQUIRE_THROWS_AS(
        vanedb::ApproxIndex(test_case.dimension, vanedb::Metric::L2,
                         test_case.max_elements, test_case.M),
        std::invalid_argument);
    REQUIRE_THROWS_WITH(
        vanedb::ApproxIndex(test_case.dimension, vanedb::Metric::L2,
                         test_case.max_elements, test_case.M),
        direct_error(test_case));
  }
}

TEST_CASE("HNSW load rejects derived-size overflows before allocation",
          "[conformance][index][sizes][persistence]") {
  for (const auto& test_case : cases()) {
    CAPTURE(test_case.name);
    const auto path = std::filesystem::path("test_hnsw_" + test_case.name + ".bin");
    write_header(path, test_case);
    REQUIRE_THROWS_AS(vanedb::ApproxIndex::load(path.string()), std::runtime_error);
    REQUIRE_THROWS_WITH(vanedb::ApproxIndex::load(path.string()), load_error(test_case));
    std::filesystem::remove(path);
  }
}

TEST_CASE("HNSW load rejects live-vector size overflow before allocation",
          "[conformance][index][sizes][persistence]") {
  const auto path = std::filesystem::path("test_hnsw_count_times_dimension.bin");
  const Case safe_header{"count_times_dimension", 2, 1, 2, "count_times_dimension"};
  write_header(path, safe_header);
  append_count(path, std::numeric_limits<size_t>::max());

  REQUIRE_THROWS_AS(vanedb::ApproxIndex::load(path.string()), std::runtime_error);
  REQUIRE_THROWS_WITH(vanedb::ApproxIndex::load(path.string()),
                      "Corrupted file: count * dimension overflow");
  std::filesystem::remove(path);
}

// A header alone decides what the constructor allocates: `checked_persisted_sizes`
// ran the overflow checks and then `max_elements * dimension` floats were
// reserved before a byte of payload was read. Overflow was the only bound, so a
// product that merely happened to fit in `size_t` — 1e6 x 1e8 is 1e14 floats,
// 400 TB — was accepted and handed straight to `resize`.
//
// Nothing legitimate is lost by capping it. `read_vec` already refuses to read
// more than MAX_VEC_SIZE elements into any array, and the loader then requires
// the array it read to equal this exact product, so a file above the cap could
// never have finished loading. The cap only moves the rejection to before the
// allocation instead of after it.
TEST_CASE("HNSW load caps header-declared allocation before reserving it",
          "[conformance][index][sizes][persistence]") {
  const size_t cap = vanedb::detail::MAX_VEC_SIZE;
  const auto pid = std::to_string(
      static_cast<long long>(std::hash<std::thread::id>{}(std::this_thread::get_id())));

  SECTION("max_elements alone above the cap") {
    const auto path = std::filesystem::path("test_hnsw_cap_elements_" + pid + ".bin");
    write_header(path, Case{"cap_elements", 2, cap + 1, 2, ""});
    REQUIRE_THROWS_WITH(vanedb::ApproxIndex::load(path.string()),
                        "Corrupted file: declared size exceeds the element cap");
    std::filesystem::remove(path);
  }

  SECTION("max_elements within the cap but the product above it") {
    const auto path = std::filesystem::path("test_hnsw_cap_product_" + pid + ".bin");
    write_header(path, Case{"cap_product", 2, cap, 2, ""});
    REQUIRE_THROWS_WITH(vanedb::ApproxIndex::load(path.string()),
                        "Corrupted file: declared size exceeds the element cap");
    std::filesystem::remove(path);
  }
}
