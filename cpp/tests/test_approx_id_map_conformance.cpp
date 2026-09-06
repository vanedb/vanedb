// Cross-engine HNSW id_map cases from vanedb/tests/fixtures/conformance/index_id_map_consistency.tsv.
//
// The loader accepted an id_map whose size was <= count and whose values were
// in range, without checking that each key mapped back to its own slot. A file
// could therefore resolve an external id to another slot's vector — the right
// bytes under the wrong identity (vanedb#42 / vanedb-cpp#38).

#include "core/approx_index.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

struct Case {
  std::string name;
  size_t count;
  std::vector<uint64_t> ext_ids;
  std::vector<std::pair<uint64_t, size_t>> id_map;
  bool accept;
};

std::vector<std::string> split(const std::string& line, char sep) {
  std::vector<std::string> out;
  size_t start = 0;
  for (;;) {
    const size_t at = line.find(sep, start);
    out.push_back(line.substr(start, at - start));
    if (at == std::string::npos) return out;
    start = at + 1;
  }
}

std::vector<Case> cases() {
  std::ifstream fixture(std::string(VANEDB_CONFORMANCE_DIR) + "/index_id_map_consistency.tsv");
  REQUIRE(fixture.is_open());

  std::vector<Case> result;
  for (std::string line; std::getline(fixture, line);) {
    if (line.empty() || line[0] == '#') continue;
    const auto fields = split(line, '\t');
    REQUIRE(fields.size() == 5);

    Case test_case;
    test_case.name = fields[0];
    test_case.count = static_cast<size_t>(std::stoull(fields[1]));
    for (const auto& id : split(fields[2], ',')) test_case.ext_ids.push_back(std::stoull(id));
    if (fields[3] != "-") {
      for (const auto& pair : split(fields[3], ',')) {
        const auto kv = split(pair, ':');
        REQUIRE(kv.size() == 2);
        test_case.id_map.emplace_back(std::stoull(kv[0]), static_cast<size_t>(std::stoull(kv[1])));
      }
    }
    test_case.accept = fields[4] == "accept";
    result.push_back(std::move(test_case));
  }
  // Two cases left the shared fixture when the Rust engine gained deletion:
  // a slot absent from id_map is a tombstone there, not corruption.
  REQUIRE(result.size() >= 5);
  return result;
}

// Independently encode legacy v1: VNDB derives its live map from node flags,
// but the old reader must still reject corrupt persisted maps.
std::filesystem::path craft(const std::filesystem::path& dir, const Case& test_case, size_t dim) {
  using vanedb::detail::write_bin;
  using vanedb::detail::write_vec;
  const auto path = dir / (test_case.name + ".idx");
  std::ofstream out(path, std::ios::binary);
  const size_t capacity = std::max<size_t>(test_case.count, 1);
  write_bin(out, vanedb::ApproxIndex::MAGIC);
  write_bin(out, uint32_t{1});
  write_bin(out, dim);
  write_bin(out, uint32_t{0});
  write_bin(out, capacity);
  write_bin(out, size_t{2});
  write_bin(out, size_t{10});
  write_bin(out, size_t{10});
  write_bin(out, double{1.0});
  write_bin(out, test_case.count);
  write_bin(out, test_case.count ? size_t{0} : vanedb::ApproxIndex::INVALID_ID);
  write_bin(out, test_case.count ? int{0} : int{-1});
  write_vec(out, std::vector<float>(capacity * dim, 1.0f));
  auto ids = test_case.ext_ids;
  ids.resize(capacity);
  write_vec(out, ids);
  write_vec(out, std::vector<int>(capacity, 0));
  write_bin(out, test_case.id_map.size());
  for (const auto& [key, value] : test_case.id_map) {
    write_bin(out, key);
    write_bin(out, value);
  }
  write_bin(out, capacity);
  for (size_t slot = 0; slot < capacity; ++slot) {
    write_bin(out, slot < test_case.count ? size_t{1} : size_t{0});
    if (slot < test_case.count) write_vec(out, std::vector<size_t>{});
  }
  return path;
}

}  // namespace

TEST_CASE("HNSW loader enforces the shared id_map contract", "[conformance][persistence]") {
  const auto dir = std::filesystem::temp_directory_path() / "vanedb-id-map-conformance";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  constexpr size_t DIM = 2;

  for (const auto& test_case : cases()) {
    const auto path = craft(dir, test_case, DIM);
    INFO("case=" << test_case.name);
    if (test_case.accept) {
      REQUIRE_NOTHROW(vanedb::ApproxIndex::load(path.string()));
    } else {
      REQUIRE_THROWS_AS(vanedb::ApproxIndex::load(path.string()), std::runtime_error);
    }
  }

  std::filesystem::remove_all(dir);
}
