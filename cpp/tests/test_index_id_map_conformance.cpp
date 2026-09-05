// Cross-engine HNSW id_map cases from conformance/index_id_map_consistency.tsv.
//
// The loader accepted an id_map whose size was <= count and whose values were
// in range, without checking that each key mapped back to its own slot. A file
// could therefore resolve an external id to another slot's vector — the right
// bytes under the wrong identity (vanedb#42 / vanedb-cpp#38).

#include "core/index.h"

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
  REQUIRE(result.size() >= 6);
  return result;
}

// Byte offset of the id_map block in a v3 file, derived from the field order
// in save(): three uint32, seven size_t, one double, one int, then the three
// length-prefixed arrays.
size_t id_map_offset(size_t count, size_t dim) {
  const size_t header = 3 * sizeof(uint32_t) + 7 * sizeof(size_t) + sizeof(double) + sizeof(int);
  const size_t vectors = sizeof(size_t) + count * dim * sizeof(float);
  const size_t ext_ids = sizeof(size_t) + count * sizeof(uint64_t);
  const size_t levels = sizeof(size_t) + count * sizeof(int);
  return header + vectors + ext_ids + levels;
}

template <typename T>
void append(std::vector<char>& out, const T& value) {
  const char* raw = reinterpret_cast<const char*>(&value);
  out.insert(out.end(), raw, raw + sizeof(T));
}

std::vector<char> read_all(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  REQUIRE(in.is_open());
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

/// Save a genuine index, then replace only its id_map block with the crafted
/// one. Everything else stays exactly what the real writer produced.
std::filesystem::path craft(const std::filesystem::path& dir, const Case& test_case, size_t dim) {
  const auto valid = dir / (test_case.name + "-valid.idx");
  {
    vanedb::Index index(dim, vanedb::Metric::L2, std::max<size_t>(test_case.count, 1), 2, 10);
    for (size_t i = 0; i < test_case.count; ++i) {
      const std::vector<float> vector(dim, static_cast<float>(i));
      // Placeholder ids only: the crafted ext_ids are written over these
      // below, and a fixture case may deliberately repeat an external id.
      index.add(1000 + i, vector.data());
    }
    index.save(valid.string());
  }

  auto bytes = read_all(valid);
  const size_t offset = id_map_offset(test_case.count, dim);
  REQUIRE(bytes.size() > offset + sizeof(size_t));

  size_t existing = 0;
  std::memcpy(&existing, bytes.data() + offset, sizeof(size_t));
  const size_t block = sizeof(size_t) + existing * (sizeof(uint64_t) + sizeof(size_t));
  REQUIRE(bytes.size() >= offset + block);

  std::vector<char> crafted(bytes.begin(), bytes.begin() + static_cast<long>(offset));
  append(crafted, test_case.id_map.size());
  for (const auto& [key, value] : test_case.id_map) {
    append(crafted, key);
    append(crafted, value);
  }
  // Overwrite the live ext_ids so the crafted map is evaluated against the
  // identifiers the fixture names.
  const size_t ext_ids_at = 3 * sizeof(uint32_t) + 7 * sizeof(size_t) + sizeof(double) + sizeof(int)
                            + sizeof(size_t) + test_case.count * dim * sizeof(float) + sizeof(size_t);
  for (size_t i = 0; i < test_case.count; ++i) {
    std::memcpy(crafted.data() + ext_ids_at + i * sizeof(uint64_t), &test_case.ext_ids[i],
                sizeof(uint64_t));
  }
  crafted.insert(crafted.end(), bytes.begin() + static_cast<long>(offset + block), bytes.end());

  const auto path = dir / (test_case.name + ".idx");
  std::ofstream out(path, std::ios::binary);
  out.write(crafted.data(), static_cast<long>(crafted.size()));
  out.close();
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
      REQUIRE_NOTHROW(vanedb::Index::load(path.string()));
    } else {
      REQUIRE_THROWS_AS(vanedb::Index::load(path.string()), std::runtime_error);
    }
  }

  std::filesystem::remove_all(dir);
}
