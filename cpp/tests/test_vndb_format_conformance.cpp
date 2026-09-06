// The VNDB v1 on-disk contract, anchored to golden fixtures.
//
// vanedb/tests/fixtures/conformance/vndb/*.vndb are written from the specification by generate.py,
// not by either engine. The Rust and C++ engines are otherwise only ever
// compared to each other, so a layout change applied to both would pass every
// test in either repo. These fixtures are the independent anchor, and the
// Rust suite asserts the same bytes.

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "core/disk_index.h"

namespace {

constexpr size_t DIM = 4;
const std::vector<uint64_t> IDS = {10, 20, 30, 40, 50, 60};

std::string fixture(const std::string& name) {
  return std::string(VANEDB_CONFORMANCE_DIR) + "/vndb/" + name;
}

// Mirrors generate.py: row i, component d is (i * DIM + d) / 8.
std::vector<float> row(size_t i) {
  std::vector<float> v(DIM);
  for (size_t d = 0; d < DIM; ++d) {
    v[d] = static_cast<float>(i * DIM + d) / 8.0f;
  }
  return v;
}

struct Case {
  const char* file;
  vanedb::Metric metric;
};

const std::vector<Case> CASES = {
    {"v1_l2.vndb", vanedb::Metric::L2},
    {"v1_cosine.vndb", vanedb::Metric::COSINE},
    {"v1_dot.vndb", vanedb::Metric::DOT},
};

}  // namespace

TEST_CASE("every VNDB metric fixture loads with its contents intact", "[vndb]") {
  for (const auto& c : CASES) {
    vanedb::DiskIndex index(fixture(c.file));
    INFO("fixture " << c.file);
    REQUIRE(index.dimension() == DIM);
    REQUIRE(index.size() == IDS.size());
    REQUIRE(index.metric() == c.metric);

    for (size_t i = 0; i < IDS.size(); ++i) {
      REQUIRE(index.contains(IDS[i]));
      const float* got = index.get(IDS[i]);
      REQUIRE(got != nullptr);
      const std::vector<float> want = row(i);
      for (size_t d = 0; d < DIM; ++d) {
        REQUIRE(got[d] == want[d]);
      }
    }
  }
}

TEST_CASE("writing the same content reproduces the VNDB fixture byte for byte",
          "[vndb]") {
  for (const auto& c : CASES) {
    INFO("fixture " << c.file);
    const std::string out = std::string(c.file) + ".written";
    {
      vanedb::DiskIndexBuilder builder(DIM, c.metric);
      for (size_t i = 0; i < IDS.size(); ++i) {
        builder.add(IDS[i], row(i).data());
      }
      builder.save(out);
    }

    std::ifstream ours(out, std::ios::binary);
    std::ifstream golden(fixture(c.file), std::ios::binary);
    REQUIRE(ours.good());
    REQUIRE(golden.good());
    const std::vector<char> a((std::istreambuf_iterator<char>(ours)),
                              std::istreambuf_iterator<char>());
    const std::vector<char> b((std::istreambuf_iterator<char>(golden)),
                              std::istreambuf_iterator<char>());
    REQUIRE(a.size() == b.size());
    for (size_t i = 0; i < a.size(); ++i) {
      INFO("byte offset " << i);
      REQUIRE(a[i] == b[i]);
    }
    std::remove(out.c_str());
  }
}
