// The VNDB v1 on-disk contract, anchored to golden fixtures.
//
// vanedb/tests/fixtures/conformance/vndb/*.vndb are written from the specification by generate.py,
// not by either engine. The Rust and C++ engines are otherwise only ever
// compared to each other, so a layout change applied to both would pass every
// test in either repo. These fixtures are the independent anchor, and the
// Rust suite asserts the same bytes.

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "core/detail/file_utils.h"
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

TEST_CASE("a header that disagrees with the file length is rejected in both directions",
          "[vndb]") {
  // `expected` is derived from the header, so a one-sided `file_size_ <
  // expected` check lets a header that UNDERSTATES the geometry move the
  // goalpost rather than trip the guard: the payload is then read at the wrong
  // stride and `get` returns a vector straddling two stored records.
  //
  // `dim` (offsets 8..16) and `num_vectors` (16..24) are the only fields the
  // length is computed from. Every flip in them makes the header disagree with
  // the file, so none may be accepted. The Rust reader asserts the same thing
  // over the same fixture layout (`corruption_tests.rs`), which is the point:
  // both readers of VNDB v1 must reject exactly the same shapes.
  std::ifstream in(fixture("v1_l2.vndb"), std::ios::binary);
  REQUIRE(in);
  const std::vector<char> original((std::istreambuf_iterator<char>(in)),
                                   std::istreambuf_iterator<char>());
  REQUIRE(original.size() == 32 + IDS.size() * 8 + IDS.size() * DIM * 4);

  // The system temp directory, not the fixture tree. `temp_path_for` keeps a
  // file beside its destination so a save can rename same-filesystem; there is
  // no rename here, and writing into the committed fixture directory makes a
  // read-only checkout fail and leaves .tmp files behind on an aborted run.
  const std::string path = vanedb::detail::temp_path_for(
      (std::filesystem::temp_directory_path() / "vndb_geometry_flip").string());

  std::vector<std::string> accepted;
  for (size_t byte = 8; byte < 24; ++byte) {
    for (int bit = 0; bit < 8; ++bit) {
      std::vector<char> bytes = original;
      bytes[byte] = static_cast<char>(bytes[byte] ^ (1 << bit));
      {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        REQUIRE(out);
        out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
      }
      try {
        vanedb::DiskIndex index(path);
        accepted.push_back("byte " + std::to_string(byte) + " bit " +
                           std::to_string(bit) + " -> dim " +
                           std::to_string(index.dimension()) + " size " +
                           std::to_string(index.size()));
      } catch (const std::exception&) {
        // Rejected, as required.
      }
      std::remove(path.c_str());
    }
  }

  // Not INFO in a loop: a Catch2 scoped message is destroyed at the end of the
  // iteration that created it, so the list would be gone before REQUIRE runs
  // and a regression would report only "false". Build the diagnostic into the
  // assertion itself, as the Rust twin does.
  std::string detail;
  for (const auto& a : accepted) {
    detail += "\n  " + a;
  }
  INFO("accepted " << accepted.size() << " flip(s):" << detail);
  REQUIRE(accepted.empty());
}
