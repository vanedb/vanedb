#include "core/disk_index.h"
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <filesystem>
#include <random>
#include <thread>
#include <vector>

using Catch::Approx;

TEST_CASE("DiskIndexBuilder - construction", "[disk][builder]") {
  SECTION("Valid dimension") {
    REQUIRE_NOTHROW(vanedb::DiskIndexBuilder(768));
    REQUIRE_NOTHROW(vanedb::DiskIndexBuilder(128, vanedb::Metric::COSINE));
  }

  SECTION("Zero dimension throws") {
    REQUIRE_THROWS_AS(vanedb::DiskIndexBuilder(0), std::invalid_argument);
  }
}

TEST_CASE("DiskIndexBuilder - add vectors", "[disk][builder]") {
  vanedb::DiskIndexBuilder builder(3);

  SECTION("Add single vector") {
    float vec[] = {1.0f, 2.0f, 3.0f};
    REQUIRE_NOTHROW(builder.add(1, vec));
    REQUIRE(builder.size() == 1);
  }

  SECTION("Add multiple vectors") {
    float vec1[] = {1.0f, 2.0f, 3.0f};
    float vec2[] = {4.0f, 5.0f, 6.0f};

    builder.add(1, vec1);
    builder.add(2, vec2);

    REQUIRE(builder.size() == 2);
  }

  SECTION("Null vector throws") {
    REQUIRE_THROWS_AS(builder.add(1, nullptr), std::invalid_argument);
  }

  SECTION("Duplicate ID throws") {
    float vec1[] = {1.0f, 2.0f, 3.0f};
    float vec2[] = {4.0f, 5.0f, 6.0f};

    builder.add(1, vec1);
    REQUIRE_THROWS_AS(builder.add(1, vec2), std::invalid_argument);
  }
}

TEST_CASE("DiskIndex - save and load", "[disk]") {
  const std::string filename = "test_mmap_store.bin";

  // Cleanup before test
  std::filesystem::remove(filename);

  constexpr size_t dim = 4;
  vanedb::DiskIndexBuilder builder(dim, vanedb::Metric::L2);

  // Add test vectors
  float vec1[] = {1.0f, 0.0f, 0.0f, 0.0f};
  float vec2[] = {0.0f, 1.0f, 0.0f, 0.0f};
  float vec3[] = {0.0f, 0.0f, 1.0f, 0.0f};

  builder.add(10, vec1);
  builder.add(20, vec2);
  builder.add(30, vec3);

  SECTION("Save and load successfully") {
    REQUIRE_NOTHROW(builder.save(filename));

    vanedb::DiskIndex store(filename);

    REQUIRE(store.size() == 3);
    REQUIRE(store.dimension() == dim);
    REQUIRE(store.metric() == vanedb::Metric::L2);
  }

  SECTION("Get vector by ID") {
    builder.save(filename);
    vanedb::DiskIndex store(filename);

    const float* retrieved = store.get(10);
    REQUIRE(retrieved != nullptr);
    REQUIRE(retrieved[0] == 1.0f);
    REQUIRE(retrieved[1] == 0.0f);
    REQUIRE(retrieved[2] == 0.0f);
    REQUIRE(retrieved[3] == 0.0f);
  }

  SECTION("Get non-existent ID returns nullptr") {
    builder.save(filename);
    vanedb::DiskIndex store(filename);

    REQUIRE(store.get(999) == nullptr);
  }

  SECTION("Contains works correctly") {
    builder.save(filename);
    vanedb::DiskIndex store(filename);

    REQUIRE(store.contains(10));
    REQUIRE(store.contains(20));
    REQUIRE(store.contains(30));
    REQUIRE_FALSE(store.contains(999));
  }

  SECTION("Search finds nearest neighbors") {
    builder.save(filename);
    vanedb::DiskIndex store(filename);

    float query[] = {0.9f, 0.0f, 0.0f, 0.0f};
    auto results = store.search(query, 1);

    REQUIRE(results.size() == 1);
    REQUIRE(results[0].id == 10);  // Closest to vec1
  }

  SECTION("Search with k > size returns all") {
    builder.save(filename);
    vanedb::DiskIndex store(filename);

    float query[] = {0.0f, 0.0f, 0.0f, 0.0f};
    auto results = store.search(query, 100);

    REQUIRE(results.size() == 3);
  }

  // Cleanup
  std::filesystem::remove(filename);
}

/// Writes a complete, self-consistent `VNDB` file so that a crafted defect is
/// the only thing a loader can reject.
static void write_vndb_file(const std::string& path, uint32_t magic, uint64_t dim,
                            const std::vector<uint64_t>& ids,
                            const std::vector<float>& vectors) {
  std::ofstream ofs(path, std::ios::binary);
  uint32_t version = vanedb::DiskIndex::VERSION;
  uint64_t n = ids.size();
  uint32_t metric = 0, reserved = 0;
  ofs.write(reinterpret_cast<const char*>(&magic), 4);
  ofs.write(reinterpret_cast<const char*>(&version), 4);
  ofs.write(reinterpret_cast<const char*>(&dim), 8);
  ofs.write(reinterpret_cast<const char*>(&n), 8);
  ofs.write(reinterpret_cast<const char*>(&metric), 4);
  ofs.write(reinterpret_cast<const char*>(&reserved), 4);
  for (uint64_t id : ids) ofs.write(reinterpret_cast<const char*>(&id), 8);
  for (float v : vectors) ofs.write(reinterpret_cast<const char*>(&v), 4);
}

TEST_CASE("DiskIndex - error handling", "[disk]") {
  SECTION("Non-existent file throws") {
    REQUIRE_THROWS_AS(vanedb::DiskIndex("nonexistent_file.bin"), std::runtime_error);
  }

  SECTION("Invalid magic throws") {
    // A complete, self-consistent file whose only defect is the magic. A
    // 4-byte file is rejected by the size check before the magic is compared,
    // which makes the magic guard look tested when it never runs.
    const std::string filename = "test_bad_magic.bin";
    write_vndb_file(filename, 0xDEADBEEF, 2, {1, 2}, {1.0f, 0.0f, 0.0f, 1.0f});
    REQUIRE_THROWS_AS(vanedb::DiskIndex(filename), std::runtime_error);
    std::filesystem::remove(filename);
  }

  SECTION("Duplicate ids throw") {
    // Neither builder writes duplicates, but VNDB is the shared cross-engine
    // format, so the loader is what has to reject them. Accepting them made
    // size() overcount, get() return a row the id does not name, and search()
    // emit one id twice.
    const std::string filename = "test_duplicate_ids.bin";
    write_vndb_file(filename, vanedb::DiskIndex::MAGIC, 2, {7, 7, 9},
                    {1.0f, 0.0f, 0.0f, 1.0f, 5.0f, 4.0f});
    REQUIRE_THROWS_AS(vanedb::DiskIndex(filename), std::runtime_error);
    std::filesystem::remove(filename);
  }

  SECTION("Nonzero reserved header bytes throw") {
    const std::string filename = "test_reserved_header.bin";
    write_vndb_file(filename, vanedb::DiskIndex::MAGIC, 2, {7}, {1.0f, 0.0f});
    {
      std::fstream file(filename, std::ios::binary | std::ios::in | std::ios::out);
      file.seekp(28);
      const uint32_t reserved = 1;
      file.write(reinterpret_cast<const char*>(&reserved), sizeof(reserved));
    }
    REQUIRE_THROWS_AS(vanedb::DiskIndex(filename), std::runtime_error);
    std::filesystem::remove(filename);
  }

  SECTION("Truncated file throws") {
    const std::string filename = "test_truncated.bin";
    {
      std::ofstream ofs(filename, std::ios::binary);
      // Write only partial header
      uint32_t magic = vanedb::DiskIndex::MAGIC;
      ofs.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    }
    REQUIRE_THROWS_AS(vanedb::DiskIndex(filename), std::runtime_error);
    std::filesystem::remove(filename);
  }

  SECTION("Invalid metric value throws") {
    const std::string filename = "test_bad_metric.bin";
    {
      std::ofstream ofs(filename, std::ios::binary);
      uint32_t magic = vanedb::DiskIndex::MAGIC;
      uint32_t version = 1;
      uint64_t dim = 3;  // Fixed: was uint32_t, should be uint64_t per file format
      uint64_t num_vectors = 0;
      uint32_t bad_metric = 999; // Invalid metric value
      ofs.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
      ofs.write(reinterpret_cast<const char*>(&version), sizeof(version));
      ofs.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
      ofs.write(reinterpret_cast<const char*>(&num_vectors), sizeof(num_vectors));
      ofs.write(reinterpret_cast<const char*>(&bad_metric), sizeof(bad_metric));
    }
    REQUIRE_THROWS_AS(vanedb::DiskIndex(filename), std::runtime_error);
    std::filesystem::remove(filename);
  }

  SECTION("Unsupported version throws") {
    const std::string filename = "test_bad_version.bin";
    {
      std::ofstream ofs(filename, std::ios::binary);
      uint32_t magic = vanedb::DiskIndex::MAGIC;
      uint32_t version = 99; // Unsupported version
      uint64_t dim = 3;
      uint64_t num_vectors = 0;
      uint32_t metric = 0;
      uint32_t reserved = 0;
      ofs.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
      ofs.write(reinterpret_cast<const char*>(&version), sizeof(version));
      ofs.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
      ofs.write(reinterpret_cast<const char*>(&num_vectors), sizeof(num_vectors));
      ofs.write(reinterpret_cast<const char*>(&metric), sizeof(metric));
      ofs.write(reinterpret_cast<const char*>(&reserved), sizeof(reserved));
    }
    REQUIRE_THROWS_AS(vanedb::DiskIndex(filename), std::runtime_error);
    std::filesystem::remove(filename);
  }

  SECTION("Zero dimension with vectors throws") {
    // Prevents division by zero in overflow check (dim=0 makes vec_bytes_per=0)
    const std::string filename = "test_zero_dim.bin";
    {
      std::ofstream ofs(filename, std::ios::binary);
      uint32_t magic = vanedb::DiskIndex::MAGIC;
      uint32_t version = 1;
      uint64_t dim = 0; // Zero dimension
      uint64_t num_vectors = 1; // Non-zero vectors
      uint32_t metric = 0;
      uint32_t reserved = 0;
      ofs.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
      ofs.write(reinterpret_cast<const char*>(&version), sizeof(version));
      ofs.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
      ofs.write(reinterpret_cast<const char*>(&num_vectors), sizeof(num_vectors));
      ofs.write(reinterpret_cast<const char*>(&metric), sizeof(metric));
      ofs.write(reinterpret_cast<const char*>(&reserved), sizeof(reserved));
    }
    REQUIRE_THROWS_AS(vanedb::DiskIndex(filename), std::runtime_error);
    std::filesystem::remove(filename);
  }

  SECTION("File truncated mid-data throws") {
    const std::string filename = "test_truncated_data.bin";
    {
      std::ofstream ofs(filename, std::ios::binary);
      uint32_t magic = vanedb::DiskIndex::MAGIC;
      uint32_t version = 1;
      uint64_t dim = 4;
      uint64_t num_vectors = 10; // Claims 10 vectors but won't provide them
      uint32_t metric = 0;
      uint32_t reserved = 0;
      ofs.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
      ofs.write(reinterpret_cast<const char*>(&version), sizeof(version));
      ofs.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
      ofs.write(reinterpret_cast<const char*>(&num_vectors), sizeof(num_vectors));
      ofs.write(reinterpret_cast<const char*>(&metric), sizeof(metric));
      ofs.write(reinterpret_cast<const char*>(&reserved), sizeof(reserved));
      // Write only one ID and one vector (should have 10)
      uint64_t id = 1;
      float vec[4] = {1.0f, 2.0f, 3.0f, 4.0f};
      ofs.write(reinterpret_cast<const char*>(&id), sizeof(id));
      ofs.write(reinterpret_cast<const char*>(vec), sizeof(vec));
    }
    REQUIRE_THROWS_AS(vanedb::DiskIndex(filename), std::runtime_error);
    std::filesystem::remove(filename);
  }

  SECTION("Size overflow - huge num_vectors throws") {
    const std::string filename = "test_overflow_num.bin";
    {
      std::ofstream ofs(filename, std::ios::binary);
      uint32_t magic = vanedb::DiskIndex::MAGIC;
      uint32_t version = 1;
      uint64_t dim = 4;
      uint64_t num_vectors = SIZE_MAX; // Causes overflow
      uint32_t metric = 0;
      uint32_t reserved = 0;
      ofs.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
      ofs.write(reinterpret_cast<const char*>(&version), sizeof(version));
      ofs.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
      ofs.write(reinterpret_cast<const char*>(&num_vectors), sizeof(num_vectors));
      ofs.write(reinterpret_cast<const char*>(&metric), sizeof(metric));
      ofs.write(reinterpret_cast<const char*>(&reserved), sizeof(reserved));
    }
    REQUIRE_THROWS_AS(vanedb::DiskIndex(filename), std::runtime_error);
    std::filesystem::remove(filename);
  }

  SECTION("Size overflow - huge dimension throws") {
    const std::string filename = "test_overflow_dim.bin";
    {
      std::ofstream ofs(filename, std::ios::binary);
      uint32_t magic = vanedb::DiskIndex::MAGIC;
      uint32_t version = 1;
      uint64_t dim = SIZE_MAX; // Causes overflow
      uint64_t num_vectors = 1;
      uint32_t metric = 0;
      uint32_t reserved = 0;
      ofs.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
      ofs.write(reinterpret_cast<const char*>(&version), sizeof(version));
      ofs.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
      ofs.write(reinterpret_cast<const char*>(&num_vectors), sizeof(num_vectors));
      ofs.write(reinterpret_cast<const char*>(&metric), sizeof(metric));
      ofs.write(reinterpret_cast<const char*>(&reserved), sizeof(reserved));
    }
    REQUIRE_THROWS_AS(vanedb::DiskIndex(filename), std::runtime_error);
    std::filesystem::remove(filename);
  }

  SECTION("Size overflow - combined dim*num_vectors overflow throws") {
    // Test the combined overflow check: num_vectors * dim * sizeof(float) overflows
    // even though each value individually passes earlier checks
    const std::string filename = "test_overflow_combined.bin";
    {
      std::ofstream ofs(filename, std::ios::binary);
      uint32_t magic = vanedb::DiskIndex::MAGIC;
      uint32_t version = 1;
      // dim = SIZE_MAX/8 passes dim check (< SIZE_MAX/sizeof(float))
      // num_vectors = 3 passes num_vectors check (< SIZE_MAX/sizeof(uint64_t))
      // But dim * sizeof(float) * num_vectors = (SIZE_MAX/2) * 3 overflows
      uint64_t dim = SIZE_MAX / 8;
      uint64_t num_vectors = 3;
      uint32_t metric = 0;
      uint32_t reserved = 0;
      ofs.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
      ofs.write(reinterpret_cast<const char*>(&version), sizeof(version));
      ofs.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
      ofs.write(reinterpret_cast<const char*>(&num_vectors), sizeof(num_vectors));
      ofs.write(reinterpret_cast<const char*>(&metric), sizeof(metric));
      ofs.write(reinterpret_cast<const char*>(&reserved), sizeof(reserved));
    }
    REQUIRE_THROWS_AS(vanedb::DiskIndex(filename), std::runtime_error);
    std::filesystem::remove(filename);
  }
}

TEST_CASE("DiskIndex - search validation", "[disk]") {
  const std::string filename = "test_mmap_search_validation.bin";
  std::filesystem::remove(filename);

  vanedb::DiskIndexBuilder builder(4, vanedb::Metric::L2);
  float vec[] = {1.0f, 0.0f, 0.0f, 0.0f};
  builder.add(1, vec);
  builder.save(filename);

  {
    vanedb::DiskIndex store(filename);

    SECTION("Search with null query throws") {
      REQUIRE_THROWS_AS(store.search(nullptr, 1), std::invalid_argument);
    }

    SECTION("Search with k=0 throws") {
      float query[] = {1.0f, 0.0f, 0.0f, 0.0f};
      REQUIRE_THROWS_AS(store.search(query, 0), std::invalid_argument);
    }

    SECTION("Search with a non-finite query throws") {
      float query[] = {std::numeric_limits<float>::quiet_NaN(), 0.0f, 0.0f, 0.0f};
      REQUIRE_THROWS_AS(store.search(query, 1), std::invalid_argument);
    }
  }  // Scope ensures store is destroyed and file unmapped before removal (Windows file locking)

  std::filesystem::remove(filename);
}

TEST_CASE("DiskIndex - rejects non-finite stored vectors", "[disk][persistence]") {
  const std::string filename = "test_mmap_non_finite.bin";
  vanedb::DiskIndexBuilder builder(2);
  const float vector[] = {0.0f, 0.0f};
  builder.add(1, vector);
  builder.save(filename);

  {
    std::fstream file(filename, std::ios::binary | std::ios::in | std::ios::out);
    const auto vector_offset = static_cast<std::streamoff>(
        vanedb::DiskIndex::HEADER_SIZE + sizeof(uint64_t));
    file.seekp(vector_offset);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    file.write(reinterpret_cast<const char*>(&nan), sizeof(nan));
  }

  REQUIRE_THROWS_AS(vanedb::DiskIndex(filename), std::runtime_error);
  std::filesystem::remove(filename);
}

TEST_CASE("DiskIndex - large scale", "[disk][stress]") {
  const std::string filename = "test_mmap_large.bin";
  std::filesystem::remove(filename);

  constexpr size_t dim = 128;
  constexpr size_t num_vectors = 1000;

  vanedb::DiskIndexBuilder builder(dim, vanedb::Metric::COSINE);
  builder.reserve(num_vectors);

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

  std::vector<std::vector<float>> all_vectors(num_vectors);
  for (uint64_t i = 0; i < num_vectors; ++i) {
    all_vectors[i].resize(dim);
    for (size_t j = 0; j < dim; ++j) {
      all_vectors[i][j] = dis(gen);
    }
    builder.add(i, all_vectors[i].data());
  }

  builder.save(filename);

  SECTION("Load and search") {
    vanedb::DiskIndex store(filename);

    REQUIRE(store.size() == num_vectors);
    REQUIRE(store.dimension() == dim);

    // Search should find exact matches
    for (size_t i = 0; i < 10; ++i) {
      auto results = store.search(all_vectors[i].data(), 1);
      REQUIRE(results.size() == 1);
      REQUIRE(results[0].id == i);
      REQUIRE(results[0].distance == Approx(0.0f).margin(1e-5f));
    }
  }

  SECTION("Vectors are preserved") {
    vanedb::DiskIndex store(filename);

    for (size_t i = 0; i < num_vectors; ++i) {
      const float* retrieved = store.get(i);
      REQUIRE(retrieved != nullptr);
      for (size_t j = 0; j < dim; ++j) {
        REQUIRE(retrieved[j] == all_vectors[i][j]);
      }
    }
  }

  std::filesystem::remove(filename);
}

TEST_CASE("DiskIndex - cosine metric", "[disk]") {
  const std::string filename = "test_mmap_cosine.bin";
  std::filesystem::remove(filename);

  vanedb::DiskIndexBuilder builder(4, vanedb::Metric::COSINE);

  // Same direction, different magnitudes
  float vec1[] = {1.0f, 0.0f, 0.0f, 0.0f};
  float vec2[] = {2.0f, 0.0f, 0.0f, 0.0f};
  float vec3[] = {0.0f, 1.0f, 0.0f, 0.0f};  // Orthogonal

  builder.add(1, vec1);
  builder.add(2, vec2);
  builder.add(3, vec3);
  builder.save(filename);

  {
    vanedb::DiskIndex store(filename);

    float query[] = {3.0f, 0.0f, 0.0f, 0.0f};
    auto results = store.search(query, 2);

    // vec1 and vec2 should be closest (same direction)
    REQUIRE(results.size() == 2);
    REQUIRE((results[0].id == 1 || results[0].id == 2));
    REQUIRE((results[1].id == 1 || results[1].id == 2));
    REQUIRE(results[0].distance == Approx(0.0f).margin(1e-5f));
  }  // Scope ensures store is destroyed and file unmapped before removal (Windows file locking)

  std::filesystem::remove(filename);
}

TEST_CASE("DiskIndex - dot product metric", "[disk]") {
  const std::string filename = "test_mmap_dot.bin";
  std::filesystem::remove(filename);

  vanedb::DiskIndexBuilder builder(4, vanedb::Metric::DOT);

  float vec1[] = {1.0f, 0.0f, 0.0f, 0.0f};
  float vec2[] = {2.0f, 0.0f, 0.0f, 0.0f};  // Higher dot product
  float vec3[] = {0.0f, 1.0f, 0.0f, 0.0f};

  builder.add(1, vec1);
  builder.add(2, vec2);
  builder.add(3, vec3);
  builder.save(filename);

  {
    vanedb::DiskIndex store(filename);

    float query[] = {1.0f, 0.0f, 0.0f, 0.0f};
    auto results = store.search(query, 1);

    // vec2 should be first (highest dot product = smallest negative distance)
    REQUIRE(results.size() == 1);
    REQUIRE(results[0].id == 2);
  }  // Scope ensures store is destroyed and file unmapped before removal (Windows file locking)

  std::filesystem::remove(filename);
}

TEST_CASE("DiskIndex - concurrent saves to one path do not corrupt it", "[disk][concurrency]") {
  // Both save paths wrote to `filename + ".tmp"`. That is unique per
  // destination, so it never had the extension-replacing collision Rust fixed
  // in vanedb#38 — but two threads saving the *same* path shared one temp
  // file and interleaved their writes into it.
  //
  // Each writer saves a *differently sized* index, and every candidate is
  // written once, alone, beforehand. The published file must then be
  // byte-identical to exactly one candidate. Sharing a temp file blends
  // several writers' bytes into the file that gets renamed, so it matches
  // none — which a same-content workload cannot show, because there every
  // interleaving still produces the right bytes.
  const std::string path = "test_disk_concurrent_save.bin";
  constexpr size_t kDim = 8, kThreads = 8;
  std::filesystem::remove(path);

  auto fill = [](vanedb::DiskIndexBuilder& b, size_t n) {
    for (size_t i = 0; i < n; ++i) {
      std::vector<float> v(kDim, static_cast<float>(i));
      b.add(i, v.data());
    }
  };
  auto read_all = [](const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
  };

  // One candidate per writer, each a different size, each written alone.
  std::vector<std::string> candidates;
  for (size_t t = 0; t < kThreads; ++t) {
    vanedb::DiskIndexBuilder b(kDim);
    fill(b, 64 * (t + 1));
    const std::string only = "test_disk_candidate_" + std::to_string(t) + ".bin";
    b.save(only);
    candidates.push_back(read_all(only));
    std::filesystem::remove(only);
  }

  std::vector<std::thread> writers;
  // Not vector<bool>: it is bit-packed, so writes to distinct elements share
  // a word and are a data race — the one container the standard excludes from
  // its distinct-element guarantee. ThreadSanitizer flags it while the test
  // still passes.
  std::vector<char> ok(kThreads, 0);
  writers.reserve(kThreads);
  for (size_t t = 0; t < kThreads; ++t) {
    writers.emplace_back([&, t] {
      vanedb::DiskIndexBuilder b(kDim);
      fill(b, 64 * (t + 1));
      try {
        b.save(path);
        ok[t] = 1;
      } catch (const std::exception&) {
        // Windows can refuse a rename whose destination another writer is
        // replacing at that instant. Losing that race is acceptable;
        // publishing a blend of two writers is not.
      }
    });
  }
  for (auto& w : writers) w.join();
  REQUIRE(std::count(ok.begin(), ok.end(), char{1}) >= 1);

  const std::string published = read_all(path);
  REQUIRE_FALSE(published.empty());
  REQUIRE(std::find(candidates.begin(), candidates.end(), published) != candidates.end());

  // Scoped: Windows refuses to delete a mapped file.
  {
    vanedb::DiskIndex index(path);
    REQUIRE(index.size() % 64 == 0);
    REQUIRE(index.dimension() == kDim);
  }

  // No temp file may survive a clean run.
  size_t leftovers = 0;
  for (const auto& entry : std::filesystem::directory_iterator(".")) {
    const std::string name = entry.path().filename().string();
    if (name.rfind("test_disk_concurrent_save.bin.", 0) == 0) ++leftovers;
  }
  REQUIRE(leftovers == 0);

  std::filesystem::remove(path);
}

TEST_CASE("detail::temp_path_for is unique per writer", "[disk][concurrency]") {
  const std::string dest = "some/dir/index.bin";
  REQUIRE(vanedb::detail::temp_path_for(dest) != vanedb::detail::temp_path_for(dest));
  // Distinct destinations must still not collide, which `filename + ".tmp"`
  // already guaranteed and the fix must not regress.
  REQUIRE(vanedb::detail::temp_path_for("index.bin") !=
          vanedb::detail::temp_path_for("index.idx"));
  // Stays beside the destination so the rename is same-filesystem.
  REQUIRE(vanedb::detail::temp_path_for(dest).rfind("some/dir/", 0) == 0);
  // The pid specifically: the counter alone satisfies both inequalities
  // above, so dropping the pid passes every other assertion here while
  // reintroducing collisions between *processes* saving the same path. That
  // is the one case the threaded test can never reach.
#if defined(_WIN32) || defined(_WIN64)
  const std::string pid = std::to_string(GetCurrentProcessId());
#else
  const std::string pid = std::to_string(getpid());
#endif
  REQUIRE(vanedb::detail::temp_path_for(dest).find("." + pid + ".") != std::string::npos);
}

TEST_CASE("DiskIndex - saving twice to one path replaces it", "[disk]") {
  // `std::rename` has implementation-defined behaviour when the destination
  // exists; on Windows it fails instead of replacing. Every save after the
  // first therefore threw there while working on POSIX, and no test covered
  // it because none saved twice to one path. `ApproxIndex::save` already used
  // `std::filesystem::rename`, which the standard requires to replace.
  const std::string path = "test_disk_save_twice.bin";
  std::filesystem::remove(path);

  vanedb::DiskIndexBuilder first(2);
  float a[2] = {1.0f, 0.0f};
  first.add(1, a);
  REQUIRE_NOTHROW(first.save(path));

  vanedb::DiskIndexBuilder second(2);
  float b[2] = {0.0f, 1.0f};
  second.add(7, b);
  second.add(9, a);
  REQUIRE_NOTHROW(second.save(path));

  // The second save must have replaced the first, not merged with it.
  // Scoped: Windows refuses to delete a file while it is mapped, so the
  // DiskIndex has to be destroyed before the cleanup below.
  {
    vanedb::DiskIndex index(path);
    REQUIRE(index.size() == 2);
    REQUIRE(index.contains(7));
    REQUIRE(index.contains(9));
    REQUIRE_FALSE(index.contains(1));
  }

  std::filesystem::remove(path);
}
