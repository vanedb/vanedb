#include "core/approx_index.h"
#include <atomic>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <random>
#include <thread>
#include <unordered_set>
#include <vector>

using Catch::Approx;

static std::filesystem::path legacy_graph_fixture(int version) {
#ifdef __GLIBCXX__
  const std::string rng_layout = "_indexed";
#else
  const std::string rng_layout = "_state";  // libc++ and MSVC
#endif
  return std::filesystem::path(VANEDB_LEGACY_GRAPH_DIR) /
      ("v" + std::to_string(version) + (version == 1 ? "" : rng_layout) + ".qvrd");
}

TEST_CASE("ApproxIndex - fixed legacy files preserve graph state", "[index][persistence][legacy]") {
  auto bytes = [](const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    REQUIRE(input.good());
    return std::vector<char>(std::istreambuf_iterator<char>(input), {});
  };
  for (int version = 1; version <= 3; ++version) {
    INFO(version);
    auto index = vanedb::ApproxIndex::load(legacy_graph_fixture(version).string());
    REQUIRE(index->size() == 3);
    REQUIRE(index->dimension() == 2);
    REQUIRE(index->capacity() == 4);
    REQUIRE(index->get_ef_search() == 16);
    REQUIRE(index->get_vector(101) == std::vector<float>{1.0f, 0.0f});
    REQUIRE(index->get_vector(202) == std::vector<float>{0.0f, 1.0f});
    REQUIRE(index->get_vector(UINT64_MAX) == std::vector<float>{0.8f, 0.2f});
    const float query[] = {1.0f, 0.0f};
    auto hits = index->search(query, 3);
    REQUIRE(hits.size() == 3);
    REQUIRE(hits[0].id == 101);
    REQUIRE(hits[1].id == UINT64_MAX);
    REQUIRE(hits[2].id == 202);
    REQUIRE(hits[0].distance == (version == 3 ? -1.0f : 0.0f));

    const std::string saved = "legacy_graph_roundtrip_" + std::to_string(version) + ".qvrd";
    index->save(saved);
    auto actual = bytes(saved);
    const std::string metric = version == 1 ? "l2" : (version == 2 ? "cosine" : "dot");
    auto expected = bytes(std::filesystem::path(VANEDB_GRAPH_DIR) /
        (metric + "_rng" + std::to_string(vanedb::detail::graph::NATIVE_RNG) + ".vndb"));
    std::filesystem::remove(saved);
    // Legacy fixtures retain M=2 and ef_search=16. The canonical VNDB fixture
    // uses distinct header values (M=5, ef_search=32) to detect transposition.
    // Adjust only those two expected fields; all legacy configuration and
    // graph/RNG bytes must still survive migration exactly.
    expected[40] = 2;
    expected[56] = 16;
    // Exact bytes cover topology, configuration, vectors, identity and RNG.
    REQUIRE(actual == expected);
    const float added[] = {0.25f, 0.75f};
    REQUIRE_NOTHROW(index->add(303, added));
    REQUIRE(index->get_vector(303) == std::vector<float>{0.25f, 0.75f});
  }
}

TEST_CASE("ApproxIndex - legacy level multiplier is derived on load", "[index][persistence][legacy]") {
  for (double multiplier : {-100.0, std::numeric_limits<double>::infinity(),
                            std::numeric_limits<double>::quiet_NaN()}) {
    const std::string path = "legacy_graph_untrusted_multiplier.qvrd";
    std::filesystem::copy_file(legacy_graph_fixture(3),
                              path, std::filesystem::copy_options::overwrite_existing);
    {
      std::fstream file(path, std::ios::in | std::ios::out | std::ios::binary);
      file.seekp(52);
      file.write(reinterpret_cast<const char*>(&multiplier), sizeof(multiplier));
    }
    auto index = vanedb::ApproxIndex::load(path);
    std::filesystem::remove(path);
    const float added[] = {0.25f, 0.75f};
    REQUIRE_NOTHROW(index->add(303, added));
    REQUIRE(index->get_vector(303) == std::vector<float>{0.25f, 0.75f});
  }
}

TEST_CASE("ApproxIndex - construction", "[index]") {
  SECTION("Valid construction") {
    REQUIRE_NOTHROW(vanedb::ApproxIndex(768));
    REQUIRE_NOTHROW(vanedb::ApproxIndex(128, vanedb::Metric::COSINE));
    REQUIRE_NOTHROW(vanedb::ApproxIndex(64, vanedb::Metric::L2, 10000, 32, 400));
  }

  SECTION("Zero dimension throws") {
    REQUIRE_THROWS_AS(vanedb::ApproxIndex(0), std::invalid_argument);
  }

  SECTION("Zero max_elements throws") {
    REQUIRE_THROWS_AS(vanedb::ApproxIndex(768, vanedb::Metric::L2, 0), std::invalid_argument);
  }

  SECTION("M < 2 throws") {
    // M=0 should throw
    REQUIRE_THROWS_AS(vanedb::ApproxIndex(768, vanedb::Metric::L2, 1000, 0), std::invalid_argument);
    // M=1 should throw
    REQUIRE_THROWS_AS(vanedb::ApproxIndex(768, vanedb::Metric::L2, 1000, 1), std::invalid_argument);
    // M=2 should succeed
    REQUIRE_NOTHROW(vanedb::ApproxIndex(768, vanedb::Metric::L2, 1000, 2));
  }

  SECTION("Check initial state") {
    vanedb::ApproxIndex index(768);
    REQUIRE(index.size() == 0);
    REQUIRE(index.dimension() == 768);
    REQUIRE(index.capacity() == 100000);  // default
  }
}

TEST_CASE("ApproxIndex - add and search", "[index]") {
  constexpr size_t dim = 64;
  vanedb::ApproxIndex index(dim, vanedb::Metric::L2, 1000);

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

  SECTION("Add single vector and search") {
    std::vector<float> vec(dim);
    for (size_t i = 0; i < dim; ++i) {
      vec[i] = dis(gen);
    }

    index.add(1, vec.data());
    REQUIRE(index.size() == 1);
    REQUIRE(index.contains(1));

    // Search should return the same vector
    auto results = index.search(vec.data(), 1);
    REQUIRE(results.size() == 1);
    REQUIRE(results[0].id == 1);
    REQUIRE(results[0].distance == Approx(0.0f).margin(1e-6f));
  }

  SECTION("Add multiple vectors") {
    for (uint64_t i = 0; i < 100; ++i) {
      std::vector<float> vec(dim);
      for (size_t j = 0; j < dim; ++j) {
        vec[j] = dis(gen);
      }
      index.add(i, vec.data());
    }

    REQUIRE(index.size() == 100);
  }

  SECTION("Duplicate ID throws") {
    std::vector<float> vec1(dim, 1.0f);
    std::vector<float> vec2(dim, 2.0f);

    index.add(42, vec1.data());
    REQUIRE_THROWS_AS(index.add(42, vec2.data()), std::invalid_argument);
  }

  SECTION("Null vector throws") {
    REQUIRE_THROWS_AS(index.add(1, nullptr), std::invalid_argument);

    std::vector<float> vec(dim, 1.0f);
    index.add(1, vec.data());
    REQUIRE_THROWS_AS(index.search(nullptr, 1), std::invalid_argument);
  }

  SECTION("k=0 throws") {
    std::vector<float> vec(dim, 1.0f);
    index.add(1, vec.data());
    REQUIRE_THROWS_AS(index.search(vec.data(), 0), std::invalid_argument);
  }

  SECTION("ApproxIndex full throws") {
    constexpr size_t small_capacity = 5;
    vanedb::ApproxIndex small_index(dim, vanedb::Metric::L2, small_capacity);

    std::vector<float> vec(dim, 1.0f);
    for (uint64_t i = 0; i < small_capacity; ++i) {
      small_index.add(i, vec.data());
    }
    REQUIRE(small_index.size() == small_capacity);

    // Adding one more should throw
    REQUIRE_THROWS_AS(small_index.add(small_capacity, vec.data()), std::runtime_error);
  }
}

TEST_CASE("ApproxIndex - search quality", "[index]") {
  constexpr size_t dim = 32;
  constexpr size_t num_vectors = 500;
  vanedb::ApproxIndex index(dim, vanedb::Metric::L2, num_vectors, 16, 100);

  std::mt19937 gen(123);
  std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

  // FlatIndex vectors for ground truth computation
  std::vector<std::vector<float>> all_vectors(num_vectors);

  // Add vectors
  for (uint64_t i = 0; i < num_vectors; ++i) {
    all_vectors[i].resize(dim);
    for (size_t j = 0; j < dim; ++j) {
      all_vectors[i][j] = dis(gen);
    }
    index.add(i, all_vectors[i].data());
  }

  SECTION("Search finds exact match") {
    // Search for an existing vector
    auto results = index.search(all_vectors[42].data(), 1);
    REQUIRE(results.size() == 1);
    REQUIRE(results[0].id == 42);
    REQUIRE(results[0].distance == Approx(0.0f).margin(1e-5f));
  }

  SECTION("Search returns k results") {
    std::vector<float> query(dim);
    for (size_t i = 0; i < dim; ++i) {
      query[i] = dis(gen);
    }

    auto results = index.search(query.data(), 10);
    REQUIRE(results.size() == 10);

    // Results should be sorted by distance
    for (size_t i = 1; i < results.size(); ++i) {
      REQUIRE(results[i].distance >= results[i-1].distance);
    }
  }

  SECTION("Higher ef_search improves recall") {
    // Compute ground truth (brute force)
    std::vector<float> query(dim);
    for (size_t i = 0; i < dim; ++i) {
      query[i] = dis(gen);
    }

    std::vector<std::pair<float, uint64_t>> ground_truth;
    for (uint64_t i = 0; i < num_vectors; ++i) {
      float dist = vanedb::l2_sq(query.data(), all_vectors[i].data(), dim);
      ground_truth.emplace_back(dist, i);
    }
    std::sort(ground_truth.begin(), ground_truth.end());

    // Search with low ef
    index.set_ef_search(10);
    auto results_low = index.search(query.data(), 10);

    // Search with high ef
    index.set_ef_search(100);
    auto results_high = index.search(query.data(), 10);

    // Count recall
    std::unordered_set<uint64_t> gt_set;
    for (size_t i = 0; i < 10; ++i) {
      gt_set.insert(ground_truth[i].second);
    }

    int recall_low = 0, recall_high = 0;
    for (const auto& r : results_low) {
      if (gt_set.count(r.id)) recall_low++;
    }
    for (const auto& r : results_high) {
      if (gt_set.count(r.id)) recall_high++;
    }

    // Higher ef should give same or better recall
    REQUIRE(recall_high >= recall_low);

    // With ef=100 on 500 vectors, recall should be high
    REQUIRE(recall_high >= 8);  // At least 80% recall
  }
}

TEST_CASE("ApproxIndex - distance metrics", "[index]") {
  constexpr size_t dim = 8;

  SECTION("L2 distance") {
    vanedb::ApproxIndex index(dim, vanedb::Metric::L2, 100);

    std::vector<float> v1 = {1, 0, 0, 0, 0, 0, 0, 0};
    std::vector<float> v2 = {0, 1, 0, 0, 0, 0, 0, 0};
    std::vector<float> v3 = {1, 0, 0, 0, 0, 0, 0, 0};  // Same as v1

    index.add(1, v1.data());
    index.add(2, v2.data());
    index.add(3, v3.data());

    // Query with v1 - should find v3 (identical) then v2
    auto results = index.search(v1.data(), 3);
    REQUIRE(results.size() == 3);
    // First result should be v1 or v3 (distance 0)
    REQUIRE((results[0].id == 1 || results[0].id == 3));
    REQUIRE(results[0].distance == Approx(0.0f).margin(1e-6f));
  }

  SECTION("Cosine distance") {
    vanedb::ApproxIndex index(dim, vanedb::Metric::COSINE, 100);

    std::vector<float> v1 = {1, 0, 0, 0, 0, 0, 0, 0};
    std::vector<float> v2 = {2, 0, 0, 0, 0, 0, 0, 0};  // Same direction, different magnitude
    std::vector<float> v3 = {0, 1, 0, 0, 0, 0, 0, 0};  // Orthogonal

    index.add(1, v1.data());
    index.add(2, v2.data());
    index.add(3, v3.data());

    // Query with v1 - v2 should be closest (same direction)
    auto results = index.search(v1.data(), 3);
    REQUIRE(results.size() == 3);
    // v1 and v2 have cosine distance ~0
    REQUIRE((results[0].id == 1 || results[0].id == 2));
  }

  SECTION("Dot product (MIPS)") {
    vanedb::ApproxIndex index(dim, vanedb::Metric::DOT, 100);

    std::vector<float> v1 = {1, 0, 0, 0, 0, 0, 0, 0};
    std::vector<float> v2 = {2, 0, 0, 0, 0, 0, 0, 0};  // Higher dot product
    std::vector<float> v3 = {0, 1, 0, 0, 0, 0, 0, 0};  // Orthogonal

    index.add(1, v1.data());
    index.add(2, v2.data());
    index.add(3, v3.data());

    // Query with v1 - v2 should be "closest" (highest dot product = lowest negative)
    auto results = index.search(v1.data(), 3);
    REQUIRE(results.size() == 3);
    // v2 has highest dot product with v1
    REQUIRE(results[0].id == 2);
  }
}

TEST_CASE("ApproxIndex - stress test", "[index][stress]") {
  constexpr size_t dim = 128;
  constexpr size_t num_vectors = 1000;
  vanedb::ApproxIndex index(dim, vanedb::Metric::L2, num_vectors);

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

  SECTION("Insert many vectors") {
    for (uint64_t i = 0; i < num_vectors; ++i) {
      std::vector<float> vec(dim);
      for (size_t j = 0; j < dim; ++j) {
        vec[j] = dis(gen);
      }
      index.add(i, vec.data());
    }

    REQUIRE(index.size() == num_vectors);

    // Should be able to search
    std::vector<float> query(dim);
    for (size_t j = 0; j < dim; ++j) {
      query[j] = dis(gen);
    }

    auto results = index.search(query.data(), 10);
    REQUIRE(results.size() == 10);
  }
}

TEST_CASE("ApproxIndex - concurrent search", "[index][thread]") {
  constexpr size_t dim = 64;
  constexpr size_t num_vectors = 500;
  vanedb::ApproxIndex index(dim, vanedb::Metric::L2, num_vectors);

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

  // Pre-populate index
  for (uint64_t i = 0; i < num_vectors; ++i) {
    std::vector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = dis(gen);
    }
    index.add(i, vec.data());
  }

  SECTION("Multiple concurrent searches") {
    std::atomic<int> completed{0};
    constexpr int num_threads = 4;
    constexpr int searches_per_thread = 100;

    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    for (int t = 0; t < num_threads; ++t) {
      threads.emplace_back([&index, &completed, t]() {
        std::mt19937 local_gen(t * 1000);
        std::uniform_real_distribution<float> local_dis(-1.0f, 1.0f);

        for (int i = 0; i < searches_per_thread; ++i) {
          std::vector<float> query(dim);
          for (size_t j = 0; j < dim; ++j) {
            query[j] = local_dis(local_gen);
          }
          auto results = index.search(query.data(), 10);
          REQUIRE(results.size() == 10);
        }
        ++completed;
      });
    }

    for (auto& t : threads) {
      t.join();
    }

    REQUIRE(completed == num_threads);
  }
}

TEST_CASE("ApproxIndex - concurrent add and search", "[index][thread]") {
  constexpr size_t dim = 32;
  constexpr size_t initial_vectors = 100;
  constexpr size_t max_elements = 1000;
  vanedb::ApproxIndex index(dim, vanedb::Metric::L2, max_elements);

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

  // Pre-populate with some vectors
  for (uint64_t i = 0; i < initial_vectors; ++i) {
    std::vector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) vec[j] = dis(gen);
    index.add(i, vec.data());
  }

  SECTION("Concurrent writers and readers") {
    std::atomic<int> search_completed{0};
    std::atomic<int> add_completed{0};
    std::atomic<uint64_t> next_id{initial_vectors};
    constexpr int num_readers = 4;
    constexpr int num_writers = 2;
    constexpr int searches_per_reader = 50;
    constexpr int adds_per_writer = 50;

    std::vector<std::thread> threads;
    threads.reserve(num_readers + num_writers);

    // Launch reader threads
    for (int t = 0; t < num_readers; ++t) {
      threads.emplace_back([&index, &search_completed, t]() {
        std::mt19937 local_gen(t * 1000);
        std::uniform_real_distribution<float> local_dis(-1.0f, 1.0f);

        for (int i = 0; i < searches_per_reader; ++i) {
          std::vector<float> query(dim);
          for (size_t j = 0; j < dim; ++j) query[j] = local_dis(local_gen);
          auto results = index.search(query.data(), 5);
          // Results should be valid (may vary as index grows)
          REQUIRE(results.size() <= 5);
          REQUIRE(results.size() > 0);
        }
        ++search_completed;
      });
    }

    // Launch writer threads
    for (int t = 0; t < num_writers; ++t) {
      threads.emplace_back([&index, &add_completed, &next_id, t]() {
        std::mt19937 local_gen((t + 100) * 1000);
        std::uniform_real_distribution<float> local_dis(-1.0f, 1.0f);

        for (int i = 0; i < adds_per_writer; ++i) {
          std::vector<float> vec(dim);
          for (size_t j = 0; j < dim; ++j) vec[j] = local_dis(local_gen);
          uint64_t id = next_id.fetch_add(1);
          index.add(id, vec.data());
        }
        ++add_completed;
      });
    }

    for (auto& t : threads) {
      t.join();
    }

    REQUIRE(search_completed == num_readers);
    REQUIRE(add_completed == num_writers);
    REQUIRE(index.size() == initial_vectors + num_writers * adds_per_writer);
  }
}

TEST_CASE("ApproxIndex - serialization", "[index][serialization]") {
  const std::string filename = "test_hnsw_index.bin";
  constexpr size_t dim = 16;
  constexpr size_t max_elements = 100;
  constexpr size_t M = 8;
  constexpr size_t ef_construction = 50;
  constexpr vanedb::Metric metric = vanedb::Metric::COSINE;

  // Create and populate an index
  vanedb::ApproxIndex original_index(dim, metric, max_elements, M, ef_construction);
  original_index.set_ef_search(30);

  std::mt19937 gen(1234);
  std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

  std::vector<std::vector<float>> test_vectors(max_elements / 2); // Populate half capacity
  for (uint64_t i = 0; i < test_vectors.size(); ++i) {
    test_vectors[i].resize(dim);
    for (size_t j = 0; j < dim; ++j) {
      test_vectors[i][j] = dis(gen);
    }
    original_index.add(i + 1, test_vectors[i].data()); // IDs starting from 1
  }

  SECTION("Save and load index successfully") {
    // Save the original index
    REQUIRE_NOTHROW(original_index.save(filename));

    // Load into a new index
    std::unique_ptr<vanedb::ApproxIndex> loaded_index_ptr;
    REQUIRE_NOTHROW(loaded_index_ptr = vanedb::ApproxIndex::load(filename));
    vanedb::ApproxIndex& loaded_index = *loaded_index_ptr;

    // Verify configuration parameters
    REQUIRE(loaded_index.dimension() == original_index.dimension());
    REQUIRE(loaded_index.capacity() == original_index.capacity());
    REQUIRE(loaded_index.size() == original_index.size());
    REQUIRE(loaded_index.get_ef_search() == original_index.get_ef_search());
    // Metric is private, cannot directly check. Assume it's loaded correctly.

    // Verify search results are identical
    for (const auto& vec : test_vectors) {
      std::vector<vanedb::HNSWSearchResult> original_results = original_index.search(vec.data(), 5);
      std::vector<vanedb::HNSWSearchResult> loaded_results = loaded_index.search(vec.data(), 5);

      REQUIRE(original_results.size() == loaded_results.size());
      for (size_t i = 0; i < original_results.size(); ++i) {
        REQUIRE(original_results[i].id == loaded_results[i].id);
        REQUIRE(original_results[i].distance == Approx(loaded_results[i].distance).margin(1e-5f));
      }
    }
  }

  SECTION("Loading from non-existent file throws") {
    std::filesystem::remove(filename); // Ensure file doesn't exist
    REQUIRE_THROWS_AS(vanedb::ApproxIndex::load(filename + "_nonexistent"), std::runtime_error);
  }

  SECTION("Loading corrupted file - bad magic number") {
    // Create a file with wrong magic number
    std::string corrupt_file = filename + "_corrupt_magic";
    {
      std::ofstream ofs(corrupt_file, std::ios::binary);
      uint32_t bad_magic = 0xDEADBEEF;
      ofs.write(reinterpret_cast<const char*>(&bad_magic), sizeof(bad_magic));
    }
    REQUIRE_THROWS_AS(vanedb::ApproxIndex::load(corrupt_file), std::runtime_error);
    std::filesystem::remove(corrupt_file);
  }

  SECTION("Loading corrupted file - unsupported version") {
    // Create a file with correct magic but wrong version
    std::string corrupt_file = filename + "_corrupt_version";
    {
      std::ofstream ofs(corrupt_file, std::ios::binary);
      uint32_t magic = 0x51565244; // "QVRD" - correct magic
      uint32_t bad_version = 999;
      ofs.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
      ofs.write(reinterpret_cast<const char*>(&bad_version), sizeof(bad_version));
    }
    REQUIRE_THROWS_AS(vanedb::ApproxIndex::load(corrupt_file), std::runtime_error);
    std::filesystem::remove(corrupt_file);
  }

  SECTION("Loading truncated file") {
    // Save valid index then truncate it
    original_index.save(filename);

    // Truncate the file to 50 bytes (incomplete header)
    {
      std::ofstream ofs(filename, std::ios::binary | std::ios::trunc);
      uint32_t magic = 0x51565244; // "QVRD" - correct magic but truncated
      ofs.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
      // Write partial data - this should fail on load
    }
    REQUIRE_THROWS(vanedb::ApproxIndex::load(filename));
  }

  SECTION("Loading empty file") {
    std::string empty_file = filename + "_empty";
    {
      std::ofstream ofs(empty_file, std::ios::binary);
      // Empty file
    }
    REQUIRE_THROWS(vanedb::ApproxIndex::load(empty_file));
    std::filesystem::remove(empty_file);
  }

  SECTION("Loading corrupted file - invalid entry point") {
    // Save valid index first
    original_index.save(filename);

    // Read file, corrupt ep_ to be >= count, write back
    std::string corrupt_file = filename + "_corrupt_ep";
    {
      std::ifstream ifs(filename, std::ios::binary);
      std::vector<char> data((std::istreambuf_iterator<char>(ifs)),
                              std::istreambuf_iterator<char>());
      ifs.close();

      // VNDB v2 entry point is at byte 72.
      // Write invalid ep_ value (999999, much larger than count)
      size_t invalid_ep = 999999;
      std::memcpy(data.data() + 72, &invalid_ep, sizeof(invalid_ep));

      std::ofstream ofs(corrupt_file, std::ios::binary);
      ofs.write(data.data(), data.size());
    }
    REQUIRE_THROWS_AS(vanedb::ApproxIndex::load(corrupt_file), std::runtime_error);
    std::filesystem::remove(corrupt_file);
  }

  SECTION("Loading corrupted file - invalid max_level") {
    original_index.save(filename);

    std::string corrupt_file = filename + "_corrupt_maxlevel";
    {
      std::ifstream ifs(filename, std::ios::binary);
      std::vector<char> data((std::istreambuf_iterator<char>(ifs)),
                              std::istreambuf_iterator<char>());
      ifs.close();

      // VNDB v2 max_level is at byte 80.
      int invalid_max_level = 100;  // > MAX_LEVEL (32)
      std::memcpy(data.data() + 80, &invalid_max_level, sizeof(invalid_max_level));

      std::ofstream ofs(corrupt_file, std::ios::binary);
      ofs.write(data.data(), data.size());
    }
    REQUIRE_THROWS_AS(vanedb::ApproxIndex::load(corrupt_file), std::runtime_error);
    std::filesystem::remove(corrupt_file);
  }

  SECTION("get_vector returns correct data after save/load") {
    original_index.save(filename);
    auto loaded = vanedb::ApproxIndex::load(filename);

    // Verify vectors are preserved
    for (size_t i = 0; i < test_vectors.size(); ++i) {
      std::vector<float> retrieved = loaded->get_vector(static_cast<uint64_t>(i + 1));
      REQUIRE(retrieved.size() == dim);
      for (size_t j = 0; j < dim; ++j) {
        REQUIRE(retrieved[j] == Approx(test_vectors[i][j]).margin(1e-6f));
      }
    }
  }

  // Cleanup
  std::filesystem::remove(filename);
}

TEST_CASE("ApproxIndex - recall benchmark", "[index][.benchmark]") {
  // This test measures recall rate - marked as hidden benchmark

  constexpr size_t dim = 128;
  constexpr size_t num_vectors = 5000;
  constexpr size_t num_queries = 100;
  constexpr size_t k = 10;

  vanedb::ApproxIndex index(dim, vanedb::Metric::L2, num_vectors, 16, 200);

  std::mt19937 gen(123);
  std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

  // FlatIndex all vectors
  std::vector<std::vector<float>> all_vectors(num_vectors);
  for (uint64_t i = 0; i < num_vectors; ++i) {
    all_vectors[i].resize(dim);
    for (size_t j = 0; j < dim; ++j) {
      all_vectors[i][j] = dis(gen);
    }
    index.add(i, all_vectors[i].data());
  }

  // Generate queries
  std::vector<std::vector<float>> queries(num_queries);
  for (size_t q = 0; q < num_queries; ++q) {
    queries[q].resize(dim);
    for (size_t j = 0; j < dim; ++j) {
      queries[q][j] = dis(gen);
    }
  }

  // Compute ground truth (brute force)
  std::vector<std::unordered_set<uint64_t>> ground_truth(num_queries);
  for (size_t q = 0; q < num_queries; ++q) {
    std::vector<std::pair<float, uint64_t>> distances;
    for (uint64_t i = 0; i < num_vectors; ++i) {
      float dist = vanedb::l2_sq(queries[q].data(), all_vectors[i].data(), dim);
      distances.emplace_back(dist, i);
    }
    std::partial_sort(distances.begin(), distances.begin() + k, distances.end());
    for (size_t i = 0; i < k; ++i) {
      ground_truth[q].insert(distances[i].second);
    }
  }

  // Test different ef values
  for (size_t ef : {10, 50, 100, 200}) {
    index.set_ef_search(ef);

    double total_recall = 0.0;
    for (size_t q = 0; q < num_queries; ++q) {
      auto results = index.search(queries[q].data(), k);
      int hits = 0;
      for (const auto& r : results) {
        if (ground_truth[q].count(r.id)) hits++;
      }
      total_recall += static_cast<double>(hits) / k;
    }

    double avg_recall = total_recall / num_queries;
    INFO("ef=" << ef << " recall=" << avg_recall);
    REQUIRE(avg_recall > 0.5);  // Should have at least 50% recall
  }
}

TEST_CASE("ApproxIndex - ef_search validation", "[index]") {
  constexpr size_t dim = 16;
  vanedb::ApproxIndex index(dim, vanedb::Metric::L2, 100);

  SECTION("ef_search = 0 throws") {
    REQUIRE_THROWS_AS(index.set_ef_search(0), std::invalid_argument);
  }

  SECTION("Valid ef_search values succeed") {
    REQUIRE_NOTHROW(index.set_ef_search(1));
    REQUIRE(index.get_ef_search() == 1);

    REQUIRE_NOTHROW(index.set_ef_search(100));
    REQUIRE(index.get_ef_search() == 100);

    REQUIRE_NOTHROW(index.set_ef_search(10000));
    REQUIRE(index.get_ef_search() == 10000);
  }
}

TEST_CASE("ApproxIndex - get_vector edge cases", "[index]") {
  constexpr size_t dim = 8;
  vanedb::ApproxIndex index(dim, vanedb::Metric::L2, 100);

  SECTION("get_vector on non-existent ID throws") {
    REQUIRE_THROWS_AS(index.get_vector(42), std::runtime_error);
  }

  SECTION("get_vector on empty index throws") {
    REQUIRE_THROWS_AS(index.get_vector(0), std::runtime_error);
  }

  SECTION("get_vector returns correct vector") {
    std::vector<float> original = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    index.add(42, original.data());

    std::vector<float> retrieved = index.get_vector(42);
    REQUIRE(retrieved.size() == dim);
    for (size_t i = 0; i < dim; ++i) {
      REQUIRE(retrieved[i] == Approx(original[i]).margin(1e-6f));
    }
  }
}

TEST_CASE("ApproxIndex - corrupted RNG state", "[index][serialization]") {
  const std::string filename = "test_hnsw_rng_corrupt.bin";
  constexpr size_t dim = 8;

  // Create and save a valid index
  vanedb::ApproxIndex original(dim, vanedb::Metric::L2, 50);
  std::vector<float> vec(dim, 1.0f);
  original.add(1, vec.data());
  original.save(filename);

  SECTION("Loading file with corrupted RNG state throws") {
    // Read the file content
    std::ifstream ifs(filename, std::ios::binary);
    std::vector<char> data((std::istreambuf_iterator<char>(ifs)),
                            std::istreambuf_iterator<char>());
    ifs.close();

    // The RNG state is serialized near the end as a string stream
    // We corrupt it by truncating the file before RNG state is complete
    size_t corrupt_size = data.size() - 100; // Truncate last 100 bytes
    if (corrupt_size > 0) {
      std::string corrupt_file = filename + "_rng_corrupt";
      std::ofstream ofs(corrupt_file, std::ios::binary);
      ofs.write(data.data(), corrupt_size);
      ofs.close();

      REQUIRE_THROWS(vanedb::ApproxIndex::load(corrupt_file));
      std::filesystem::remove(corrupt_file);
    }
  }

  std::filesystem::remove(filename);
}

TEST_CASE("ApproxIndex - corruption validation tests", "[index][serialization]") {
  SECTION("Loading file with invalid metric throws") {
    const std::string filename = "test_hnsw_bad_metric.bin";
    {
      std::ofstream ofs(filename, std::ios::binary);
      uint32_t magic = vanedb::ApproxIndex::MAGIC;
      uint32_t version = vanedb::ApproxIndex::VERSION;
      size_t dim = 8;
      uint32_t bad_metric = 99; // Invalid metric
      ofs.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
      ofs.write(reinterpret_cast<const char*>(&version), sizeof(version));
      ofs.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
      ofs.write(reinterpret_cast<const char*>(&bad_metric), sizeof(bad_metric));
    }
    REQUIRE_THROWS_AS(vanedb::ApproxIndex::load(filename), std::runtime_error);
    std::filesystem::remove(filename);
  }

  SECTION("Loading file with count exceeding max_elements throws") {
    const std::string filename = "test_hnsw_bad_count.bin";
    {
      std::ofstream ofs(filename, std::ios::binary);
      uint32_t magic = vanedb::ApproxIndex::MAGIC;
      uint32_t version = vanedb::ApproxIndex::VERSION;
      size_t dim = 8;
      uint32_t metric = 0;
      size_t max_el = 10;  // max_elements = 10
      size_t M = 16;
      size_t ef_con = 200;
      size_t ef_s = 50;
      double mult = 0.5;
      size_t cnt = 100;  // count > max_elements (invalid)
      ofs.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
      ofs.write(reinterpret_cast<const char*>(&version), sizeof(version));
      ofs.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
      ofs.write(reinterpret_cast<const char*>(&metric), sizeof(metric));
      ofs.write(reinterpret_cast<const char*>(&max_el), sizeof(max_el));
      ofs.write(reinterpret_cast<const char*>(&M), sizeof(M));
      ofs.write(reinterpret_cast<const char*>(&ef_con), sizeof(ef_con));
      ofs.write(reinterpret_cast<const char*>(&ef_s), sizeof(ef_s));
      ofs.write(reinterpret_cast<const char*>(&mult), sizeof(mult));
      ofs.write(reinterpret_cast<const char*>(&cnt), sizeof(cnt));
    }
    REQUIRE_THROWS_AS(vanedb::ApproxIndex::load(filename), std::runtime_error);
    std::filesystem::remove(filename);
  }

}

TEST_CASE("ApproxIndex - contains edge cases", "[index]") {
  constexpr size_t dim = 8;
  vanedb::ApproxIndex index(dim, vanedb::Metric::L2, 100);

  SECTION("contains returns false for empty index") {
    REQUIRE_FALSE(index.contains(0));
    REQUIRE_FALSE(index.contains(1));
    REQUIRE_FALSE(index.contains(999));
  }

  SECTION("contains returns true after add") {
    std::vector<float> vec(dim, 1.0f);
    index.add(42, vec.data());

    REQUIRE(index.contains(42));
    REQUIRE_FALSE(index.contains(0));
    REQUIRE_FALSE(index.contains(43));
  }
}

TEST_CASE("ApproxIndex - search_layer epoch wrap", "[index]") {
  // Drive >65k searches to exercise the visited-bitmap epoch wrap-and-reset.
  constexpr size_t dim = 4;
  constexpr size_t n = 32;
  vanedb::ApproxIndex index(dim, vanedb::Metric::L2, n);
  std::mt19937 gen(7);
  std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
  for (size_t i = 0; i < n; ++i) {
    std::vector<float> v(dim);
    for (auto& x : v) x = dis(gen);
    index.add(i, v.data());
  }

  std::vector<float> q(dim);
  for (auto& x : q) x = dis(gen);
  auto first = index.search(q.data(), 5);
  REQUIRE(first.size() == 5);

  // 70_000 > 65_535 ensures at least one epoch wrap.
  for (int i = 0; i < 70'000; ++i) {
    auto r = index.search(q.data(), 5);
    REQUIRE(r.size() == 5);
  }

  auto last = index.search(q.data(), 5);
  REQUIRE(last.size() == first.size());
  for (size_t i = 0; i < first.size(); ++i) {
    REQUIRE(last[i].id == first[i].id);
    REQUIRE(last[i].distance == Catch::Approx(first[i].distance));
  }
}

TEST_CASE("ApproxIndex - save writes count-proportional files", "[index][persistence]") {
  // Issue #24 (mirror of vanedb/vanedb#18): an index with a large
  // pre-allocated capacity but few inserted vectors must not write
  // capacity-sized arrays. 10 vectors x 32 dims x 4 bytes is ~1.3 KB of
  // payload; 20 KB allows generous overhead, while capacity-sized arrays
  // would exceed 140 KB.
  const std::string filename = "test_hnsw_compact_save.bin";
  vanedb::ApproxIndex idx(32, vanedb::Metric::L2, 1000);
  std::vector<float> v(32);
  for (uint64_t i = 0; i < 10; ++i) {
    for (size_t d = 0; d < 32; ++d) v[d] = static_cast<float>(i * 32 + d);
    idx.add(i, v.data());
  }
  idx.save(filename);
  const auto file_size = std::filesystem::file_size(filename);
  std::filesystem::remove(filename);
  REQUIRE(file_size < 20'000);
}

namespace {
// Hand-writes an HNSW file in the given version's layout: a consistent
// 2-of-4-slots index. `full_arrays` selects the legacy v1/v2 layout (arrays
// span the whole capacity) vs the v3 compact layout (count-sized).
void write_hnsw_fixture(const std::string& filename, uint32_t ver, bool full_arrays,
                        float first_value = 1.0f, const std::string& corruption = "") {
  using vanedb::detail::write_bin;
  using vanedb::detail::write_vec;
  const size_t stored = full_arrays ? 4 : 2;
  std::ofstream f(filename, std::ios::binary);
  write_bin(f, vanedb::ApproxIndex::MAGIC);
  write_bin(f, ver);
  write_bin(f, size_t{2});    // dim
  write_bin(f, uint32_t{0});  // metric = L2
  write_bin(f, size_t{4});    // max_elements
  write_bin(f, size_t{2});    // M
  write_bin(f, size_t{10});   // ef_construction
  write_bin(f, size_t{10});   // ef_search
  write_bin(f, double{1.0});  // mult
  write_bin(f, size_t{2});    // count
  write_bin(f, size_t{0});    // entry point
  write_bin(f, int{corruption == "max_level" || corruption == "neighbor_layer" ? 1 : 0});
  std::vector<float> vectors = {first_value, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  vectors.resize(stored * 2);
  write_vec(f, vectors);
  std::vector<uint64_t> ext_ids = {10, 20, 0, 0};
  ext_ids.resize(stored);
  write_vec(f, ext_ids);
  std::vector<int> levels(stored, 0);
  if (corruption == "negative_level") levels[0] = -1;
  if (corruption == "neighbor_layer") levels[0] = 1;
  write_vec(f, levels);
  write_bin(f, size_t{2});  // id_map size
  write_bin(f, uint64_t{10});
  write_bin(f, size_t{0});
  write_bin(f, uint64_t{20});
  write_bin(f, size_t{1});
  write_bin(f, stored);  // neighbors size
  const size_t layer_count = corruption == "missing_layer" ? 0 :
      (corruption == "extra_layer" || corruption == "neighbor_layer" ? 2 : 1);
  write_bin(f, layer_count);
  std::vector<size_t> first_neighbors{1};
  if (corruption == "self_link") first_neighbors.push_back(0);
  if (corruption == "duplicate_link") first_neighbors.push_back(1);
  if (corruption == "degree") first_neighbors.assign(5, 1);
  if (layer_count) write_vec(f, first_neighbors);
  if (layer_count == 2) write_vec(f, std::vector<size_t>{1});
  write_bin(f, size_t{1});  // node 1: one level
  write_vec(f, std::vector<size_t>{0});
  for (size_t i = 2; i < stored; ++i) write_bin(f, size_t{0});  // unused slots
  if (ver >= 2) {
    std::mt19937 gen(7);
    std::stringstream ss;
    ss << gen;
    const std::string state = ss.str();
    write_bin(f, state.size());
    f.write(state.data(), static_cast<std::streamsize>(state.size()));
  }
}
}  // namespace

TEST_CASE("ApproxIndex - load accepts legacy v1/v2 full-capacity files", "[index][persistence]") {
  for (uint32_t ver : {1u, 2u}) {
    const std::string filename = "test_hnsw_legacy_v" + std::to_string(ver) + ".bin";
    write_hnsw_fixture(filename, ver, /*full_arrays=*/true);
    std::unique_ptr<vanedb::ApproxIndex> idx;
    REQUIRE_NOTHROW(idx = vanedb::ApproxIndex::load(filename));
    std::filesystem::remove(filename);
    REQUIRE(idx->size() == 2);
    REQUIRE(idx->capacity() == 4);
    const float q[2] = {1.0f, 0.1f};
    auto results = idx->search(q, 1);
    REQUIRE(results.size() == 1);
    REQUIRE(results[0].id == 10);
    // Spare capacity from the legacy file must remain usable.
    const float nv[2] = {0.5f, 0.5f};
    REQUIRE_NOTHROW(idx->add(30, nv));
    REQUIRE(idx->size() == 3);
  }
}

TEST_CASE("ApproxIndex - load rejects v3 with capacity-sized arrays", "[index][persistence]") {
  // The legacy full-capacity layout is NOT valid under v3, which stores
  // exactly `count` entries per array.
  const std::string filename = "test_hnsw_v3_full_arrays.bin";
  write_hnsw_fixture(filename, 3, /*full_arrays=*/true);
  REQUIRE_THROWS_AS(vanedb::ApproxIndex::load(filename), std::runtime_error);
  std::filesystem::remove(filename);
}

TEST_CASE("ApproxIndex - load rejects non-finite stored vectors", "[index][persistence]") {
  const std::string filename = "test_hnsw_non_finite.bin";
  write_hnsw_fixture(filename, 3, /*full_arrays=*/false,
                     std::numeric_limits<float>::quiet_NaN());
  REQUIRE_THROWS_AS(vanedb::ApproxIndex::load(filename), std::runtime_error);
  std::filesystem::remove(filename);
}

TEST_CASE("ApproxIndex - empty index save/load roundtrip", "[index][persistence]") {
  // VNDB stores no nodes for an empty index; load must re-expand to
  // full capacity so subsequent adds work.
  const std::string filename = "test_hnsw_empty_roundtrip.bin";
  vanedb::ApproxIndex idx(4, vanedb::Metric::L2, 10);
  idx.save(filename);
  std::unique_ptr<vanedb::ApproxIndex> loaded;
  REQUIRE_NOTHROW(loaded = vanedb::ApproxIndex::load(filename));
  std::filesystem::remove(filename);
  REQUIRE(loaded->size() == 0);
  REQUIRE(loaded->capacity() == 10);
  const float v[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  REQUIRE_NOTHROW(loaded->add(1, v));
  REQUIRE(loaded->size() == 1);
  const auto results = loaded->search(v, 1);
  REQUIRE(results.size() == 1);
  REQUIRE(results[0].id == 1);
}

TEST_CASE("ApproxIndex - load rejects inconsistent graph structure", "[index][persistence]") {
  for (const std::string corruption : {"negative_level", "missing_layer", "extra_layer", "max_level",
                                       "self_link", "duplicate_link", "degree", "neighbor_layer"}) {
    INFO(corruption);
    const std::string filename = "test_hnsw_graph_" + corruption + ".bin";
    write_hnsw_fixture(filename, 1, true, 1.0f, corruption);
    REQUIRE_THROWS_AS(vanedb::ApproxIndex::load(filename), std::runtime_error);
    std::filesystem::remove(filename);
  }
}

TEST_CASE("ApproxIndex - portable graph fixtures preserve every byte", "[index][persistence][vndb]") {
  for (const auto& entry : std::filesystem::directory_iterator(VANEDB_GRAPH_DIR)) {
    if (entry.path().extension() != ".vndb") continue;
    INFO(entry.path().filename().string());
    auto index = vanedb::ApproxIndex::load(entry.path().string());
    const std::string saved = "portable_graph_roundtrip.vndb";
    index->save(saved);
    auto bytes = [](const std::filesystem::path& path) {
      std::ifstream input(path, std::ios::binary);
      return std::vector<char>(std::istreambuf_iterator<char>(input), {});
    };
    REQUIRE(bytes(saved) == bytes(entry.path()));
    REQUIRE(index->dimension() == 2);
    REQUIRE(index->capacity() == 4);
    const float query[] = {1.0f, 0.0f};
    const auto hits = index->search(query, 3);
    const auto name = entry.path().stem().string();
    const size_t count = (name == "empty" || name == "all_deleted") ? 0 : (name.starts_with("deleted_") ? 2 : 3);
    REQUIRE(index->size() == count);
    REQUIRE(hits.size() == count);
    if (count) {
      REQUIRE(hits[0].id == (name == "deleted_entry" ? UINT64_MAX : 101));
      REQUIRE(hits[1].id == (name == "deleted_entry" ? 202 : UINT64_MAX));
      if (name != "deleted_entry")
        REQUIRE(hits[0].distance == (name.starts_with("dot") ? -1.0f : 0.0f));
      REQUIRE(index->get_vector(UINT64_MAX) == std::vector<float>{0.8f, 0.2f});
    }
    const float added[] = {0.25f, 0.75f};
    index->add(303, added);
    index->save(saved);
    auto reloaded = vanedb::ApproxIndex::load(saved);
    REQUIRE(reloaded->size() == count + 1);
    REQUIRE(reloaded->get_vector(303) == std::vector<float>{0.25f, 0.75f});
    std::filesystem::remove(saved);
  }
}

TEST_CASE("ApproxIndex - portable graph rejects malformed files", "[index][persistence][vndb]") {
  std::ifstream input(std::filesystem::path(VANEDB_GRAPH_DIR) / "l2_rng1.vndb", std::ios::binary);
  const std::vector<char> base(std::istreambuf_iterator<char>(input), {});
  REQUIRE(base.size() == 272);
  const std::string path = "portable_graph_corruption.vndb";
  auto rejects = [&](const std::vector<char>& bytes) {
    { std::ofstream file(path, std::ios::binary); file.write(bytes.data(), bytes.size()); }
    REQUIRE_THROWS_AS(vanedb::ApproxIndex::load(path), std::runtime_error);
  };
  for (size_t end = 0; end < base.size(); ++end) {
    INFO(end);
    rejects(std::vector<char>(base.begin(), base.begin() + end));
  }
  auto trailing = base;
  trailing.push_back(0);
  rejects(trailing);
  struct Change { size_t offset, width; uint64_t value; };
  for (auto change : std::vector<Change>{
      {4,4,3}, {8,4,2}, {12,4,99}, {16,8,0}, {24,8,UINT64_MAX}, {32,8,2},
      {40,8,UINT64_MAX}, {40,8,1}, {48,8,0}, {72,8,3}, {72,8,1}, {72,8,UINT64_MAX},
      {80,4,UINT32_MAX}, {84,4,0}, {88,8,UINT64_MAX}, {104,4,33}, {108,4,2},
      {112,4,0x7fc00000}, {120,8,5}, {128,8,0}, {136,8,1}, {136,8,3}, {152,8,1}, {160,8,101}}) {
    INFO(change.offset);
    auto bytes = base;
    for (size_t i = 0; i < change.width; ++i)
      bytes[change.offset + i] = static_cast<char>(change.value >> (8 * i));
    rejects(bytes);
  }
  std::filesystem::remove(path);
}

TEST_CASE("ApproxIndex - portable graph validates continuation metadata", "[index][persistence][vndb]") {
  auto read = [](const std::string& name) {
    std::ifstream file(std::filesystem::path(VANEDB_GRAPH_DIR) / name, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(file), {});
  };
  const auto base = read("l2_rng1.vndb");
  const auto words = read("l2_rng2.vndb").substr(272);
  const std::string path = "portable_graph_rng_corruption.vndb";
  const auto rest = words.substr(words.find(' '));
  for (auto [kind, state] : std::vector<std::pair<uint32_t, std::string>>{
      {1, "0"}, {2, ""}, {2, "-42" + rest}, {2, "+42" + rest},
      {2, "4294967296" + rest}, {2, words.substr(0, words.rfind(' ')) + " 625"},
      {3, words}, {2, words + "\xc2\xa0"}}) {
    {
      std::ofstream file(path, std::ios::binary);
      file.write(base.data(), 84);
      vanedb::detail::graph::write(file, kind);
      vanedb::detail::graph::write(file, static_cast<uint64_t>(state.size()));
      file.write(base.data() + 96, base.size() - 96);
      file.write(state.data(), state.size());
    }
    REQUIRE_THROWS_AS(vanedb::ApproxIndex::load(path), std::runtime_error);
  }
  std::filesystem::remove(path);
}

TEST_CASE("ApproxIndex - graph capacity hint does not allocate unused slots", "[index][persistence][vndb]") {
  const std::string path = "portable_graph_large_capacity.vndb";
  std::filesystem::copy_file(std::filesystem::path(VANEDB_GRAPH_DIR) / "l2_rng1.vndb",
                            path, std::filesystem::copy_options::overwrite_existing);
  {
    std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
    file.seekp(32);
    vanedb::detail::graph::write(file, uint64_t{100000000});
  }
  auto index = vanedb::ApproxIndex::load(path);
  REQUIRE(index->capacity() == 100000000);
  REQUIRE(index->size() == 3);
  const float added[] = {0.25f, 0.75f};
  index->add(303, added);
  REQUIRE(index->get_vector(303) == std::vector<float>{0.25f, 0.75f});
  index->save(path);
  REQUIRE(std::filesystem::file_size(path) < 10000);
  auto reloaded = vanedb::ApproxIndex::load(path);
  REQUIRE(reloaded->capacity() == 100000000);
  REQUIRE(reloaded->size() == 4);
  std::filesystem::remove(path);
}
