// VaneDB - Copyright (c) 2025 Anton Tsvetkov - MIT License
#pragma once
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <vector>

namespace vanedb::detail::graph {
// VNDB v2 graph field table: conformance/graph/README.md. Disk v1 is separate.
inline void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
template <typename T> void write(std::ostream& out, T value) {
  static_assert(std::is_unsigned_v<T>);
  char bytes[sizeof(T)];
  for (size_t i = 0; i < sizeof(T); ++i) bytes[i] = static_cast<char>(value >> (8 * i));
  out.write(bytes, sizeof(bytes));
}

class Reader {
  std::ifstream& file_;
public:
  uint64_t remaining;
  explicit Reader(std::ifstream& file) : file_(file) {
    const auto start = file.tellg();
    file.seekg(0, std::ios::end);
    const auto end = file.tellg();
    require(start >= 0 && end >= start, "Cannot measure VNDB graph");
    remaining = static_cast<uint64_t>(end - start);
    file.seekg(start);
  }
  std::string bytes(size_t count) {
    require(count <= remaining, "Truncated VNDB graph");
    std::string result(count, '\0');
    require(static_cast<bool>(file_.read(result.data(), count)), "Truncated VNDB graph");
    remaining -= count;
    return result;
  }
  template <typename T> T read() {
    static_assert(std::is_unsigned_v<T>);
    require(sizeof(T) <= remaining, "Truncated VNDB graph");
    unsigned char bytes[sizeof(T)];
    require(static_cast<bool>(file_.read(reinterpret_cast<char*>(bytes), sizeof(bytes))),
            "Truncated VNDB graph");
    remaining -= sizeof(T);
    T result = 0;
    for (size_t i = 0; i < sizeof(T); ++i) result |= static_cast<T>(bytes[i]) << (8 * i);
    return result;
  }
  size_t size() {
    const auto value = read<uint64_t>();
    require(value <= SIZE_MAX, "VNDB graph size exceeds this platform");
    return static_cast<size_t>(value);
  }
};

#ifdef __GLIBCXX__
inline constexpr uint32_t NATIVE_RNG = 2;
#else
inline constexpr uint32_t NATIVE_RNG = 3;
#endif

inline void check_rng(uint32_t kind, const std::string& bytes) {
  if (kind == 1 && bytes.empty()) return;
  require(kind == 2 || kind == 3, "Unsupported VNDB graph RNG encoding");
  std::istringstream input(bytes);
  input.imbue(std::locale::classic());
  std::string word;
  size_t count = 0;
  uint64_t value = 0;
  while (input >> word) {
    value = 0;
    require(!word.empty(), "Invalid VNDB graph RNG word");
    for (char digit : word) {
      require(digit >= '0' && digit <= '9', "Invalid VNDB graph RNG word");
      value = value * 10 + static_cast<unsigned>(digit - '0');
      require(value <= UINT32_MAX, "Invalid VNDB graph RNG word");
    }
    ++count;
  }
  require((kind == 2 && count == 625 && value <= 624) || (kind == 3 && count == 624),
          "Invalid VNDB graph RNG state length or position");
}

struct Data {
  uint32_t metric, rng_kind;
  size_t dim, count, capacity, m, ef_construction, ef_search;
  uint64_t seed, entry;
  int32_t max_level;
  std::vector<float> vectors;
  std::vector<uint64_t> ids;
  std::vector<int> levels;
  std::vector<bool> deleted;
  std::vector<std::vector<std::vector<size_t>>> neighbors;
  std::string rng;
};

inline Data read(std::ifstream& file) {
  Reader in(file);
  require(in.bytes(4) == "VNDB", "Invalid VNDB graph magic");
  require(in.read<uint32_t>() == 2, "Unsupported VNDB graph version");
  require(in.read<uint32_t>() == 1, "Unsupported VNDB graph kind");
  Data data;
  data.metric = in.read<uint32_t>();
  data.dim = in.size();
  data.count = in.size();
  data.capacity = in.size();
  data.m = in.size();
  data.ef_construction = in.size();
  data.ef_search = in.size();
  data.seed = in.read<uint64_t>();
  data.entry = in.read<uint64_t>();
  data.max_level = std::bit_cast<int32_t>(in.read<uint32_t>());
  data.rng_kind = in.read<uint32_t>();
  const size_t rng_length = in.size();
  require(rng_length <= 65536 && rng_length <= in.remaining, "Invalid VNDB graph RNG length");
  in.remaining -= rng_length;  // Node lengths must not consume the RNG section.
  require(data.metric <= 2 && data.dim > 0 && data.dim <= (SIZE_MAX - 24) / 4 &&
          data.capacity > 0 && data.capacity <= 100000000 && data.count <= data.capacity &&
          data.m >= 2 && data.m <= SIZE_MAX / 2 && data.ef_construction > 0 &&
          data.count <= in.remaining / (24 + 4 * data.dim), "Invalid VNDB graph parameters");
  require(data.capacity <= SIZE_MAX / data.dim, "VNDB graph capacity overflows");
  require(data.count ? data.entry < data.count && data.max_level >= 0 && data.max_level <= 32 :
          data.entry == UINT64_MAX && data.max_level == -1, "Invalid VNDB graph entry or level");
  data.vectors.reserve(data.count * data.dim);
  data.ids.reserve(data.count);
  data.levels.reserve(data.count);
  data.deleted.reserve(data.count);
  data.neighbors.resize(data.count);
  std::unordered_set<uint64_t> live_ids;
  int observed_max = -1;
  for (size_t slot = 0; slot < data.count; ++slot) {
    const auto id = in.read<uint64_t>();
    const auto level = in.read<uint32_t>();
    const auto flags = in.read<uint32_t>();
    require(level <= 32 && flags <= 1, "Invalid VNDB graph node level or flags");
    require(flags == 1 || live_ids.insert(id).second, "Duplicate live VNDB graph ID");
    data.ids.push_back(id);
    data.levels.push_back(static_cast<int>(level));
    data.deleted.push_back(flags == 1);
    observed_max = std::max(observed_max, static_cast<int>(level));
    for (size_t d = 0; d < data.dim; ++d) {
      const float value = std::bit_cast<float>(in.read<uint32_t>());
      require(std::isfinite(value), "Non-finite VNDB graph vector");
      data.vectors.push_back(value);
    }
    auto& layers = data.neighbors[slot];
    layers.resize(level + 1);
    for (size_t layer = 0; layer <= level; ++layer) {
      const size_t degree = in.size();
      require(degree <= (layer == 0 ? 2 * data.m : data.m) && degree <= in.remaining / 8,
              "Invalid VNDB graph degree");
      std::unordered_set<size_t> seen;
      layers[layer].reserve(degree);
      for (size_t n = 0; n < degree; ++n) {
        const size_t neighbor = in.size();
        require(neighbor < data.count && neighbor != slot && seen.insert(neighbor).second,
                "Invalid or duplicate VNDB graph neighbor");
        layers[layer].push_back(neighbor);
      }
    }
  }
  require(data.max_level == observed_max &&
          (data.count == 0 || data.levels[data.entry] == data.max_level),
          "VNDB graph entry/max_level disagrees with node levels");
  for (const auto& layers : data.neighbors)
    for (size_t layer = 0; layer < layers.size(); ++layer)
      for (size_t neighbor : layers[layer])
        require(data.levels[neighbor] >= static_cast<int>(layer), "VNDB graph edge above node level");
  require(in.remaining == 0, "Trailing bytes in VNDB graph");
  in.remaining = rng_length;
  data.rng = in.bytes(rng_length);
  check_rng(data.rng_kind, data.rng);
  return data;
}
} // namespace vanedb::detail::graph
