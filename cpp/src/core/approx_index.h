// VaneDB - Copyright (c) 2025 Anton Tsvetkov - MIT License
#pragma once
#include "distance_strategy.h"
#include "detail/file_utils.h"
#include "detail/graph_format.h"
#include "validation.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <queue>
#include <random>
#include <shared_mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
namespace vanedb {

namespace detail {
template <typename T> void write_bin(std::ofstream& f, const T& v) {
  f.write(reinterpret_cast<const char*>(&v), sizeof(T));
}
template <typename T> void read_bin(std::ifstream& f, T& v) {
  if (!f.read(reinterpret_cast<char*>(&v), sizeof(T)))
    throw std::runtime_error("Unexpected end of file or read error");
}
template <typename T> void write_vec(std::ofstream& f, const std::vector<T>& v) {
  write_bin(f, v.size());
  if (!v.empty()) f.write(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(T));
}
constexpr size_t MAX_VEC_SIZE = 100000000ULL;
constexpr size_t MAX_RNG_STATE_SIZE = 10000;  // Reasonable upper bound for serialized RNG state
template <typename T> void read_vec(std::ifstream& f, std::vector<T>& v) {
  size_t sz; read_bin(f, sz);
  if (sz > MAX_VEC_SIZE || sz > SIZE_MAX / sizeof(T))
    throw std::runtime_error("Corrupted file: vector too large");
  v.resize(sz);
  if (!v.empty() && !f.read(reinterpret_cast<char*>(v.data()), sz * sizeof(T)))
    throw std::runtime_error("Unexpected end of file or read error");
}
} // namespace detail

struct HNSWSearchResult {
  uint64_t id;
  float distance;
  bool operator<(const HNSWSearchResult& o) const {
    if (detail::distance_less(distance, o.distance)) return true;
    if (detail::distance_less(o.distance, distance)) return false;
    return id < o.id;
  }
  bool operator>(const HNSWSearchResult& o) const { return o < *this; }
};

class ApproxIndex {
  struct DerivedSizes {
    size_t vector_count;
    size_t m_max0;
  };

  static DerivedSizes checked_direct_sizes(size_t dimension, size_t max_elements, size_t M) {
    if (dimension == 0) throw std::invalid_argument("Dimension must be > 0");
    if (max_elements == 0) throw std::invalid_argument("max_elements must be > 0");
    if (M < 2) throw std::invalid_argument("M must be >= 2");
    if (max_elements > std::numeric_limits<size_t>::max() / dimension)
      throw std::invalid_argument("max_elements * dimension overflow");
    if (M > std::numeric_limits<size_t>::max() / 2)
      throw std::invalid_argument("M * 2 overflow");
    return {max_elements * dimension, M * 2};
  }

  static DerivedSizes checked_persisted_sizes(size_t dimension, size_t max_elements, size_t M) {
    if (dimension == 0) throw std::runtime_error("Corrupted file: invalid dimension");
    if (max_elements == 0) throw std::runtime_error("Corrupted file: invalid max_elements");
    if (M < 2) throw std::runtime_error("Corrupted file: invalid M");
    if (max_elements > std::numeric_limits<size_t>::max() / dimension)
      throw std::runtime_error("Corrupted file: max_elements * dimension overflow");
    if (M > std::numeric_limits<size_t>::max() / 2)
      throw std::runtime_error("Corrupted file: M * 2 overflow");
    return {max_elements * dimension, M * 2};
  }

  static size_t checked_persisted_product(size_t left, size_t right, const char* name) {
    if (right != 0 && left > std::numeric_limits<size_t>::max() / right)
      throw std::runtime_error(std::string("Corrupted file: ") + name + " overflow");
    return left * right;
  }

  ApproxIndex(size_t dimension, Metric metric, size_t max_elements, size_t M,
            size_t ef_construction, uint32_t seed, DerivedSizes sizes)
      : dim_(dimension), metric_(metric), dist_(metric, dimension),
        max_elements_(max_elements), M_(M), M_max_(M),
        M_max0_(sizes.m_max0), ef_construction_(std::max(ef_construction, M)), ef_search_(50),
        mult_(M > 1 ? 1.0 / std::log(static_cast<double>(M)) : 1.0), level_gen_(seed), serialization_seed_(seed) {
    vectors_.resize(sizes.vector_count);
    ext_ids_.resize(max_elements);
    levels_.resize(max_elements, 0);
    deleted_.resize(max_elements, false);
    neighbors_.resize(max_elements);
  }

public:
  static constexpr uint32_t MAGIC = 0x51565244;  // "QVRD" (legacy QuiverDB magic, retained for on-disk compat)
  static constexpr uint32_t VERSION = 3;  // Latest readable legacy version; save writes VNDB v2.
  static constexpr int MAX_LEVEL = 32;  // Reasonable upper bound for HNSW levels
  static constexpr size_t INVALID_ID = static_cast<size_t>(-1);  // Sentinel for empty entry point

  explicit ApproxIndex(size_t dimension, Metric metric = Metric::L2,
      size_t max_elements = 100000, size_t M = 16, size_t ef_construction = 200, uint32_t seed = 42)
      : ApproxIndex(dimension, metric, max_elements, M, ef_construction, seed,
                  checked_direct_sizes(dimension, max_elements, M)) {}

  // Thread-safety: global_mtx_ is the single sync point. add() holds it
  // exclusive; readers (search/size/contains/get_vector/save) hold it shared.
  // No per-node locks are needed because add() can never run concurrently
  // with any reader.
  void add(uint64_t id, const float* vec) {
    if (!vec) throw std::invalid_argument("Vector must not be null");
    detail::require_finite(vec, dim_, "Vector");
    std::unique_lock glock(global_mtx_);  // Exclusive: only one add() at a time
    if (id_map_.count(id)) throw std::invalid_argument("ID " + std::to_string(id) + " exists");
    if (count_ >= max_elements_) throw std::runtime_error("ApproxIndex full");

    const size_t next = count_.load() + 1;
    if (neighbors_.size() < next) {
      // Loaded VNDB files allocate stored slots, independent of the capacity
      // hint. Grow within the existing hard limit before publishing a slot.
      vectors_.resize(next * dim_);
      ext_ids_.resize(next);
      levels_.resize(next);
      deleted_.resize(next, false);
      neighbors_.resize(next);
    }
    size_t iid = count_++;
    id_map_[id] = iid;
    ext_ids_[iid] = id;
    std::copy_n(vec, dim_, vectors_.begin() + iid * dim_);

    int level = get_level();
    levels_[iid] = level;
    neighbors_[iid].resize(level + 1);
    for (int l = 0; l <= level; ++l)
      neighbors_[iid][l].reserve(l == 0 ? M_max0_ : M_max_);

    if (ep_.load() == INVALID_ID) { ep_.store(iid); max_level_.store(level); return; }

    size_t curr = ep_.load();
    int cur_max_level = max_level_.load();
    if (level < cur_max_level) {
      float d = dist_(vec, get_vec(curr));
      for (int l = cur_max_level; l > level; --l) {
        bool changed = true;
        while (changed) {
          changed = false;
          for (size_t n : neighbors_[curr][l]) {
            float nd = dist_(vec, get_vec(n));
            if (detail::distance_less(nd, d)) { d = nd; curr = n; changed = true; }
          }
        }
      }
    }

    for (int l = std::min(level, cur_max_level); l >= 0; --l) {
      auto top = search_layer(vec, curr, ef_construction_, l);
      // Capture the nearest candidate before `select_neighbors` drains `top`.
      // Reading it afterwards, as this loop used to, always saw an empty heap,
      // so the entry point never advanced and every layer restarted its beam
      // search from the node the greedy descent found.
      size_t next_entry = curr;
      {
        float best = std::numeric_limits<float>::infinity();
        MaxHeap scan = top;
        while (!scan.empty()) {
          if (detail::distance_less(scan.top().first, best)) {
            best = scan.top().first;
            next_entry = scan.top().second;
          }
          scan.pop();
        }
      }
      auto sel = select_neighbors(top, M_, l);
      neighbors_[iid][l] = std::move(sel);

      size_t max_conn = l == 0 ? M_max0_ : M_max_;
      for (size_t nid : neighbors_[iid][l]) {
        auto& nc = neighbors_[nid][l];
        if (nc.size() < max_conn) { nc.push_back(iid); }
        else {
          float d2new = dist_(get_vec(nid), vec);
          std::vector<std::pair<float, size_t>> cands;
          cands.reserve(nc.size() + 1);
          for (size_t c : nc) cands.emplace_back(dist_(get_vec(nid), get_vec(c)), c);
          cands.emplace_back(d2new, iid);
          std::sort(cands.begin(), cands.end(), detail::DistanceIdLess{});
          nc.clear();
          for (size_t i = 0; i < max_conn && i < cands.size(); ++i) nc.push_back(cands[i].second);
        }
      }
      curr = next_entry;
    }
    if (level > cur_max_level) { ep_.store(iid); max_level_.store(level); }
  }

  std::vector<HNSWSearchResult> search(const float* query, size_t k) const {
    if (!query) throw std::invalid_argument("Query must not be null");
    detail::require_finite(query, dim_, "Query");
    if (k == 0) throw std::invalid_argument("k must be > 0");
    std::shared_lock glock(global_mtx_);
    if (id_map_.empty()) return {};

    size_t curr = ep_.load();
    float d = dist_(query, get_vec(curr));
    for (int l = max_level_.load(); l > 0; --l) {
      bool changed = true;
      while (changed) {
        changed = false;
        if (static_cast<int>(neighbors_[curr].size()) <= l) continue;
        for (size_t n : neighbors_[curr][l]) {
          float nd = dist_(query, get_vec(n));
          if (detail::distance_less(nd, d)) { d = nd; curr = n; changed = true; }
        }
      }
    }

    const size_t ef = std::max(ef_search_.load(std::memory_order_relaxed), k);
    auto top = id_map_.size() == count_.load(std::memory_order_relaxed)
        ? search_layer(query, curr, ef, 0) : search_layer<true>(query, curr, ef, 0);
    std::vector<std::pair<float, size_t>> temp;
    while (!top.empty()) { temp.push_back(top.top()); top.pop(); }
    std::sort(temp.begin(), temp.end(), detail::DistanceIdLess{});

    std::vector<HNSWSearchResult> res;
    res.reserve(std::min(k, temp.size()));
    for (size_t i = 0; i < k && i < temp.size(); ++i)
      res.push_back({ext_ids_[temp[i].second], temp[i].first});
    return res;
  }

  void set_ef_search(size_t ef) {
    if (ef == 0) throw std::invalid_argument("ef_search must be > 0");
    ef_search_.store(ef, std::memory_order_relaxed);
  }
  size_t get_ef_search() const { return ef_search_.load(std::memory_order_relaxed); }
  size_t size() const { std::shared_lock lk(global_mtx_); return id_map_.size(); }
  size_t dimension() const { return dim_; }
  size_t capacity() const { return max_elements_; }
  bool contains(uint64_t id) const { std::shared_lock lk(global_mtx_); return id_map_.count(id); }

  std::vector<float> get_vector(uint64_t id) const {
    std::shared_lock lk(global_mtx_);
    auto it = id_map_.find(id);
    if (it == id_map_.end()) throw std::runtime_error("ID not found: " + std::to_string(id));
    const float* p = vectors_.data() + it->second * dim_;
    return std::vector<float>(p, p + dim_);
  }

  void save(const std::string& filename) const {
    std::shared_lock glock(global_mtx_);
    std::string tmp = detail::temp_path_for(filename);
    std::ofstream f(tmp, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + tmp);
    try {
      using detail::graph::write;
      const size_t count = count_.load();
      const bool preserve_rng = origin_count_ == count;
      std::ostringstream rng_stream;
      rng_stream.imbue(std::locale::classic());
      if (!preserve_rng) rng_stream << level_gen_;
      const std::string rng = preserve_rng ? origin_rng_ : rng_stream.str();
      f.write("VNDB", 4);
      for (uint32_t value : {2u, 1u, static_cast<uint32_t>(metric_)}) write(f, value);
      for (uint64_t value : {static_cast<uint64_t>(dim_), static_cast<uint64_t>(count),
           static_cast<uint64_t>(max_elements_), static_cast<uint64_t>(M_),
           static_cast<uint64_t>(ef_construction_), static_cast<uint64_t>(ef_search_.load()),
           serialization_seed_, count == 0 ? UINT64_MAX : static_cast<uint64_t>(ep_.load())})
        write(f, value);
      write(f, std::bit_cast<uint32_t>(static_cast<int32_t>(max_level_.load())));
      write(f, preserve_rng ? origin_rng_kind_ : detail::graph::NATIVE_RNG);
      write(f, static_cast<uint64_t>(rng.size()));
      for (size_t slot = 0; slot < count; ++slot) {
        write(f, ext_ids_[slot]);
        write(f, static_cast<uint32_t>(levels_[slot]));
        write(f, static_cast<uint32_t>(deleted_[slot]));
        detail::graph::write_floats(f, get_vec(slot), dim_);
        for (const auto& layer : neighbors_[slot]) {
          write(f, static_cast<uint64_t>(layer.size()));
          for (size_t neighbor : layer) write(f, static_cast<uint64_t>(neighbor));
        }
      }
      f.write(rng.data(), rng.size());
      f.flush();
      if (!f) { std::filesystem::remove(tmp); throw std::runtime_error("Write failed: " + tmp); }
      f.close();  // close before fsync_file (see file_utils.h: Windows lock contract)
      detail::fsync_file(tmp);
      std::filesystem::rename(tmp, filename);
    } catch (...) { f.close(); std::filesystem::remove(tmp); throw; }
  }

  static std::unique_ptr<ApproxIndex> load(const std::string& filename) {
    std::ifstream f(filename, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + filename);
    char prefix[4];
    if (!f.read(prefix, 4)) throw std::runtime_error("Truncated graph file");
    f.seekg(0);
    if (std::string(prefix, 4) == "VNDB") return load_vndb(f);
    uint32_t magic, ver;
    detail::read_bin(f, magic);
    if (magic != MAGIC) throw std::runtime_error("Invalid magic");
    detail::read_bin(f, ver);
    if (ver < 1 || ver > VERSION) throw std::runtime_error("Unsupported version");

    size_t dim, max_el, M, ef_con, ef_s; uint32_t met; double mult;
    detail::read_bin(f, dim);
    detail::read_bin(f, met);
    if (met > 2) throw std::runtime_error("Corrupted file: invalid metric");
    detail::read_bin(f, max_el);
    detail::read_bin(f, M);
    detail::read_bin(f, ef_con);
    detail::read_bin(f, ef_s);
    detail::read_bin(f, mult);

    // Validate every derived allocation size while this input is still just a
    // file header. Passing the products into the private constructor keeps the
    // checks ahead of allocation and avoids recomputing them unchecked.
    const DerivedSizes sizes = checked_persisted_sizes(dim, max_el, M);

    size_t cnt, ep_val;
    int max_level_val;
    detail::read_bin(f, cnt);
    const size_t live_vector_count =
        checked_persisted_product(cnt, dim, "count * dimension");
    if (cnt > max_el) throw std::runtime_error("Corrupted file: count exceeds max_elements");
    detail::read_bin(f, ep_val);
    detail::read_bin(f, max_level_val);
    // Validate ep_ and max_level_
    if (cnt > 0) {
      if (ep_val >= cnt) throw std::runtime_error("Corrupted file: invalid entry point");
      if (max_level_val < 0 || max_level_val > MAX_LEVEL)
        throw std::runtime_error("Corrupted file: invalid max_level");
    } else {
      // Empty index must have invalid entry point
      if (ep_val != INVALID_ID)
        throw std::runtime_error("Corrupted file: non-empty entry point for empty index");
    }
    // Array lengths are version-specific: v1/v2 stored full pre-allocated
    // arrays, v3 stores only the `cnt` live entries.
    const size_t stored = ver >= 3 ? cnt : max_el;
    const size_t stored_vector_count =
        ver >= 3 ? live_vector_count : sizes.vector_count;

    auto idx = std::unique_ptr<ApproxIndex>(
        new ApproxIndex(dim, static_cast<Metric>(met), max_el, M, ef_con, 42, sizes));
    idx->ef_search_.store(ef_s);
    // mult is derived from M by the constructor. Trusting the stored value
    // can produce negative or non-finite levels on the next insertion.
    idx->count_.store(cnt);
    idx->ep_.store(ep_val);
    idx->max_level_.store(max_level_val);
    detail::read_vec(f, idx->vectors_);
    detail::read_vec(f, idx->ext_ids_);
    detail::read_vec(f, idx->levels_);
    if (idx->vectors_.size() != stored_vector_count)
      throw std::runtime_error("Corrupted file: vectors length mismatch");
    for (size_t i = 0; i < live_vector_count; ++i) {
      if (!std::isfinite(idx->vectors_[i]))
        throw std::runtime_error("Corrupted file: vector values must be finite");
    }
    if (idx->ext_ids_.size() != stored || idx->levels_.size() != stored)
      throw std::runtime_error("Corrupted file: ext_ids/levels length mismatch");
    const int observed_max = cnt == 0 ? -1 :
        *std::max_element(idx->levels_.begin(), idx->levels_.begin() + cnt);
    if (max_level_val != observed_max || (cnt > 0 && idx->levels_[ep_val] != max_level_val))
      throw std::runtime_error("Corrupted file: entry point/max_level disagrees with node levels");
    for (size_t i = 0; i < cnt; ++i) {
      if (idx->levels_[i] < 0 || idx->levels_[i] > MAX_LEVEL)
        throw std::runtime_error("Corrupted file: invalid node level");
    }
    // Re-expand to the pre-allocated capacity layout the index expects
    // (no-ops for v1/v2).
    idx->vectors_.resize(sizes.vector_count);
    idx->ext_ids_.resize(max_el);
    idx->levels_.resize(max_el);

    size_t msz;
    detail::read_bin(f, msz);
    if (msz > cnt) throw std::runtime_error("Corrupted file: id_map size exceeds count");
    idx->id_map_.reserve(msz);
    for (size_t i = 0; i < msz; ++i) {
      uint64_t k; size_t v;
      detail::read_bin(f, k);
      detail::read_bin(f, v);
      if (v >= cnt) throw std::runtime_error("Corrupted file: invalid internal index in id_map");
      idx->id_map_[k] = v;
    }
    // Every live slot must be reachable by its own external id. Combined with
    // a size equal to the live count this forces a bijection, rejecting
    // key/value mismatches, duplicate external ids, duplicated internal ids,
    // and missing entries. Checking only size <= count and value range
    // accepted a file whose external id resolved to another slot's vector
    // (vanedb#42 / vanedb-cpp#38).
    if (idx->id_map_.size() != cnt)
      throw std::runtime_error("Corrupted file: id_map size does not match count");
    for (size_t i = 0; i < cnt; ++i) {
      auto entry = idx->id_map_.find(idx->ext_ids_[i]);
      if (entry == idx->id_map_.end() || entry->second != i)
        throw std::runtime_error("Corrupted file: id_map is not consistent with ext_ids");
    }

    size_t nsz;
    detail::read_bin(f, nsz);
    if (nsz != stored) throw std::runtime_error("Corrupted file: neighbors length mismatch");
    idx->neighbors_.resize(nsz);
    for (size_t i = 0; i < nsz; ++i) {
      size_t lsz;
      detail::read_bin(f, lsz);
      if (lsz > static_cast<size_t>(MAX_LEVEL) + 1) throw std::runtime_error("Corrupted file: too many levels");
      if (i < cnt && lsz != static_cast<size_t>(idx->levels_[i]) + 1)
        throw std::runtime_error("Corrupted file: neighbor layers disagree with node level");
      idx->neighbors_[i].resize(lsz);
      for (size_t l = 0; l < lsz; ++l) {
        detail::read_vec(f, idx->neighbors_[i][l]);
        const auto& layer = idx->neighbors_[i][l];
        if (layer.size() > (l == 0 ? sizes.m_max0 : M))
          throw std::runtime_error("Corrupted file: neighbor degree exceeds layer limit");
        std::unordered_set<size_t> seen;
        for (size_t nid : layer) {
          if (nid >= cnt || nid == i || !seen.insert(nid).second ||
              idx->levels_[nid] < static_cast<int>(l))
            throw std::runtime_error("Corrupted file: invalid or duplicate neighbor in layer");
        }
      }
    }

    idx->neighbors_.resize(max_el);

    // Restore RNG state for deterministic behavior (v2+)
    if (ver >= 2) {
      size_t rng_state_size;
      detail::read_bin(f, rng_state_size);
      if (rng_state_size > detail::MAX_RNG_STATE_SIZE)
        throw std::runtime_error("Corrupted file: RNG state too large");
      std::string rng_state(rng_state_size, '\0');
      if (!f.read(rng_state.data(), rng_state_size))
        throw std::runtime_error("Unexpected end of file or read error");
      std::stringstream rng_ss(rng_state);
      rng_ss >> idx->level_gen_;
      if (rng_ss.fail()) throw std::runtime_error("Corrupted file: invalid RNG state");
    }
    // Note: v1 files don't have RNG state, level_gen_ keeps default initialization
    return idx;
  }

private:
  static std::unique_ptr<ApproxIndex> load_vndb(std::ifstream& file) {
    auto data = detail::graph::read(file);
    const auto sizes = checked_persisted_sizes(data.dim, data.capacity, data.m);
    auto index = std::unique_ptr<ApproxIndex>(new ApproxIndex(
        data.dim, static_cast<Metric>(data.metric), 0, data.m,
        data.ef_construction, static_cast<uint32_t>(data.seed), {0, sizes.m_max0}));
    index->max_elements_ = data.capacity;
    index->ef_construction_ = data.ef_construction;
    index->ef_search_.store(data.ef_search);
    index->serialization_seed_ = data.seed;
    index->count_.store(data.count);
    index->ep_.store(data.count == 0 ? INVALID_ID : static_cast<size_t>(data.entry));
    index->max_level_.store(data.max_level);
    index->vectors_ = std::move(data.vectors);
    index->ext_ids_ = std::move(data.ids);
    index->levels_ = std::move(data.levels);
    index->deleted_ = std::move(data.deleted);
    index->neighbors_ = std::move(data.neighbors);
    for (size_t slot = 0; slot < data.count; ++slot)
      if (!index->deleted_[slot]) index->id_map_.emplace(index->ext_ids_[slot], slot);
    if (data.rng_kind == detail::graph::NATIVE_RNG) {
      std::istringstream input(data.rng);
      input.imbue(std::locale::classic());
      detail::graph::require(static_cast<bool>(input >> index->level_gen_),
                             "Invalid native VNDB graph RNG state");
    } else {
      for (size_t slot = 0; slot < data.count; ++slot) index->get_level();
    }
    index->origin_rng_kind_ = data.rng_kind;
    index->origin_rng_ = std::move(data.rng);
    index->origin_count_ = data.count;
    return index;
  }

  static constexpr double MIN_LEVEL_RANDOM = 1e-9;  // Clamp floor for level generation RNG
  using DistanceId = std::pair<float, size_t>;
  using MaxHeap = std::priority_queue<DistanceId, std::vector<DistanceId>, detail::DistanceIdLess>;
  using MinHeap = std::priority_queue<DistanceId, std::vector<DistanceId>, detail::DistanceIdGreater>;

  int get_level() {
    std::uniform_real_distribution<double> d(0.0, 1.0);
    double r = std::max(d(level_gen_), MIN_LEVEL_RANDOM);
    int level = static_cast<int>(-std::log(r) * mult_);
    return std::min(level, MAX_LEVEL);
  }

  const float* get_vec(size_t iid) const { return vectors_.data() + iid * dim_; }

  template <bool LiveOnly = false>
  MaxHeap search_layer(const float* q, size_t ep, size_t ef, int level) const {
    // Versioned thread-local visited bitmap. vis[i] == vis_epoch means
    // visited; bumping the epoch each call replaces the per-search O(N)
    // zero-init a fresh bitmap would need with one O(count_) fill every
    // 65k searches when the uint16_t epoch wraps. Buffer is shared across
    // ApproxIndex instances on a thread (monotonic epoch keeps cross-index
    // marks distinct) and is never shrunk.
    //
    // Relaxed load on count_ is safe: every caller holds global_mtx_
    // (exclusive in add(), shared in search()).
    static thread_local std::vector<uint16_t> vis;
    static thread_local uint16_t vis_epoch = 0;
    const size_t total = count_.load(std::memory_order_relaxed);
    // The entry is a stored node, possibly a tombstone. load() validates it;
    // this hot-path guard also catches in-memory corruption or future
    // call-site bugs. Defensive — unreachable by construction in tests.
    if (ep >= total) [[unlikely]]  // LCOV_EXCL_LINE
      throw std::logic_error("ApproxIndex::search_layer: entry point out of range");  // LCOV_EXCL_LINE
    if (vis.size() < total) vis.resize(total, 0);
    if (++vis_epoch == 0) {
      std::fill(vis.begin(), vis.end(), 0);
      vis_epoch = 1;
    }
    vis[ep] = vis_epoch;
    MinHeap cands;
    MaxHeap res;
    float d = dist_(q, get_vec(ep));
    cands.emplace(d, ep);
    if (!LiveOnly || !deleted_[ep]) res.emplace(d, ep);
    float lb = d;

    while (!cands.empty()) {
      auto [cd, cid] = cands.top();
      if (detail::distance_less(lb, cd) && res.size() >= ef) break;
      cands.pop();
      if (static_cast<int>(neighbors_[cid].size()) <= level) continue;
      for (size_t n : neighbors_[cid][level]) {
        if (vis[n] == vis_epoch) continue;
        vis[n] = vis_epoch;
        float nd = dist_(q, get_vec(n));
        if (res.size() < ef || detail::distance_less(nd, lb)) {
          cands.emplace(nd, n);
          if (!LiveOnly || !deleted_[n]) res.emplace(nd, n);
          if (res.size() > ef) res.pop();
          if (!res.empty()) lb = res.top().first;
        }
      }
    }
    return res;
  }

  std::vector<size_t> select_neighbors(MaxHeap& cands, size_t M, int /*level*/) const {
    if (cands.size() <= M) {
      std::vector<size_t> r;
      r.reserve(cands.size());
      while (!cands.empty()) { r.push_back(cands.top().second); cands.pop(); }
      return r;
    }
    std::vector<std::pair<float, size_t>> sorted;
    sorted.reserve(cands.size());
    while (!cands.empty()) { sorted.push_back(cands.top()); cands.pop(); }
    std::sort(sorted.begin(), sorted.end(), detail::DistanceIdLess{});

    std::vector<size_t> r;
    r.reserve(M);
    for (auto& [dq, cid] : sorted) {
      if (r.size() >= M) break;
      bool ok = true;
      for (size_t s : r)
        if (detail::distance_less(dist_(get_vec(cid), get_vec(s)), dq)) { ok = false; break; }
      if (ok) r.push_back(cid);
    }
    if (r.size() < M) {
      for (auto& p : sorted) {
        if (r.size() >= M) break;
        if (std::find(r.begin(), r.end(), p.second) == r.end()) r.push_back(p.second);
      }
    }
    return r;
  }

  size_t dim_;
  Metric metric_;
  DistanceComputer dist_;
  size_t max_elements_, M_, M_max_, M_max0_, ef_construction_;
  std::atomic<size_t> ef_search_;  // Atomic for thread-safe reads during search
  double mult_;
  std::mt19937 level_gen_;
  uint64_t serialization_seed_;
  size_t origin_count_ = INVALID_ID;
  uint32_t origin_rng_kind_ = detail::graph::NATIVE_RNG;
  std::string origin_rng_;  // Retain foreign continuation metadata until insertion.
  std::vector<bool> deleted_;  // Loaded tombstones: traversable, absent from results.

  std::vector<float> vectors_;
  std::vector<uint64_t> ext_ids_;
  std::unordered_map<uint64_t, size_t> id_map_;
  std::vector<int> levels_;
  std::vector<std::vector<std::vector<size_t>>> neighbors_;
  std::atomic<size_t> ep_{INVALID_ID};
  std::atomic<int> max_level_{-1};
  std::atomic<size_t> count_{0};
  mutable std::shared_mutex global_mtx_;
};

} // namespace vanedb
