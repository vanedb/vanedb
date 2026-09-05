#include "capi/vanedb_capi.h"
#include "core/store.h"
#include "core/index.h"
#include "core/disk_store.h"

using namespace vanedb;

namespace {
Metric to_metric(vanedb_metric m) {
  switch (m) {
    case VANEDB_COSINE: return Metric::COSINE;
    case VANEDB_DOT:    return Metric::DOT;
    case VANEDB_L2:
    default:            return Metric::L2;
  }
}
} // namespace

extern "C" {

float vanedb_cpp_l2_sq(const float* a, const float* b, size_t dim) {
  return l2_sq(a, b, dim);
}
float vanedb_cpp_cosine_distance(const float* a, const float* b, size_t dim) {
  return cosine_distance(a, b, dim);
}
float vanedb_cpp_dot_product(const float* a, const float* b, size_t dim) {
  return dot_product(a, b, dim);
}

vanedb_cpp_store* vanedb_cpp_store_new(size_t dim, vanedb_metric metric) {
  try { return reinterpret_cast<vanedb_cpp_store*>(new Store(dim, to_metric(metric))); }
  catch (...) { return nullptr; }
}
int vanedb_cpp_store_add(vanedb_cpp_store* s, uint64_t id, const float* v) {
  if (!s) return 1;
  try { reinterpret_cast<Store*>(s)->add(id, v); return 0; }
  catch (...) { return 1; }
}
size_t vanedb_cpp_store_search(vanedb_cpp_store* s, const float* q, size_t k,
                               uint64_t* out_ids, float* out_dists) {
  if (!s) return 0;
  try {
    auto res = reinterpret_cast<Store*>(s)->search(q, k);
    size_t n = res.size() < k ? res.size() : k;
    for (size_t i = 0; i < n; ++i) { out_ids[i] = res[i].id; out_dists[i] = res[i].distance; }
    return n;
  } catch (...) { return 0; }
}
void vanedb_cpp_store_free(vanedb_cpp_store* s) {
  delete reinterpret_cast<Store*>(s);
}

vanedb_cpp_index* vanedb_cpp_index_new(size_t dim, vanedb_metric metric, size_t capacity,
                                     size_t M, size_t ef_construction, uint64_t seed) {
  try {
    return reinterpret_cast<vanedb_cpp_index*>(
      // seed is uint64_t in the ABI (Rust parity) but the core takes uint32_t; high bits are dropped.
      new Index(dim, to_metric(metric), capacity, M, ef_construction,
                    static_cast<uint32_t>(seed)));
  } catch (...) { return nullptr; }
}
int vanedb_cpp_index_add(vanedb_cpp_index* h, uint64_t id, const float* v) {
  if (!h) return 1;
  try { reinterpret_cast<Index*>(h)->add(id, v); return 0; }
  catch (...) { return 1; }
}
size_t vanedb_cpp_index_search(vanedb_cpp_index* h, const float* q, size_t k, size_t ef_search,
                              uint64_t* out_ids, float* out_dists) {
  if (!h) return 0;
  try {
    auto* idx = reinterpret_cast<Index*>(h);
    idx->set_ef_search(ef_search);
    auto res = idx->search(q, k);
    size_t n = res.size() < k ? res.size() : k;
    for (size_t i = 0; i < n; ++i) { out_ids[i] = res[i].id; out_dists[i] = res[i].distance; }
    return n;
  } catch (...) { return 0; }
}
int vanedb_cpp_index_save(vanedb_cpp_index* h, const char* path) {
  if (!h) return 1;
  if (!path) return 1;
  try { reinterpret_cast<Index*>(h)->save(path); return 0; }
  catch (...) { return 1; }
}
vanedb_cpp_index* vanedb_cpp_index_load(const char* path) {
  if (!path) return nullptr;
  try { return reinterpret_cast<vanedb_cpp_index*>(Index::load(path).release()); }
  catch (...) { return nullptr; }
}
void vanedb_cpp_index_free(vanedb_cpp_index* h) {
  delete reinterpret_cast<Index*>(h);
}

int vanedb_cpp_disk_build(const char* path, size_t dim, vanedb_metric metric,
                          const uint64_t* ids, const float* vecs, size_t n) {
  if (!path) return 1;
  try {
    DiskStoreBuilder b(dim, to_metric(metric));
    for (size_t i = 0; i < n; ++i) b.add(ids[i], vecs + i * dim);
    b.save(path);
    return 0;
  } catch (...) { return 1; }
}
vanedb_cpp_disk* vanedb_cpp_disk_open(const char* path) {
  if (!path) return nullptr;
  try { return reinterpret_cast<vanedb_cpp_disk*>(new DiskStore(path)); }
  catch (...) { return nullptr; }
}
size_t vanedb_cpp_disk_search(vanedb_cpp_disk* m, const float* q, size_t k,
                              uint64_t* out_ids, float* out_dists) {
  if (!m) return 0;
  try {
    auto res = reinterpret_cast<DiskStore*>(m)->search(q, k);
    size_t n = res.size() < k ? res.size() : k;
    for (size_t i = 0; i < n; ++i) { out_ids[i] = res[i].id; out_dists[i] = res[i].distance; }
    return n;
  } catch (...) { return 0; }
}
void vanedb_cpp_disk_free(vanedb_cpp_disk* m) {
  delete reinterpret_cast<DiskStore*>(m);
}

}
