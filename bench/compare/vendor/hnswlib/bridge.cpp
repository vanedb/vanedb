#include "bridge.h"

#include "hnswlib/hnswlib.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

namespace {

void l2_normalize(float *v, size_t dim) {
  double sum = 0.0;
  for (size_t i = 0; i < dim; ++i) {
    sum += static_cast<double>(v[i]) * static_cast<double>(v[i]);
  }
  if (!(sum > 0.0) || !std::isfinite(sum)) {
    return;
  }
  const float inv = static_cast<float>(1.0 / std::sqrt(sum));
  for (size_t i = 0; i < dim; ++i) {
    v[i] *= inv;
  }
}

} // namespace

struct HnswBridge {
  size_t dim = 0;
  int metric = 0;
  std::unique_ptr<hnswlib::SpaceInterface<float>> space;
  std::unique_ptr<hnswlib::HierarchicalNSW<float>> index;
};

extern "C" HnswBridge *hnsw_bridge_create(size_t dim, size_t max_elements,
                                          size_t m, size_t ef_construction,
                                          int metric) {
  auto *bridge = new (std::nothrow) HnswBridge();
  if (!bridge) {
    return nullptr;
  }
  bridge->dim = dim;
  bridge->metric = metric;
  try {
    if (metric == 1) {
      bridge->space = std::make_unique<hnswlib::InnerProductSpace>(dim);
    } else {
      bridge->space = std::make_unique<hnswlib::L2Space>(dim);
    }
    bridge->index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
        bridge->space.get(), max_elements, m, ef_construction);
  } catch (...) {
    delete bridge;
    return nullptr;
  }
  return bridge;
}

extern "C" void hnsw_bridge_free(HnswBridge *index) { delete index; }

extern "C" int hnsw_bridge_add(HnswBridge *index, uint64_t id,
                               const float *vector) {
  if (!index || !index->index || !vector) {
    return -1;
  }
  try {
    if (index->metric == 1) {
      std::vector<float> copy(vector, vector + index->dim);
      l2_normalize(copy.data(), index->dim);
      index->index->addPoint(copy.data(), static_cast<hnswlib::labeltype>(id));
    } else {
      index->index->addPoint(vector, static_cast<hnswlib::labeltype>(id));
    }
  } catch (...) {
    return -1;
  }
  return 0;
}

extern "C" int hnsw_bridge_mark_deleted(HnswBridge *index, uint64_t id) {
  if (!index || !index->index) {
    return -1;
  }
  try {
    index->index->markDelete(static_cast<hnswlib::labeltype>(id));
  } catch (...) {
    return -1;
  }
  return 0;
}

extern "C" void hnsw_bridge_set_ef(HnswBridge *index, size_t ef) {
  if (index && index->index) {
    index->index->setEf(ef);
  }
}

extern "C" int hnsw_bridge_search(HnswBridge *index, const float *query,
                                  size_t k, uint64_t *out_ids) {
  if (!index || !index->index || !query || !out_ids) {
    return -1;
  }
  try {
    std::priority_queue<std::pair<float, hnswlib::labeltype>> result;
    if (index->metric == 1) {
      std::vector<float> copy(query, query + index->dim);
      l2_normalize(copy.data(), index->dim);
      result = index->index->searchKnn(copy.data(), k);
    } else {
      result = index->index->searchKnn(query, k);
    }
    std::vector<uint64_t> ids;
    ids.reserve(result.size());
    while (!result.empty()) {
      ids.push_back(static_cast<uint64_t>(result.top().second));
      result.pop();
    }
    std::reverse(ids.begin(), ids.end());
    const int n = static_cast<int>(ids.size());
    for (int i = 0; i < n; ++i) {
      out_ids[i] = ids[static_cast<size_t>(i)];
    }
    return n;
  } catch (...) {
    return -1;
  }
}

extern "C" int hnsw_bridge_save(HnswBridge *index, const char *path) {
  if (!index || !index->index || !path) {
    return -1;
  }
  try {
    index->index->saveIndex(path);
  } catch (...) {
    return -1;
  }
  return 0;
}

extern "C" HnswBridge *hnsw_bridge_load(const char *path, size_t dim,
                                        int metric) {
  if (!path || dim == 0) {
    return nullptr;
  }
  auto *bridge = new (std::nothrow) HnswBridge();
  if (!bridge) {
    return nullptr;
  }
  bridge->dim = dim;
  bridge->metric = metric;
  try {
    if (metric == 1) {
      bridge->space = std::make_unique<hnswlib::InnerProductSpace>(dim);
    } else {
      bridge->space = std::make_unique<hnswlib::L2Space>(dim);
    }
    bridge->index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
        bridge->space.get(), std::string(path));
  } catch (...) {
    delete bridge;
    return nullptr;
  }
  return bridge;
}
