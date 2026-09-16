#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct HnswBridge HnswBridge;

/* metric: 0 = L2 squared, 1 = cosine (inner-product space on normalized vectors) */
HnswBridge *hnsw_bridge_create(size_t dim, size_t max_elements, size_t m,
                               size_t ef_construction, int metric,
                               size_t random_seed);
void hnsw_bridge_free(HnswBridge *index);

int hnsw_bridge_add(HnswBridge *index, uint64_t id, const float *vector);
int hnsw_bridge_mark_deleted(HnswBridge *index, uint64_t id);
void hnsw_bridge_set_ef(HnswBridge *index, size_t ef);

/* Writes up to k ids into out_ids; returns the number written, or -1 on error. */
int hnsw_bridge_search(HnswBridge *index, const float *query, size_t k,
                       uint64_t *out_ids);

int hnsw_bridge_save(HnswBridge *index, const char *path);
HnswBridge *hnsw_bridge_load(const char *path, size_t dim, int metric);

#ifdef __cplusplus
}
#endif
