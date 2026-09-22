/* The same consumer contract runs on desktop, iOS, and Android. */
#include "vanedb_rs_capi.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef VANEDB_COEXISTENCE
extern uint32_t independent_rust_exercise(void);
extern bool independent_rust_filter(uint64_t id, void *data);
#endif

#define CHECK(expr) do { if (!(expr)) { \
    fprintf(stderr, "line %d: %s failed\n", __LINE__, #expr); exit(1); \
} } while (0)

/* Handles are ids, not pointers: a stale, truncated or foreign id is refused
 * with VANEDB_RS_INVALID_HANDLE and never dereferenced (RFC 0002 stage 1). */
static void reject_bad_handles(vanedb_rs_store live, vanedb_rs_index other_kind) {
    const vanedb_rs_store truncated = (vanedb_rs_store)(uint32_t)live;
    uint64_t found[1];
    float distances[1];
    const float query[] = {1, 0, 0};
    CHECK(vanedb_rs_store_len(truncated) == 0);
    CHECK(vanedb_rs_last_error() == VANEDB_RS_INVALID_HANDLE);
    CHECK(vanedb_rs_store_search(truncated, query, 1, found, distances) == 0);
    CHECK(vanedb_rs_last_error() == VANEDB_RS_INVALID_HANDLE);
    CHECK(vanedb_rs_store_len(other_kind) == 0);
    CHECK(vanedb_rs_last_error() == VANEDB_RS_INVALID_HANDLE);
    CHECK(vanedb_rs_store_len(UINT64_C(0xDEADBEEFCAFEF00D)) == 0);
    CHECK(vanedb_rs_last_error() == VANEDB_RS_INVALID_HANDLE);
    CHECK(vanedb_rs_store_len(VANEDB_RS_NULL_HANDLE) == 0);
    CHECK(vanedb_rs_last_error() == VANEDB_RS_NULL_ARGUMENT);
    /* The live handle is untouched by any of that. */
    CHECK(vanedb_rs_store_len(live) == 2);
    CHECK(vanedb_rs_last_error() == VANEDB_RS_OK);
}

static void exercise(uint32_t metric, const char *directory) {
    const uint64_t ids[] = {UINT64_MAX, (UINT64_C(1) << 53) + 1};
    const uint64_t conflicting_ids[] = {5, UINT64_MAX};
    const float vectors[] = {1, 0, 0, 0, 1, 0};
    const float replacement[] = {0, 0, 1};
    const float expected_distance = metric == VANEDB_RS_DOT ? -1 : 0;
    uint64_t found[2];
    float distances[2], vector[3];
    char path[4096];
    const uintptr_t handles_before = vanedb_rs_handle_count();

    vanedb_rs_store flat = vanedb_rs_store_new(3, metric);
    CHECK(flat != VANEDB_RS_NULL_HANDLE);
    CHECK(vanedb_rs_store_add_batch(flat, ids, vectors, 2) == 0);
    CHECK(vanedb_rs_store_add_batch(flat, conflicting_ids, vectors, 2) != 0);
    CHECK(!vanedb_rs_store_contains(flat, 5));
    CHECK(vanedb_rs_store_len(flat) == 2);
    CHECK(vanedb_rs_store_dimension(flat) == 3);
    CHECK(vanedb_rs_store_search(flat, vectors, 2, found, distances) == 2);
    CHECK(found[0] == ids[0] && fabsf(distances[0] - expected_distance) < 1e-6f);
    CHECK(vanedb_rs_store_get(flat, ids[0], vector) == 0);
    CHECK(memcmp(vector, vectors, sizeof(vector)) == 0);

    vanedb_rs_index graph = vanedb_rs_index_new(3, metric, 2, 4, 16, 42);
    CHECK(graph != VANEDB_RS_NULL_HANDLE);
    reject_bad_handles(flat, graph);

    CHECK(vanedb_rs_store_remove(flat, ids[0]) == 0);
    CHECK(!vanedb_rs_store_contains(flat, ids[0]));
    CHECK(vanedb_rs_store_len(flat) == 1);
    vanedb_rs_store_free(flat);
    /* Use after free and double free are refused, not undefined. */
    CHECK(vanedb_rs_store_len(flat) == 0);
    CHECK(vanedb_rs_last_error() == VANEDB_RS_INVALID_HANDLE);
    vanedb_rs_store_free(flat);
    CHECK(vanedb_rs_last_error() == VANEDB_RS_INVALID_HANDLE);

    CHECK(vanedb_rs_index_add_batch(graph, ids, vectors, 2) == 0);
    CHECK(vanedb_rs_index_add_batch(graph, conflicting_ids, vectors, 2) != 0);
    CHECK(!vanedb_rs_index_contains(graph, 5));
    CHECK(vanedb_rs_index_len(graph) == 2);
    CHECK(vanedb_rs_index_dimension(graph) == 3);
    CHECK(vanedb_rs_index_search(graph, vectors, 2, 16, found, distances) == 2);
    CHECK(found[0] == ids[0] && fabsf(distances[0] - expected_distance) < 1e-6f);
    CHECK(vanedb_rs_index_upsert(graph, ids[0], replacement) == 0);
    CHECK(vanedb_rs_index_remove(graph, ids[1]) == 0);
    CHECK(!vanedb_rs_index_contains(graph, ids[1]));
    CHECK(vanedb_rs_index_tombstones(graph) > 0);
    CHECK(vanedb_rs_index_compact(graph) == 0);
    CHECK(vanedb_rs_index_tombstones(graph) == 0);
    int written = snprintf(path, sizeof(path), "%s/graph-%u.bin", directory, (unsigned)metric);
    CHECK(written > 0 && (size_t)written < sizeof(path));
    CHECK(vanedb_rs_index_save(graph, path) == 0);
    vanedb_rs_index_free(graph);
    graph = vanedb_rs_index_load(path);
    CHECK(graph != VANEDB_RS_NULL_HANDLE && vanedb_rs_index_len(graph) == 1);
    CHECK(vanedb_rs_index_get_vector(graph, ids[0], vector) == 0);
    CHECK(memcmp(vector, replacement, sizeof(vector)) == 0);
    CHECK(vanedb_rs_index_search(graph, replacement, 2, 16, found, distances) == 1);
    CHECK(found[0] == ids[0] && fabsf(distances[0] - expected_distance) < 1e-6f);
    {
        uintptr_t needed = 0, wrote = 0;
        uint8_t *buf;
        vanedb_rs_index from_buf;
        CHECK(vanedb_rs_index_save_to_buffer(graph, NULL, 0, &needed) == 0);
        CHECK(needed > 4);
        buf = (uint8_t *)malloc(needed);
        CHECK(buf != NULL);
        wrote = needed;
        CHECK(vanedb_rs_index_save_to_buffer(graph, buf, needed, &wrote) == 0);
        CHECK(wrote == needed);
        from_buf = vanedb_rs_index_load_from_buffer(buf, wrote);
        CHECK(from_buf != VANEDB_RS_NULL_HANDLE && vanedb_rs_index_len(from_buf) == 1);
        vanedb_rs_index_free(from_buf);
        free(buf);
    }
    vanedb_rs_index_free(graph);
    CHECK(remove(path) == 0);

    written = snprintf(path, sizeof(path), "%s/disk-%u.bin", directory, (unsigned)metric);
    CHECK(written > 0 && (size_t)written < sizeof(path));
    CHECK(vanedb_rs_disk_build(path, 3, metric, ids, vectors, 2) == 0);
    vanedb_rs_disk disk = vanedb_rs_disk_open(path);
    CHECK(disk != VANEDB_RS_NULL_HANDLE);
    CHECK(vanedb_rs_disk_len(disk) == 2 && vanedb_rs_disk_dimension(disk) == 3);
    CHECK(vanedb_rs_disk_contains(disk, ids[0]));
    CHECK(vanedb_rs_disk_get(disk, ids[0], vector) == 0);
    CHECK(memcmp(vector, vectors, sizeof(vector)) == 0);
    CHECK(vanedb_rs_disk_search(disk, vectors, 2, found, distances) == 2);
    CHECK(found[0] == ids[0] && fabsf(distances[0] - expected_distance) < 1e-6f);
    vanedb_rs_disk_free(disk);
    CHECK(remove(path) == 0);
    /* Every handle this pass created has been released. */
    CHECK(vanedb_rs_handle_count() == handles_before);
    printf("C ABI acceptance passed for metric %u\n", (unsigned)metric);
}

int main(int argc, char **argv) {
    CHECK(argc == 2);
#ifdef VANEDB_COEXISTENCE
    CHECK(independent_rust_exercise() == 42);
#endif
    /* The library must be the one this header describes. */
    CHECK(vanedb_rs_abi_version() == VANEDB_RS_ABI_VERSION);
    CHECK(strcmp(vanedb_rs_version(), VANEDB_RS_VERSION) == 0);
    exercise(VANEDB_RS_L2, argv[1]);
    exercise(VANEDB_RS_COSINE, argv[1]);
    exercise(VANEDB_RS_DOT, argv[1]);
#ifdef VANEDB_COEXISTENCE
    {
        const float vector[] = {1.0f};
        uint64_t id = 99;
        float distance = 99.0f;
        uint32_t callbacks = 0;
        vanedb_rs_store store = vanedb_rs_store_new(1, VANEDB_RS_L2);
        CHECK(store != VANEDB_RS_NULL_HANDLE);
        CHECK(vanedb_rs_store_add(store, 42, vector) == 0);
        CHECK(vanedb_rs_store_search_filtered(store, vector, 1,
            independent_rust_filter, &callbacks, NULL, 0, NULL, 0, &id, &distance) == 0);
        CHECK(callbacks == 1);
        CHECK(vanedb_rs_last_error() == VANEDB_RS_OK);
        CHECK(id == 99 && distance == 99.0f);
        CHECK(vanedb_rs_store_search(store, vector, 1, &id, &distance) == 1);
        CHECK(id == 42);
        vanedb_rs_store_free(store);
    }
#endif
    CHECK(vanedb_rs_handle_count() == 0);
#ifdef VANEDB_COEXISTENCE
    CHECK(independent_rust_exercise() == 42);
#endif
    return 0;
}
