/* The same consumer contract runs on desktop, iOS, and Android. */
#include "vanedb_rs_capi.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

#define CHECK(expr) do { if (!(expr)) { \
    fprintf(stderr, "line %d: %s failed\n", __LINE__, #expr); exit(1); \
} } while (0)

static void exercise(uint32_t metric, const char *directory) {
    const uint64_t ids[] = {UINT64_MAX, (UINT64_C(1) << 53) + 1};
    const uint64_t conflicting_ids[] = {5, UINT64_MAX};
    const float vectors[] = {1, 0, 0, 0, 1, 0};
    const float replacement[] = {0, 0, 1};
    const float expected_distance = metric == VANEDB_RS_DOT ? -1 : 0;
    uint64_t found[2];
    float distances[2], vector[3];
    char path[4096];

    vanedb_rs_store *flat = vanedb_rs_store_new(3, metric);
    CHECK(flat != NULL);
    CHECK(vanedb_rs_store_add_batch(flat, ids, vectors, 2) == 0);
    CHECK(vanedb_rs_store_add_batch(flat, conflicting_ids, vectors, 2) != 0);
    CHECK(!vanedb_rs_store_contains(flat, 5));
    CHECK(vanedb_rs_store_len(flat) == 2);
    CHECK(vanedb_rs_store_dimension(flat) == 3);
    CHECK(vanedb_rs_store_search(flat, vectors, 2, found, distances) == 2);
    CHECK(found[0] == ids[0] && fabsf(distances[0] - expected_distance) < 1e-6f);
    CHECK(vanedb_rs_store_get(flat, ids[0], vector) == 0);
    CHECK(memcmp(vector, vectors, sizeof(vector)) == 0);
    CHECK(vanedb_rs_store_remove(flat, ids[0]) == 0);
    CHECK(!vanedb_rs_store_contains(flat, ids[0]));
    CHECK(vanedb_rs_store_len(flat) == 1);
    vanedb_rs_store_free(flat);

    vanedb_rs_index *graph = vanedb_rs_index_new(3, metric, 2, 4, 16, 42);
    CHECK(graph != NULL);
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
    CHECK(graph != NULL && vanedb_rs_index_len(graph) == 1);
    CHECK(vanedb_rs_index_get_vector(graph, ids[0], vector) == 0);
    CHECK(memcmp(vector, replacement, sizeof(vector)) == 0);
    CHECK(vanedb_rs_index_search(graph, replacement, 2, 16, found, distances) == 1);
    CHECK(found[0] == ids[0] && fabsf(distances[0] - expected_distance) < 1e-6f);
    vanedb_rs_index_free(graph);
    CHECK(remove(path) == 0);

    written = snprintf(path, sizeof(path), "%s/disk-%u.bin", directory, (unsigned)metric);
    CHECK(written > 0 && (size_t)written < sizeof(path));
    CHECK(vanedb_rs_disk_build(path, 3, metric, ids, vectors, 2) == 0);
    vanedb_rs_disk *disk = vanedb_rs_disk_open(path);
    CHECK(disk != NULL);
    CHECK(vanedb_rs_disk_len(disk) == 2 && vanedb_rs_disk_dimension(disk) == 3);
    CHECK(vanedb_rs_disk_contains(disk, ids[0]));
    CHECK(vanedb_rs_disk_get(disk, ids[0], vector) == 0);
    CHECK(memcmp(vector, vectors, sizeof(vector)) == 0);
    CHECK(vanedb_rs_disk_search(disk, vectors, 2, found, distances) == 2);
    CHECK(found[0] == ids[0] && fabsf(distances[0] - expected_distance) < 1e-6f);
    vanedb_rs_disk_free(disk);
    CHECK(remove(path) == 0);
    printf("C ABI acceptance passed for metric %u\n", (unsigned)metric);
}

int main(int argc, char **argv) {
    CHECK(argc == 2);
    exercise(VANEDB_RS_L2, argv[1]);
    exercise(VANEDB_RS_COSINE, argv[1]);
    exercise(VANEDB_RS_DOT, argv[1]);
    return 0;
}
