#include "vanedb_rs_capi.h"
#include <stdio.h>

int main(void) {
    const float vectors[] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
    const uint64_t ids[] = {101, 202};
    uint64_t nearest = 0;
    float distance = 0.0f;
    vanedb_rs_store *store = vanedb_rs_store_new(3, VANEDB_RS_COSINE);
    if (!store) return 1;
    int status = 1;
    if (vanedb_rs_store_add_batch(store, ids, vectors, 2) == 0 &&
        vanedb_rs_store_search(store, vectors, 1, &nearest, &distance) == 1 &&
        nearest == 101 && distance == 0.0f) {
        printf("Nearest id: %llu, distance: %.1f\n",
               (unsigned long long)nearest, (double)distance);
        status = 0;
    }
    vanedb_rs_store_free(store);
    return status;
}
