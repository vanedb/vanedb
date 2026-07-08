# VaneDB Benchmark Results

Workload: dim=128, n=10000, k=10, L2. Latencies are medians (one query); recall is averaged over 100 queries.

| Op | C++ (ns) | Rust (ns) | ratio (rs/cpp) |
|---|---:|---:|---:|
| l2_sq | 28 | 27 | 0.96 |
| store_search | 237000 | 164125 | 0.69 |
| hnsw_search | 31500 | 27500 | 0.87 |

HNSW recall@10: C++ 0.689, Rust 0.700
