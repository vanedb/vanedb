# VaneDB Benchmark Results

Workload: dim=128, n=10000, k=10, L2. Latencies are medians (one query); recall is averaged over 100 queries.

| Op | C++ (ns) | Rust (ns) | ratio (rs/cpp) |
|---|---:|---:|---:|
| l2_sq | 26 | 24 | 0.92 |
| store_search | 231000 | 200959 | 0.87 |
| hnsw_search | 25791 | 21583 | 0.84 |

HNSW recall@10: C++ 0.689, Rust 0.700
