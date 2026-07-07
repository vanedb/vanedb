# VaneDB Benchmark Results

Workload: dim=128, n=10000, k=10, L2. Latencies are medians (one query); recall is averaged over 100 queries.

| Op | C++ (ns) | Rust (ns) | ratio (rs/cpp) |
|---|---:|---:|---:|
| l2_sq | 28 | 22 | 0.79 |
| store_search | 236500 | 288375 | 1.22 |
| hnsw_search | 28375 | 28667 | 1.01 |

HNSW recall@10: C++ 0.689, Rust 0.746
