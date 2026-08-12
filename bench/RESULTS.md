# VaneDB Benchmark Results

Engines: vanedb-cpp f02bb27 (CMake Release), vanedb (Rust) 89f5144.
Workload: dim=128, n=10000, k=10, L2. Latencies are medians of 501 interleaved paired samples (one query) after a joint warmup; recall is averaged over 100 queries. Both engines' data stays resident in one process (interleaved construction).

Covers l2_sq, store_search, and hnsw_search + recall@10 only; hnsw_build and mmap_search live in the criterion suite (see README).

| Op | C++ (ns) | Rust (ns) | ratio (rs/cpp) |
|---|---:|---:|---:|
| l2_sq | 14 | 17 | 1.17 |
| store_search | 81917 | 88500 | 1.08 |
| hnsw_search | 24959 | 21416 | 0.86 |

HNSW recall@10: C++ 0.689, Rust 0.700
