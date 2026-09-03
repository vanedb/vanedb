# VaneDB Benchmark Results

Engines: vanedb-cpp (CMake Release) and vanedb (Rust), monorepo 47f6195.
Workload: dim=128, n=10000, k=10, L2. Latencies are medians of 501 interleaved paired samples (one query) after a joint warmup; recall is averaged over 100 queries. Both engines' data stays resident in one process (interleaved construction).

Covers l2_sq, store_search, and hnsw_search + recall@10 only; hnsw_build and mmap_search live in the criterion suite (see README).

| Op | C++ (ns) | Rust (ns) | ratio (rs/cpp) |
|---|---:|---:|---:|
| l2_sq | 13 | 17 | 1.26 |
| store_search | 81833 | 99750 | 1.22 |
| hnsw_search | 20542 | 22458 | 1.09 |

HNSW recall@10: C++ 0.689, Rust 0.700

Criterion is canonical; see the README table. This bin times l2_sq in batches
of 1000 calls, which inlines differently from criterion's per-call harness.
