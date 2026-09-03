# VaneDB Benchmark Results

Engines: vanedb-cpp (CMake Release) and vanedb (Rust), monorepo 80066a2.
Workload: dim=128, n=10000, k=10, L2. Latencies are medians of 501 interleaved paired samples (one query) after a joint warmup; recall is averaged over 100 queries. Both engines' data stays resident in one process (interleaved construction).

Covers l2_sq, store_search, and hnsw_search + recall@10 only; every other operation is criterion-only (see README).

Criterion is canonical; see the README table. This bin times l2_sq in batches of 1000 calls, which inlines differently from criterion's per-call harness.

| Op | C++ (ns) | Rust (ns) | ratio (rs/cpp) |
|---|---:|---:|---:|
| l2_sq | 14 | 17 | 1.15 |
| store_search | 79500 | 88250 | 1.11 |
| hnsw_search | 20042 | 21834 | 1.09 |

HNSW recall@10: C++ 0.689, Rust 0.700
