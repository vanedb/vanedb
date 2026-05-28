# VaneDB Bench: Cross-Language Benchmark Harness Design Spec

**Date:** 2026-05-28
**Status:** Approved

## 1. Project Identity

- **Name:** vanedb-bench
- **Repo:** `vanedb/vanedb-bench`
- **Purpose:** A rigorous, reproducible head-to-head benchmark of the two VaneDB
  implementations — C++ (`vanedb/vanedb-cpp`) and Rust (`vanedb/vanedb`).
- **License:** MIT (matches both implementations)

## 2. Motivation

The two implementations are maintained side-by-side specifically to compare
efficiency. Today there is **no cross-language comparison** — each repo has only
within-language benchmarks (C++ via Google Benchmark, Rust via criterion), and
those two harnesses are **not directly comparable** (different statistical
methods, warm-up, and optimizer-defeat mechanisms).

Goal priority: **rigor first, presentability second.** The numbers must be
defensible enough to make real "which implementation wins where" decisions; a
clean portfolio-facing summary is the secondary deliverable.

## 3. Scope

**In scope** (full operation coverage):

| Operation | Measured |
|---|---|
| Distance functions (L2, cosine, dot) | latency across several dims |
| VectorStore (brute-force) | add throughput, k-NN search latency |
| HNSW | build time, search latency, **recall@k** |
| MMap store | build/open time, search latency |

**Out of scope (v1):**
- GPU (Metal/CUDA) — availability differs by machine; separate concern.
- Python/WASM bindings — the comparison is C++-core vs Rust-core through a C ABI.
- CI-gated benchmarking — shared runners are too noisy for trustworthy µs numbers;
  local runs are the source of truth.

## 4. Architecture

The **C-ABI contract is the spine**. Each implementation ships a real, shippable
C API; the bench repo consumes both and drives them through one harness.

```
vanedb-cpp/                       vanedb/                            vanedb-bench/  (this repo)
  capi/vanedb_capi.h  ◄── same ──►  vanedb-capi/ (crate)               build.rs → link both static libs
  capi/vanedb_capi.cpp   contract    src/lib.rs (#[no_mangle])          src/ffi_cpp.rs / src/ffi_rs.rs
  → libvanedb_cpp_capi.a            cbindgen → vanedb_rs_capi.h         src/workloads.rs   (seeded data)
  symbols: vanedb_cpp_*             → libvanedb_rs_capi.a               src/ground_truth.rs (recall@k)
                                    symbols: vanedb_rs_*                benches/*.rs (criterion groups)
                                                                        src/bin/report.rs → RESULTS.md
```

- **Driver:** a Rust binary using **criterion** (bootstrap resampling + outlier
  rejection). Both implementations are called through C-ABI symbols, so the only
  difference measured is the implementation itself.
- **Symbol namespacing:** C++ exports `vanedb_cpp_*`, Rust exports `vanedb_rs_*`,
  so both link into one criterion binary without collision. Each criterion group
  benches the two families head-to-head on identical input data.

### Decomposition into implementation plans

This design is realized as **three separate implementation plans** (different
repos = separate plan executions), tied together by the shared C-ABI contract:

1. **`vanedb-cpp` C API** — `capi/vanedb_capi.{h,cpp}`, build target
   `libvanedb_cpp_capi.a`, a C smoke test.
2. **`vanedb` `vanedb-capi` crate** — `#[no_mangle] extern "C"` wrappers, cbindgen
   header generation, `staticlib` output, a smoke test.
3. **`vanedb-bench`** — FFI bindings to both, `workloads`, `ground_truth`/recall,
   criterion groups per operation, the `RESULTS.md` generator.

## 5. The C-ABI Contract

`PFX` = `vanedb_cpp` or `vanedb_rs`. Both implementations expose this exact shape.

```c
typedef enum { VANEDB_L2 = 0, VANEDB_COSINE = 1, VANEDB_DOT = 2 } vanedb_metric;

// --- Distance (stateless) ---
float  PFX_l2_sq(const float* a, const float* b, size_t dim);
float  PFX_cosine_distance(const float* a, const float* b, size_t dim);
float  PFX_dot_product(const float* a, const float* b, size_t dim);

// --- VectorStore (brute force) ---
typedef struct PFX_store PFX_store;
PFX_store* PFX_store_new(size_t dim, vanedb_metric m);
int        PFX_store_add(PFX_store*, uint64_t id, const float* v);
size_t     PFX_store_search(PFX_store*, const float* q, size_t k,
                            uint64_t* out_ids, float* out_dists);
void       PFX_store_free(PFX_store*);

// --- HNSW (seed makes each impl reproducible) ---
typedef struct PFX_hnsw PFX_hnsw;
PFX_hnsw*  PFX_hnsw_new(size_t dim, vanedb_metric m, size_t cap,
                        size_t M, size_t ef_construction, uint64_t seed);
int        PFX_hnsw_add(PFX_hnsw*, uint64_t id, const float* v);
size_t     PFX_hnsw_search(PFX_hnsw*, const float* q, size_t k, size_t ef_search,
                           uint64_t* out_ids, float* out_dists);
int        PFX_hnsw_save(PFX_hnsw*, const char* path);
PFX_hnsw*  PFX_hnsw_load(const char* path);
void       PFX_hnsw_free(PFX_hnsw*);

// --- MMap store ---
typedef struct PFX_mmap PFX_mmap;
int        PFX_mmap_build(const char* path, size_t dim, vanedb_metric m,
                          const uint64_t* ids, const float* vecs, size_t n);
PFX_mmap*  PFX_mmap_open(const char* path);
size_t     PFX_mmap_search(PFX_mmap*, const float* q, size_t k,
                           uint64_t* out_ids, float* out_dists);
void       PFX_mmap_free(PFX_mmap*);
```

**Contract conventions:**
- Opaque handle pointers; every constructor has a matching `_free`.
- Caller owns the `out_ids` / `out_dists` buffers (length `k`); `_search` returns
  the count actually written.
- Integer returns: `0` = success, non-zero = error.
- Both implementations already support seeded HNSW construction
  (`vanedb-cpp` `hnsw_index.h` ctor `uint32_t seed`; `vanedb` `StdRng`/`SeedableRng`),
  so no new seeding work is required in the implementations.

## 6. Fairness Controls

1. **Identical inputs** — `workloads` generates seeded-random vectors once; both
   implementations receive the same bytes.
2. **Identical call path** — both behind the same C-ABI signature, so call
   overhead (one non-inlined cross-library call) is equal on both sides.
3. **Same harness** — criterion's bootstrap statistics and outlier rejection
   applied identically; `black_box` defeats the optimizer on both sides.
4. **HNSW measured as a triple** — same `(M, ef_construction, ef_search, seed)`;
   report **build time, search time, AND recall@k** against brute-force ground
   truth. A faster index that recalls worse is not "winning."
5. **Seeding ≠ graph identity** — C++ uses `std::mt19937`, Rust uses ChaCha-based
   `StdRng`; the same seed value produces *different* random sequences, hence
   *different* graphs. Seeding guarantees per-implementation reproducibility, not
   cross-implementation structural identity. recall@k is the equalizer that
   confirms the two indexes are of comparable quality.
6. **Pinned versions** — exact commit of each implementation is recorded (git
   submodule SHA for C++, cargo `rev` for Rust).

## 7. Output & Reporting

- **criterion HTML** — per-operation detail with distributions (the rigor artifact).
- **`RESULTS.md` generator** (`src/bin/report.rs`) — a summary table:
  `op | dim/size | C++ | Rust | ratio`, plus `recall_cpp | recall_rs` columns for
  HNSW (the portfolio artifact).

## 8. Version Pinning

- **C++:** git submodule of `vanedb-cpp` pinned to a specific SHA.
- **Rust:** `vanedb = { git = "https://github.com/vanedb/vanedb", rev = "<sha>" }`.
- The README documents both pinned SHAs. Bumping them is the explicit
  "compare newer versions" action.

## 9. Risks & Open Questions

- **MMap on Windows** — file-locking semantics differ; the mmap benchmark may need
  to be Unix-only or guarded on Windows. To confirm during plan 3.
- **C++ build integration** — linking `libvanedb_cpp_capi.a` from `build.rs` via
  the `cc` crate; needs the C++ toolchain present. Acceptable (benchmarking is a
  dev-machine activity), but documented as a prerequisite.
- **Distance-function timing floor** — at ~100ns, even criterion needs enough
  iterations to separate signal from timer noise; batch sizing handled by
  criterion's adaptive sampling, validated in plan 3.
