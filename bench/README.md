# vanedb-bench

Rigorous, reproducible head-to-head benchmark of the two VaneDB
implementations in this repository: C++ in [`../cpp`](../cpp) and Rust in
[`../vanedb`](../vanedb).

## Status

**Implemented.** Criterion benches for distance, VectorStore, HNSW, and mmap,
plus a `report` binary that writes a [`RESULTS.md`](RESULTS.md) snapshot with
HNSW recall@10 averaged over 100 queries. Design spec:
[`docs/superpowers/specs/2026-05-28-vanedb-bench-design.md`](docs/superpowers/specs/2026-05-28-vanedb-bench-design.md).

## Running

```bash
cargo bench --manifest-path bench/Cargo.toml
cargo run --release --manifest-path bench/Cargo.toml --bin report
```

Run these commands from the repository root. They require a C++20 toolchain
and CMake. `build.rs` compiles the local `cpp/` C API, while Cargo links the
local `vanedb-capi/` crate, so one commit identifies both engines.

## Headline snapshot (Apple M4 Pro, 2026-09, monorepo 47f6195)

Criterion medians of three passes on an otherwise idle machine. Inter-pass
spread was 0.3–6.6%, and under 3% for every row except `l2_sq` at 768d — treat
differences smaller than that as noise.

| Op (n=10k, dim=128, L2) | C++ | Rust | rs/cpp |
|---|---:|---:|---:|
| l2_sq (128d) | 16.9 ns | 16.5 ns | 0.98 |
| l2_sq (768d) | 51.3 ns | 48.4 ns | 0.94 |
| hnsw_build (M=16, efC=200) | 938 ms | 1013 ms | 1.08 |
| hnsw_search (ef=50) | 18.8 µs | 21.3 µs | 1.13 |
| mmap_search (k=10) | 78.4 µs | 95.5 µs | 1.22 |
| store_search (k=10, n=1k) | 8.10 µs | 8.67 µs | 1.07 |
| store_search (k=10, n=10k) | 79.0 µs | 92.5 µs | 1.17 |

HNSW recall@10 (100 queries, ef=50): C++ 0.689, Rust 0.700 — quality-comparable.

Two corrections make this table not comparable to the 2026-07 snapshot it
replaces:

1. **The C++ engine was built without `-DNDEBUG`.** A custom
   `CMAKE_CXX_FLAGS_RELEASE` replaced CMake's defaults instead of extending
   them, leaving assertions live in the distance kernels. Restoring the
   standard flags moved hnsw_search by 12.6% and the scan paths by 4–5% — the
   isolated kernel benchmark was unaffected, so the cost was inlining rather
   than the branch itself.
2. **`hnsw_build` used to include teardown.** `hnsw_free` sat inside the
   measured closure, so every build figure carried the cost of destroying a
   10k-node graph. It is now excluded, which is why the ratio moved from 0.94
   to 1.08: teardown was the more expensive half for C++, and removing it from
   both sides changed which engine leads.

Rust leads on the raw distance kernels and trails on everything that scans:
17% at n=10k for `store_search` and 22% for `mmap_search` (see vanedb#32).
`hnsw_search` sits at 1.13 despite the identical kernels, which points at the
same selection overhead rather than at distance computation.

## Measurement policy

- Both engines run in one process through their C ABIs. Stores and indexes
  are built interleaved and stay resident together during search benches —
  the same policy in every criterion suite and the report bin, since
  residency and construction order are part of what's being compared.
- The report bin samples the two engines interleaved (cpp, rs, cpp, rs…)
  after a joint warmup. Block-ordered one-shot timing measured C++ first on
  colder machine state and flipped the store_search verdict relative to
  criterion; paired sampling removed the contradiction (2026-08).
- Criterion is the canonical source for perf claims. The report bin is a
  quick digestible snapshot and covers only l2_sq, store_search, and
  hnsw_search + recall; hnsw_build and mmap_search are criterion-only.
- Every setup return code and handle is asserted, so a failed engine fails
  the run instead of benchmarking a null handle as infinitely fast.
- On x86_64 the harness compiles the C++ capi with `-mavx2 -mfma`:
  vanedb-cpp's own CMake gives those flags to its perf targets but not to
  `vanedb_cpp_capi`, and C++ gates SIMD at compile time while Rust detects
  it at runtime — without the flags the harness would compare Rust-AVX2
  against C++-scalar. The published numbers above are Apple Silicon, where
  NEON is unconditional on both sides.

## License

MIT
