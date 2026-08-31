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

## Headline snapshot (Apple Silicon M-series, 2026-07)

Three rounds of measurement drove three rounds of fixes:

1. **First run** (pre-fix): Rust trailed 1.5× on store/mmap search and 2.5× on
   HNSW build → vanedb#22 (top-k selection), #23 (build parity), #24 (mmap top-k).
2. **Second run**: Rust led every op → exposed both engines' distance kernels as
   latency-bound → vanedb-cpp#31 + vanedb#27 (multi-accumulator SIMD unrolling,
   kernels now identical at ~37 ns/768d).
3. **Final run** (vanedb@89f5144, vanedb-cpp@f02bb27): near-parity, honors split —
   and both engines 30–50% faster than round one on every hot path.

| Op (n=10k, dim=128, L2) | C++ | Rust | rs/cpp |
|---|---:|---:|---:|
| l2_sq (768d) | 37.1 ns | 37.2 ns | 1.00 |
| store_search (k=10) | 82.3 µs | 93.4 µs | 1.14 |
| hnsw_build (M=16, efC=200) | 984 ms | 929 ms | 0.94 |
| hnsw_search (ef=50) | 24.5 µs | 21.4 µs | 0.87 |
| mmap_search (k=10) | 81.8 µs | 95.3 µs | 1.17 |

HNSW recall@10 (100 queries, ef=50): C++ 0.689, Rust 0.700 — quality-comparable.

Re-measured 2026-08 under the unified measurement policy below: ratios
unchanged (store_search 1.14, hnsw_search 0.87).

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
