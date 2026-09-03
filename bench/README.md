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

`report` writes [`RESULTS.md`](RESULTS.md) beside this README whatever the
working directory. `VANEDB_BENCH_DIM`, `_N`, `_K`, `_QUERIES` and `_OUT`
override the workload and destination; CI runs it at n=500, dim=32 as an
end-to-end smoke check and asserts recall, never a timing.

## Headline snapshot (Apple M4 Pro, 2026-09, monorepo 47f6195)

Criterion medians of three passes on an idle machine. Inter-pass spread
0.3–6.6%; treat smaller differences as noise.

| Op (n=10k, dim=128, L2) | C++ | Rust | rs/cpp |
|---|---:|---:|---:|
| l2_sq (128d) | 16.9 ns | 16.5 ns | 0.98 |
| l2_sq (768d) | 51.3 ns | 48.4 ns | 0.94 |
| hnsw_build (M=16, efC=200) | 938 ms | 1013 ms | 1.08 |
| hnsw_search (ef=50) | 18.8 µs | 21.3 µs | 1.13 |
| mmap_search (k=10) | 78.4 µs | 95.5 µs | 1.22 |
| store_search (k=10, n=1k) | 8.10 µs | 8.67 µs | 1.07 |
| store_search (k=10, n=10k) | 79.0 µs | 92.5 µs | 1.17 |

HNSW recall@10 (100 queries, ef=50): C++ 0.689, Rust 0.700.

Rust leads the distance kernels and trails every path that scans.
`hnsw_search` at 1.13 with identical kernels points at selection overhead
rather than distance computation (vanedb#32).

## Measurement policy

- Both engines run in one process through their C ABIs, built interleaved and
  resident together during search benches.
- The report bin samples the engines interleaved (cpp, rs, cpp, rs…) after a
  joint warmup.
- Criterion is canonical. The report bin covers l2_sq, store_search, and
  hnsw_search + recall only; hnsw_build and mmap_search are criterion-only.
- Every setup return code and handle is asserted, so a failed engine fails the
  run instead of timing a null handle as infinitely fast.
- On x86_64 the harness compiles the C++ capi with `-mavx2 -mfma`. C++ gates
  SIMD at compile time while Rust detects it at runtime, so without those flags
  the harness would compare Rust-AVX2 against C++-scalar.

## License

MIT
