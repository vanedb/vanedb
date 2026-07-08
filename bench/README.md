# vanedb-bench

Rigorous, reproducible head-to-head benchmark of the two [VaneDB](https://github.com/vanedb)
implementations — C++ ([vanedb-cpp](https://github.com/vanedb/vanedb-cpp)) and
Rust ([vanedb](https://github.com/vanedb/vanedb)).

## Status

**Implemented.** Criterion benches for distance, VectorStore, HNSW, and mmap,
plus a `report` binary that writes a [`RESULTS.md`](RESULTS.md) snapshot with
HNSW recall@10 averaged over 100 queries. Design spec:
[`docs/superpowers/specs/2026-05-28-vanedb-bench-design.md`](docs/superpowers/specs/2026-05-28-vanedb-bench-design.md).

## Running

```bash
cargo bench                        # criterion suites: distance, store, hnsw, mmap
cargo run --release --bin report   # digestible RESULTS.md snapshot + recall@k
```

Requires a C++20 toolchain and CMake (the `vendor/vanedb-cpp` submodule is
built by `build.rs`). Clone with `--recurse-submodules`.

## Headline snapshot (Apple Silicon M-series, 2026-07)

The harness's first run (2026-07-07, pre-fix main) found Rust trailing C++ 1.5× on
store/mmap search and 2.5× on HNSW build. Those findings drove vanedb#22 (top-k
selection), vanedb#23 (build parity), and vanedb#24 (mmap top-k). Against
post-fix main (vanedb@4a1a089), Rust leads on every op:

| Op (n=10k, dim=128, L2) | C++ | Rust | rs/cpp |
|---|---:|---:|---:|
| l2_sq (768d) | 100.2 ns | 78.7 ns | 0.79 |
| store_search (k=10) | 119.8 µs | 110.6 µs | 0.92 |
| hnsw_build (M=16, efC=200) | 1.12 s | 1.00 s | 0.90 |
| hnsw_search (ef=50) | 28.9 µs | 25.5 µs | 0.88 |
| mmap_search (k=10) | 120.1 µs | 115.8 µs | 0.96 |

HNSW recall@10 (100 queries, ef=50): C++ 0.689, Rust 0.700 — quality-comparable.

## License

MIT
