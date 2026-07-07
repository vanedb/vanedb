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

| Op (n=10k, dim=128, L2) | C++ | Rust | rs/cpp |
|---|---:|---:|---:|
| l2_sq (768d, criterion) | 101.9 ns | 81.8 ns | 0.80 |
| store_search (k=10) | 120.7 µs | 183.9 µs | 1.52 |
| hnsw_build (M=16, efC=200) | 1.12 s | 2.77 s | 2.47 |
| hnsw_search (ef=50) | 29.8 µs | 29.1 µs | 0.98 |
| mmap_search (k=10) | 121.2 µs | 192.1 µs | 1.59 |

HNSW recall@10 (100 queries, ef=50): C++ 0.689, Rust 0.746 — quality-comparable.

## Approach (summary)

Both implementations expose the same C ABI; a single Rust + criterion driver calls
both through it, so the only difference measured is the implementation itself.
Fairness controls include identical seeded inputs, identical call paths, and — for
HNSW — recall@k against brute-force ground truth alongside latency (a faster index
that recalls worse is not "winning").

See the spec for the full C-ABI contract, fairness controls, and the three-repo
implementation decomposition.

## License

MIT
