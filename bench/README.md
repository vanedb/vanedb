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

## License

MIT
