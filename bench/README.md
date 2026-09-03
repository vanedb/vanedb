# vanedb-bench

Rigorous, reproducible head-to-head benchmark of the two VaneDB
implementations in this repository: C++ in [`../cpp`](../cpp) and Rust in
[`../vanedb`](../vanedb).

## Status

**Implemented.** Criterion benches covering every operation the design spec
promises, plus a `report` binary that writes a [`RESULTS.md`](RESULTS.md)
snapshot with HNSW recall@10 averaged over 100 queries. Design spec:
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

## Coverage

| Spec operation | Measured by |
|---|---|
| L2 distance latency | `l2_sq/dim={128,768}` |
| Cosine distance latency | `cosine/dim={128,768}` |
| Dot distance latency | `dot/dim={128,768}` |
| VectorStore add throughput | `store_add/n=10000` |
| VectorStore search latency | `store_search/n={1000,10000}` |
| HNSW build latency | `hnsw_build` |
| HNSW search latency | `hnsw_search` |
| HNSW recall@k | `report` binary |
| mmap build latency | `mmap_build` |
| mmap open latency | `mmap_open` |
| mmap search latency | `mmap_search` |

`coverage::SCOPE` holds this table as data and a test fails if a bench stops
implementing a row, so a scope claim cannot drift from the code (#63).

## Headline snapshot (Apple M4 Pro, 2026-09, monorepo 47f6195)

Criterion medians of three passes on an idle machine. Inter-pass spread
0.3–6.6%; treat smaller differences as noise. Covers the operations measured
at the time; the groups added for full spec coverage join at the next
on-hardware refresh.

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
  hnsw_search + recall only; every other operation is criterion-only.
- Construction and teardown are excluded from timed intervals: `hnsw_build`,
  `store_add`, `mmap_build` and `mmap_open` time only the operation named.
- `mmap_build` writes megabytes and fsyncs. Its spread is far wider than the
  compute benches; never read a single run.
- `mmap_open` maps a file already in page cache and validates every value, so
  it is O(n·dim) by design rather than a constant-cost map.
- Every setup return code and handle is asserted, so a failed engine fails the
  run instead of timing a null handle as infinitely fast.
- On x86_64 the harness compiles the C++ capi with `-mavx2 -mfma`. C++ gates
  SIMD at compile time while Rust detects it at runtime, so without those flags
  the harness would compare Rust-AVX2 against C++-scalar.

## License

MIT
