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

### Comparing two revisions

```bash
cargo run --release --manifest-path bench/Cargo.toml --bin abtest -- \
  --a origin/main --b HEAD --bench disk
```

`abtest` builds each revision in its own git worktree, runs the selected
benches interleaved A-B-A-B, and reports each operation's median per arm
alongside the spread that arm measured for itself:

```
operation           A bb9ec08     B c42b3d6      delta  A spread  B spread  verdict
disk_build/cpp        5.01 ms       5.04 ms      +0.5%      1.1%      0.6%  noise
disk_build/rs          1.19 s      10.08 ms     -99.2%      0.7%      5.0%  SIGNIFICANT
disk_open/rs        574.34 us     579.80 us      +1.0%      1.5%      0.7%  noise
disk_search/rs       98.32 us      97.07 us      -1.3%      4.0%      0.2%  noise
```

A delta counts as real only when it clears both arms' spread and the 3% floor —
a run that happened to repeat exactly has not proved the machine is quieter
than it is known to be. Above, `disk_search/rs` moved 1.3% against its own 4.0%
spread and is correctly called noise. Add `--rounds`, `--keep`, or criterion
flags after `--`. Idle hardware only.

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
| Store add throughput | `store_add/n=10000` |
| Store search latency | `store_search/n={1000,10000}` |
| HNSW build latency | `index_build` |
| HNSW search latency | `index_search` |
| HNSW recall@k | `report` binary |
| mmap build latency | `disk_build` |
| mmap open latency | `disk_open` |
| mmap search latency | `disk_search` |

`coverage::SCOPE` holds this table as data and a test fails if a bench stops
implementing a row, so a scope claim cannot drift from the code (#63).

## Headline snapshot (Apple M4 Pro, 2026-09, monorepo 80066a2)

Criterion medians of three passes. Inter-pass spread 0.2–10.4%, median 2.2%;
treat smaller differences as noise.

| Op (dim=128, n=10k unless noted) | C++ | Rust | rs/cpp |
|---|---:|---:|---:|
| l2_sq (128d) | 16.7 ns | 16.6 ns | 1.00 |
| l2_sq (768d) | 51.8 ns | 48.4 ns | 0.93 |
| cosine (128d) | 26.8 ns | 27.2 ns | 1.02 |
| cosine (768d) | 96.4 ns | 86.7 ns | 0.90 |
| dot (128d) | 16.1 ns | 16.3 ns | 1.01 |
| dot (768d) | 53.7 ns | 46.4 ns | 0.86 |
| store_add (n=10k) | 745 µs | 1.21 ms | 1.63 |
| store_search (k=10, n=1k) | 8.12 µs | 8.73 µs | 1.08 |
| store_search (k=10, n=10k) | 78.9 µs | 93.1 µs | 1.18 |
| index_build (M=16, efC=200) | 938 ms | 1.01 s | 1.08 |
| index_search (ef=50) | 18.8 µs | 21.3 µs | 1.13 |
| disk_build | 4.98 ms | 12.0 ms | 2.40 |
| disk_open | 507 µs | 576 µs | 1.14 |
| disk_search (k=10) | 78.6 µs | 95.7 µs | 1.22 |

HNSW recall@10 (100 queries, ef=50): C++ 0.689, Rust 0.700.

Rust leads the distance kernels at 768 dimensions and is at parity at 128. It
trails on every scan (1.08–1.22, vanedb#32) and, now that the write paths are
measured, on `store_add` at 1.63 and `disk_build` at 2.40 — different work
from the scan gap, tracked separately.

## Measurement policy

- Both engines run in one process through their C ABIs, built interleaved and
  resident together during search benches.
- The report bin samples the engines interleaved (cpp, rs, cpp, rs…) after a
  joint warmup.
- Criterion is canonical. The report bin covers l2_sq, store_search, and
  index_search + recall only; every other operation is criterion-only.
- Construction and teardown are excluded from timed intervals: `index_build`,
  `store_add`, `disk_build` and `disk_open` time only the operation named.
- `disk_build` writes megabytes and fsyncs. Its spread is far wider than the
  compute benches; never read a single run.
- `disk_open` maps a file already in page cache and validates every value, so
  it is O(n·dim) by design rather than a constant-cost map.
- Every setup return code and handle is asserted, so a failed engine fails the
  run instead of timing a null handle as infinitely fast.
- On x86_64 the harness compiles the C++ capi with `-mavx2 -mfma`. C++ gates
  SIMD at compile time while Rust detects it at runtime, so without those flags
  the harness would compare Rust-AVX2 against C++-scalar.

## License

MIT
