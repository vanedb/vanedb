# vanedb-bench

Rigorous, reproducible head-to-head benchmark of the two VaneDB
implementations in this repository: C++ in [`../cpp`](../cpp) and Rust in
[`../vanedb`](../vanedb).

## Status

**Implemented.** Criterion benches covering every operation the design spec
promises, plus a `report` binary that writes a [`RESULTS.md`](RESULTS.md)
snapshot with ApproxIndex recall@10 averaged over 100 queries. Design spec:
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
| FlatIndex add throughput | `store_add/n=10000` |
| FlatIndex search latency | `store_search/n={1000,10000}` |
| ApproxIndex build latency | `index_build` |
| ApproxIndex search latency | `index_search` |
| ApproxIndex recall@k | `report` binary |
| Disk build latency | `disk_build` |
| Disk open latency | `disk_open` |
| Disk search latency | `disk_search` |

`coverage::SCOPE` holds this table as data and a test fails if a bench stops
implementing a row, so a scope claim cannot drift from the code (#63).

## Headline snapshot (Apple M4 Pro, 2026-09, monorepo af05db0)

Criterion medians of three passes, both engines interleaved in one process.
Inter-pass spread 0.1–8.3%, median 1.5%.

**Read the ratios, not the absolute times.** Absolute figures move with machine
state between sessions — this run measured `l2_sq` at 128d roughly 2.7x faster
than the previous snapshot, on unchanged code. The rs/cpp ratio is measured
within a single interleaved run and is the comparable number.

| Op (dim=128, n=10k unless noted) | C++ | Rust | rs/cpp |
|---|---:|---:|---:|
| l2_sq (128d) | 6.1 ns | 7.2 ns | 1.18 † |
| l2_sq (768d) | 37.3 ns | 37.2 ns | 1.00 |
| cosine (768d) | 71.0 ns | 73.5 ns | 1.04 |
| dot (768d) | 33.8 ns | 33.9 ns | 1.00 |
| store_add (n=10k) | 742 µs | 1.17 ms | 1.58 ‡ |
| store_search (k=10, n=1k) | 8.1 µs | 8.3 µs | 1.02 |
| store_search (k=10, n=10k) | 78.8 µs | 77.2 µs | **0.98** |
| index_build (M=16, efC=200) | 947 ms | 1.03 s | 1.08 |
| index_search (ef=50) | 19.1 µs | 23.1 µs | 1.21 ◊ |
| disk_build | 5.06 ms | 11.05 ms | 2.18 ¶ |
| disk_open | 503 µs | 565 µs | 1.12 ‡ |
| disk_search (k=10) | 78.4 µs | 79.0 µs | 1.01 |

ApproxIndex recall@10 (100 queries, ef=50): C++ 0.689, Rust 0.700.

**† ** At 6–7 ns the difference is around one nanosecond, near this harness's
resolution. Treat 128-dimension kernel ratios as noise.

**‡ ** Rust's internal id maps use the default SipHash hasher where C++ uses
identity, worth about 40 ns per add (vanedb#109).

**◊ Superseded harness; awaiting a re-run.** This figure was timed against a
*single* query, and the two engines build different graphs from the same seed
(`StdRng` versus `std::mt19937`), so per-query work differed by graph luck.
Measured over 16 queries the mean was 0.97 with a 41% spread — Rust ahead on
average, and every ratio ever published for this row sits inside that spread.
The bench now sweeps 32 queries (vanedb#111), so this row, along with the two
other search rows, is replaced by the next snapshot taken on dedicated
hardware. Until then it measures nothing.

**¶ Measured against mismatched durability; awaiting a re-run.** When this
snapshot was taken the engines called different primitives: Rust `sync_all()`
(`F_FULLFSYNC` on macOS, a media barrier) versus C++ `fsync(2)` (write cache
only). Nearly the whole gap was that difference, so C++ was less durable here
rather than faster. Both engines now issue `F_FULLFSYNC` on Darwin
(vanedb#110), so this row is replaced by the next snapshot taken on dedicated
hardware.

Any save-path comparison requires both engines to use the same durability
primitive; otherwise the faster row is only the weaker guarantee.

Rust leads the largest scan after moving both brute-force paths to a bounded
top-k heap (vanedb#32). The remaining honest gaps are on write paths and are
diagnosed rather than mysterious.

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
