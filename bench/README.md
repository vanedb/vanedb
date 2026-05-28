# vanedb-bench

Rigorous, reproducible head-to-head benchmark of the two [VaneDB](https://github.com/vanedb)
implementations — C++ ([vanedb-cpp](https://github.com/vanedb/vanedb-cpp)) and
Rust ([vanedb](https://github.com/vanedb/vanedb)).

## Status

**Design stage.** No implementation yet — the design spec is at
[`docs/superpowers/specs/2026-05-28-vanedb-bench-design.md`](docs/superpowers/specs/2026-05-28-vanedb-bench-design.md).

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
