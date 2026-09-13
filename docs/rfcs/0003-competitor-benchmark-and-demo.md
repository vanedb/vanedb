# RFC 0003: Competitor benchmark and demo

- Status: accepted (2026-09-13)
- Milestone: 0.2.0
- Tracking issue: #198
- Supersedes / superseded by: none

## Problem

The only published comparison is `bench/`, which measures the Rust engine
against the repository's own frozen C++ engine. No buyer evaluates against
that. Three days after 0.1.1 there are no external users, no stars and no
mentions ([`MARKET_ANALYSIS.md`](../MARKET_ANALYSIS.md), section 2), so every
roadmap decision is being made without a user signal. A benchmark against the
libraries a developer actually shortlists, published with a working demo, is
the cheapest way to produce one.

## Decision

Add a second benchmark arm under `bench/` that measures VaneDB against the
in-process libraries it competes with, on real embedding vectors, on dedicated
hardware, with the existing interleaved methodology. Publish the results in
`bench/COMPARISON.md` with the same caveats discipline as the existing
snapshot. Pair it with a runnable demo (`obsidian-vane-search`, already a
downstream application) and a launch post.

## Design

### Engines

| Engine | Access path | Why |
|---|---|---|
| USearch | `usearch` crate (C++ core) | the mobile-capable HNSW incumbent |
| hnswlib | via `hnswlib` C++ through a thin FFI in the harness, or the `hnsw_rs` port | the reference HNSW |
| `instant-distance` | crate | most-downloaded pure-Rust HNSW |
| `hnsw_rs` | crate | second pure-Rust HNSW |
| `sqlite-vec` | `rusqlite` with the loadable extension | the "you already ship SQLite" baseline, brute force |
| VaneDB | `vanedb` crate | |

EdgeVec, voy and `client-vector-search` are browser libraries; they are
compared separately in the WebAssembly arm (RFC 0006) once persistence exists.

### Workload

- Real embeddings, not uniform random: `nomic-embed-text` (768-d) or
  EmbeddingGemma over a public text corpus, 100k documents, 1k held-out
  queries. Vectors generated once, stored as a fixture with a checksum, never
  regenerated in CI.
- Metrics: cosine (the embedding models' native metric) and squared L2.
- Measured: build time; peak resident memory; index file size; search latency
  per query at `k = 10` for a sweep of beam widths; recall@10 against exact
  search computed in `f64`; delete-then-search correctness where the engine
  supports delete.
- Every engine built with the same `M` and construction beam where the
  parameter exists; otherwise the engine default, recorded.
- Interleaved A-B-A-B runs, medians and inter-pass spread reported, per the
  rules in `AGENTS.md`. No CI numbers.

### Hardware

- One Apple Silicon laptop (dedicated, plugged in, other work stopped).
- One Linux x86-64 box with AVX2.
- One Android ARM64 device or, failing that, the ARM64 emulator with that fact
  stated in the results.

### Output

- `bench/COMPARISON.md`: methodology, hardware, exact versions of each engine,
  the tables, and a caveats section written before the numbers.
- The harness is reproducible from a clean checkout with one command and
  pinned versions.
- The demo: `obsidian-vane-search` updated to 0.2.0, with a README that shows
  one search on a real vault.

## Alternatives rejected

- **ann-benchmarks.** Its datasets are 100-d to 960-d synthetic or old
  embeddings and its harness is Python-only; it would not measure the Rust
  or C paths a VaneDB user runs.
- **Add the competitors to the existing C++-vs-Rust harness.** Rejected: that
  harness's purpose is a conformance-adjacent comparison of two engines that
  share a file format; mixing in third parties confuses both.

## Compatibility and migration

None. No engine change. The C++ engine is not part of this arm.

## Acceptance criteria

- [ ] Harness under `bench/compare/` builds with `--locked` and runs each
      engine on the fixture with one command.
- [ ] Embedding fixture generated, checksummed, and documented with the model
      and corpus version.
- [ ] `bench/COMPARISON.md` published with results from the three hardware
      classes above, each run recorded (date, commit, engine versions).
- [ ] Caveats section reviewed against `AGENTS.md`'s performance rules before
      publication.
- [ ] `obsidian-vane-search` demo updated and linked from the README.
- [ ] Launch post drafted (Show HN, r/rust, r/LocalLLaMA) and published after
      the maintainer's review.

## Evidence required before the claim

Dedicated hardware, interleaved runs, recorded engine versions. A number that
appeared once is not a result.

## Out of scope

Browser comparison (RFC 0006). Any engine change to win a row; regressions the
benchmark reveals become their own issues.
