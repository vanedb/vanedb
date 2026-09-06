# vanedb

Embeddable vector database for edge AI.

Three ways to hold vectors, all searchable by k nearest neighbours:

- `FlatIndex` — exact brute-force scan, held in memory.
- `ApproxIndex` — approximate graph index: sub-linear search, recall traded
  against speed through `ef_search`.
- `DiskIndex` — exact scan over a memory-mapped file, so a corpus larger
  than RAM stays searchable (feature `disk`).

```rust
use vanedb::{Metric, ApproxIndex};

let index = ApproxIndex::builder(768, Metric::Cosine)
    .capacity(100_000)
    .build()?;
index.add(1, &embedding)?;

// Results come back nearest first.
let hits = index.search(&query, 10)?;
```

Distance kernels dispatch to NEON or AVX2 at runtime and fall back to a
portable scalar path.

Both formats are little-endian. `DiskIndex` writes `VNDB` v1, a specified,
versioned format anchored to shared fixtures; either engine reads the other's
file. `ApproxIndex::save` is engine-specific and not yet a stable public
format — treat a saved graph as a way to avoid rebuilding, not as a system of
record, until the `VNDB` graph payload lands. Its loader accepts the current
version and the one before it; `DiskIndex` accepts `VNDB` v1 only, since there
is no earlier version to accept.

A header-only C++ implementation is maintained alongside this crate, with a
cross-engine benchmark harness, in the
[repository](https://github.com/vanedb/vanedb). The two share graph
construction as well as the `DiskIndex` format.

## License

MIT
