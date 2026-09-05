# vanedb

Embeddable vector database for edge AI.

Three ways to hold vectors, all searchable by k nearest neighbours:

- `Store` — exact brute-force scan, held in memory.
- `Index` — approximate graph index: sub-linear search, recall traded
  against speed through `ef_search`.
- `DiskStore` — exact scan over a memory-mapped file, so a corpus larger
  than RAM stays searchable (feature `disk`).

```rust
use vanedb::{Metric, Index};

let index = Index::builder(768, Metric::Cosine)
    .capacity(100_000)
    .build()?;
index.add(1, &embedding)?;

let hits = index.search(&query, 10)?;  // nearest first
```

Distance kernels dispatch to NEON or AVX2 at runtime and fall back to a
portable scalar path. Index and mmap files are little-endian and stable across
released versions.

A header-only C++ implementation sharing these file formats and graph
construction is maintained alongside this crate, along with a cross-engine
benchmark harness, in the [repository](https://github.com/vanedb/vanedb).

## License

MIT
