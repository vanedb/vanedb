# vanedb

Embeddable vector database for edge AI. Supply your own embeddings and search
inside your process without running a database server.

- `FlatIndex` — exact search in memory.
- `ApproxIndex` — approximate graph search, with recall controlled by `ef_search`.
- `DiskIndex` — exact search over a read-only memory-mapped file (feature `disk`).

From a source checkout, run `cargo run -p vanedb --example quickstart --locked`
at the repository root. For a local application, add
`vanedb = { path = "/path/to/vanedb/vanedb" }` to its Cargo dependencies.
This complete program inserts two vectors and finds the nearest one:

```rust
use vanedb::{ApproxIndex, Metric};

fn main() -> Result<(), vanedb::VaneError> {
    let index = ApproxIndex::builder(3, Metric::Cosine)
        .capacity(100)
        .build()?;
    index.add_batch(&[101, 202], &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0])?;
    let hits = index.search(&[1.0, 0.0, 0.0], 1)?;
    assert_eq!(hits[0].id, 101);
    println!("Nearest id: {}, distance: {}", hits[0].id, hits[0].distance);
    Ok(())
}
```

Metrics are squared Euclidean distance (`Metric::L2`), cosine distance
(`Metric::Cosine`), and negative dot product (`Metric::Dot`). Lower distances
rank first. IDs are unique unsigned 64-bit integers; vectors and queries must
have the configured dimension and finite components. Approximate search can
miss a true neighbor; compare recall against exact search for your data.

Distance kernels select NEON or AVX2 at runtime and fall back to scalar code.
Capacity is a reserve hint for the growable graph index, not an insertion limit.
Removed and replaced entries occupy storage until compaction rebuilds the graph.

`DiskIndex` files use the shared VNDB v1 format; Rust and the supplementary C++
engine can read each other's disk files. Disk index construction buffers vectors
in memory before saving. `ApproxIndex` writes shared VNDB v2 graph files,
preserving vectors, graph links, IDs and deleted slots across both engines.
Further insertions can produce different graphs across engines. The Rust loader
also reads legacy Rust v1/v2 graphs; load and save to a new path to migrate.
Older readers cannot open VNDB v2. Keep originals and source vectors while
verifying migration: this is a release candidate, without a public 1.0.0 format
compatibility promise yet. `DiskIndex` continues to accept VNDB v1 only.

See the [repository guide](https://github.com/vanedb/vanedb) for bindings,
platform verification scope, persistence details, and source builds. The
supplementary C++ implementation is frozen reference code.

## License

MIT
