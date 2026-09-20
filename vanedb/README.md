# vanedb

Embeddable vector database for edge AI. Supply your own embeddings and search
inside your process without running a database server.

- `FlatIndex` — exact search in memory.
- `ApproxIndex` — approximate graph search, with recall controlled by `ef_search`.
- `DiskIndex` — exact search over a read-only memory-mapped file (feature `disk`).

VaneDB stores only `(u64, vector)` pairs. Keep metadata and your id-to-document
mapping alongside it; per-query allow lists, deny lists, or predicates restrict
search results using that external metadata. Supply your own embeddings —
VaneDB does not generate them.

Add it with `cargo add vanedb`. This complete program inserts two vectors and
finds the nearest one:

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
Every index reports its count with `len()` and `is_empty()`; `get` and
`get_vector` return `Ok(None)` for an id that is not stored, while `remove` of
one is an error.

All three indexes accept filters through `search_with`:

```rust
# use vanedb::{FlatIndex, Filter, Metric, SearchParams};
# fn main() -> vanedb::Result<()> {
# let index = FlatIndex::new(3, Metric::Cosine)?;
let allowed = [101, 202]; // strictly ascending, with no duplicates
let options = SearchParams::new().filter(Filter::Allow(&allowed));
let hits = index.search_with(&[1.0, 0.0, 0.0], 10, &options)?;
# Ok(())
# }
```

An empty allow list matches nothing; an empty deny list matches everything.
Predicates should consult external metadata and must not call methods on the
index being searched: the search holds its read lock. Exact indexes scan once.
An approximate search widens its beam when fewer than `k` results match, up to
`max_ef_search` (default four times the initial beam). This cap is a recall/work
tradeoff, not a guarantee of `k` results or a strict bound on distance evaluations.

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
verifying migration. During 0.x, APIs and persistence formats may change in a minor release.
Existing format identifiers will not be reinterpreted; new encodings require
new identifiers and readers for existing files are retained. Older readers
need not accept future formats.
Future insertions need not reproduce identical topology. `DiskIndex` continues
to accept VNDB v1 only.

`DiskIndex::open` is unsafe because its vectors borrow from a memory-mapped
file. Before calling it, ensure no process can rewrite or truncate the underlying
file until the index is dropped; violating this requirement can cause undefined
behavior or a process fault. Replacing the path with `DiskIndexBuilder::save`
is supported: its atomic rename leaves existing readers on the intact old file.
`DiskIndex::get` returns `Option<Cow<[f32]>>`, borrowing from the mapping when
the id is stored; use `as_ref()` to borrow or `into_owned()` to obtain an
independent vector.

## Feature flags

- `disk` — `DiskIndex` and `DiskIndexBuilder`, the exact index over a
  memory-mapped file.
- `gpu-metal` — experimental; builds only on macOS 10.14 or newer and needs a
  usable Metal device. It does **not** accelerate any index: `FlatIndex`,
  `ApproxIndex` and `DiskIndex` build and search on the CPU whether or not it
  is enabled, and no binding exposes it. What it adds is the standalone
  `vanedb::gpu::MetalCompute` API, which uploads a row-major `n * dim` matrix
  of finite floats to a Metal buffer (`upload`) and runs L2, cosine or
  dot-product distance scans against that buffer (`distances`, `search`) with
  the same distance definitions as the indexes. Dimensions must be nonzero and
  divisible by four, inputs stay within the device's buffer and shader limits,
  very small nonzero components are routed to the CPU kernels to preserve
  rankings, and every initialization or execution failure returns `VaneError`
  with no automatic CPU fallback. Its parameters and errors are documented on
  its methods. Whether the feature is finished into index acceleration or
  removed is decided after 0.3.0 (#257); the current state is recorded in the
  [limits page](https://github.com/vanedb/vanedb/blob/main/docs/LIMITS.md).

See the [repository guide](https://github.com/vanedb/vanedb) for bindings,
platform verification scope, persistence details, and source builds. The
supplementary C++ implementation is frozen reference code.

## License

MIT
