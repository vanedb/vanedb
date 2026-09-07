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
verifying migration. The Rust engine in VaneDB 1.x will continue to read valid VNDB v1 disk and
VNDB v2 graph files written by 1.0.0, within documented resource limits.
Future insertions need not reproduce identical topology. `DiskIndex` continues
to accept VNDB v1 only.

`DiskIndex::open` is unsafe because its vectors borrow from a memory-mapped
file. Before calling it, ensure no process can rewrite or truncate the underlying
file until the index is dropped; violating this requirement can cause undefined
behavior or a process fault. Replacing the path with `DiskIndexBuilder::save`
is supported: its atomic rename leaves existing readers on the intact old file.
`DiskIndex::get` returns `Cow<[f32]>`; use `as_ref()` to borrow or `into_owned()`
to obtain an independent vector.

## Optional Metal compute on macOS

Metal compute requires macOS 10.14 or newer and a usable Metal device. Enable
`gpu-metal` explicitly in your application's dependency:

```toml
vanedb = { path = "/path/to/vanedb/vanedb", features = ["gpu-metal"] }
```

This exposes `MetalCompute` for manually uploaded vectors and distance scans.
Enabling the feature does not move `FlatIndex`, `ApproxIndex` or `DiskIndex`
operations to the GPU. Call the Metal API directly:

```rust
use vanedb::gpu::{GpuMetric, MetalCompute};

fn main() -> vanedb::Result<()> {
    let gpu = MetalCompute::new()?;
    let ids = [101, 202];
    let vectors = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let buffer = gpu.upload(&vectors, ids.len(), 4)?;
    let hits = gpu.search(&[1.0, 0.0, 0.0, 0.0], &ids, &buffer, 1, GpuMetric::Cosine)?;
    assert_eq!(hits[0].id, 101);
    Ok(())
}
```

Upload row-major `n * dim` finite floats. Dimension must be nonzero and divisible
by four because the kernels use `float4`; ordinary Rust slices need no special
caller-provided pointer alignment. Queries must have that dimension and finite
values, IDs must match the uploaded row count, and `k` must be positive. Sizes
must fit the Metal device's buffer and shader-addressing limits. An empty upload
at a valid dimension returns empty results for a valid query.

Reuse the uploaded buffer for subsequent queries on the same Metal device.
`distances` returns one value per uploaded row; `search` returns up to `k`
nearest results. `GpuMetric::L2`, `Cosine` and `Dot` use the same distance
definitions described above.

When uploaded vectors or a query contain very small nonzero components,
`distances` and `search` automatically use CPU distance kernels to preserve
contributions that Metal may round to zero. This numerical fallback still
requires successful Metal initialization and upload.

Initialization, validation and GPU execution errors return `VaneError`; the
example propagates them. Backend failures do not trigger automatic fallback.
Your application decides whether to report an error or use a CPU index with
the original vectors.

See the [repository guide](https://github.com/vanedb/vanedb) for bindings,
platform verification scope, persistence details, and source builds. The
supplementary C++ implementation is frozen reference code.

## License

MIT
