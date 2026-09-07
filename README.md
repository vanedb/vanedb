# VaneDB

Embeddable vector database for edge AI.

The Rust engine in [`vanedb/`](vanedb) is the one that ships. A header-only
C++ engine in [`cpp/`](cpp) is kept as reference code and as the other arm of
a cross-engine benchmark; it is frozen, and features are not ported to it.

Bring your own embeddings: VaneDB stores and searches vectors, it does not
generate them.

## Status

Pre-release. Nothing is published to crates.io or PyPI yet, so there is no
install command to give — build from this repository. Install lines land with
the first release (#122).

```toml
# Cargo.toml — DiskIndex is behind the non-default `disk` feature
vanedb = { path = "vanedb", features = ["disk"] }
```

## Quick start

### Rust

```rust
use vanedb::{Metric, ApproxIndex};

let index = ApproxIndex::builder(768, Metric::Cosine)
    .capacity(100_000)
    .build()?;
index.add(1, &embedding)?;             // single insert
index.add_batch(&ids, &flat_vectors)?; // bulk insert, row-major n × dim floats
let hits = index.search(&query, 10)?;
```

### Python

```python
import numpy as np
import vanedb

index = vanedb.ApproxIndex(768, vanedb.Metric.COSINE, capacity=100_000)
vecs = np.asarray(embeddings, dtype=np.float32)  # shape (n, 768)
index.add_batch(np.arange(len(vecs), dtype=np.uint64), vecs)
hits = index.search(vecs[0], 10)  # [(id, distance), ...]
```

Vector arguments accept any buffer-protocol object (numpy `float32` arrays,
`array.array`, memoryviews) as well as plain Python lists. `add_batch` is
all-or-nothing and releases the GIL while the index builds. The same batch
API is exposed in the wasm bindings (`Float32Array`/`BigUint64Array`) and
the C ABI (`vanedb_rs_*_add_batch`).

## API

Three indexes. All answer the same question — which stored vectors are nearest
this query — and each name says why you would pick it over the others.

| Type | Exact | Data lives | Pick it when |
|---|---|---|---|
| `FlatIndex` | yes | memory | the corpus is small, or you need exact results |
| `ApproxIndex` | **no** | memory | search must stay fast as the corpus grows |
| `DiskIndex` | yes | a file, paged in on demand | the corpus is larger than RAM |

`FlatIndex` and `DiskIndex` scan every vector, so cost grows linearly and the
answer is always right. The saving is on the reading side: `DiskIndex` maps
the file and the kernel pages vectors in as the scan touches them, so a corpus
larger than RAM stays searchable. Building one is the other half —
`DiskIndexBuilder` holds the vectors in memory until `save`. `ApproxIndex` walks an HNSW graph instead: sub-linear,
and it can miss a true neighbour. `ef_search` trades that recall against speed
per query; `m` and `ef_construction` set the graph's quality at build time.

Every type takes a `Metric` (`L2`, cosine, or dot) and returns results nearest
first. Only `ApproxIndex` persists, with `save`/`load`. `DiskIndex` is written
by `DiskIndexBuilder` and then opened read-only; `FlatIndex` is in-memory only
and is rebuilt on each run.

`ApproxIndex` allocates chunks as vectors arrive, so `capacity` is a reserve
hint rather than a ceiling and an unused index costs nothing.

`remove` tombstones: the node keeps its graph links, which may be the only
route between live neighbourhoods, and simply stops appearing in results.
`upsert` replaces an entry under one lock, so a concurrent reader never sees
the id missing.

Neither reclaims space — a replaced or removed slot stays allocated, so a
long-running upsert loop grows the index even at constant length. `tombstones()`
reports how many are outstanding and `compact()` rebuilds without them. Compaction
is a full rebuild and holds the write lock throughout, so call it deliberately
rather than on every write. `save` writes tombstoned slots too, so compact first
if file size matters.

All three are reachable from Rust, Python and the C ABI. The wasm bindings
have no filesystem, so `DiskIndex` is absent there. `ApproxIndex` in wasm has
no `save`/`load` either — an index is built in the page it is used in.

The Rust crate spells enum variants in Rust style (`Metric::Cosine`) and
Python uses `Metric.COSINE`; the wasm bindings take a string naming the
metric (`"l2"` or `"L2"`, `"cosine"` or `"Cosine"`, `"dot"` or `"Dot"` — not
`"COSINE"`). Type names are otherwise identical across bindings.

## Repository layout

| Path | Purpose |
|---|---|
| [`vanedb/`](vanedb) | Rust engine; canonical crate and `vanedb` PyPI implementation |
| [`vanedb-py/`](vanedb-py) | PyO3 bindings for `pip install vanedb` |
| [`vanedb-wasm/`](vanedb-wasm) | wasm-bindgen bindings |
| [`vanedb-capi/`](vanedb-capi) | Rust engine C ABI |
| [`cpp/`](cpp) | Header-only C++ engine: reference code and the benchmark's other arm. Not published |
| [`bench/`](bench) | Reproducible cross-engine benchmark harness |
| [`conformance/`](conformance) | Shared behavioral and persistence contract |

The Rust and C++ engines may make different internal trade-offs, but distance
semantics, the `DiskIndex` format, structural safety, and search-quality
expectations are tested as one product. The graph format is Rust-only. The Python package is `vanedb`, built from the Rust
engine. The C++ engine is not published: it stays in the repository as
reference code and as the control the benchmark measures against, which is
where #32, #77, #109 and #110 came from.

Release tags are product-scoped, and the crate has its own:

| Tag | Publishes |
|---|---|
| `vanedb-crate-vX.Y.Z` | the `vanedb` crate to crates.io |
| `vanedb-vX.Y.Z` | the `vanedb` wheels to PyPI |

The crate releases on its own tag rather than sharing one with the wheels: the
name has to exist on crates.io before any wheel does, and a failed crate
publish should not strand a half-released set of wheels.

## License

MIT
