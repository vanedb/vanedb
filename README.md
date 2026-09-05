# VaneDB

Embeddable vector database for edge AI, with Rust and header-only C++
implementations maintained under one contract.

## Quick start

### Rust

```rust
use vanedb::{Metric, Index};

let index = Index::builder(768, Metric::Cosine)
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

index = vanedb.Index(768, vanedb.Metric.COSINE, capacity=100_000)
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

Three ways to hold vectors. All answer the same question — which stored
vectors are nearest this query — and differ in where the data lives and
whether the answer is exact.

| Type | Data lives | Exact | Use it when |
|---|---|---|---|
| `Store` | memory | yes | the corpus is small, or you need exact results |
| `Index` | memory | no | search must stay fast as the corpus grows |
| `DiskStore` | a file, paged in on demand | yes | the corpus is larger than RAM |

`Store` and `DiskStore` scan every vector, so cost grows linearly. `Index`
searches a graph instead: sub-linear, and it can miss a true neighbour.
`ef_search` trades that recall against speed per query; `m` and
`ef_construction` set the graph's quality at build time.

Every type takes a `Metric` (`L2`, cosine, or dot) and returns results
nearest first. Only `Index` persists, with `save`/`load`. `DiskStore` is
written by `DiskStoreBuilder` and then opened read-only; `Store` is in-memory
only and is rebuilt on each run.

`Index` allocates its full `capacity` up front, so size it to the corpus:
100k x 768 floats reserves roughly 300 MB before the first insert.

All three types are reachable from every binding except wasm, which has no
filesystem to map and so omits `DiskStore`.

The Rust crate spells enum variants in Rust style (`Metric::Cosine`); the
Python and JavaScript packages use `Metric.COSINE`. Type names are identical
in every binding, so switching engines is an import change.

## Repository layout

| Path | Purpose |
|---|---|
| [`vanedb/`](vanedb) | Rust engine; canonical crate and `vanedb` PyPI implementation |
| [`vanedb-py/`](vanedb-py) | PyO3 bindings for `pip install vanedb` |
| [`vanedb-wasm/`](vanedb-wasm) | wasm-bindgen bindings |
| [`vanedb-capi/`](vanedb-capi) | Rust engine C ABI |
| [`cpp/`](cpp) | Supplementary header-only C++ engine and `vanedb-cpp` Python package |
| [`bench/`](bench) | Reproducible cross-engine benchmark harness |
| [`conformance/`](conformance) | Shared behavioral and persistence contract |

The Rust and C++ engines may make different internal trade-offs, but distance
semantics, persistence, structural safety, and search-quality expectations are
tested as one product. The canonical Python package is `vanedb`;
`vanedb-cpp` / `import vanedb_cpp` is supplementary.

Release tags are product-scoped: `vanedb-vX.Y.Z` for the canonical product
and `vanedb-cpp-vX.Y.Z` for the supplementary C++ distribution.

## License

MIT
