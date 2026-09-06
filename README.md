# VaneDB

Embeddable vector database for edge AI.

The Rust engine in [`vanedb/`](vanedb) is the one that ships. A header-only
C++ engine in [`cpp/`](cpp) is kept as reference code and as the other arm of
a cross-engine benchmark; it is frozen, and features are not ported to it.

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
answer is always right. `ApproxIndex` walks an HNSW graph instead: sub-linear,
and it can miss a true neighbour. `ef_search` trades that recall against speed
per query; `m` and `ef_construction` set the graph's quality at build time.

Every type takes a `Metric` (`L2`, cosine, or dot) and returns results nearest
first. Only `ApproxIndex` persists, with `save`/`load`. `DiskIndex` is written
by `DiskIndexBuilder` and then opened read-only; `FlatIndex` is in-memory only
and is rebuilt on each run.

`ApproxIndex` allocates chunks as vectors arrive, so `capacity` is a reserve
hint rather than a ceiling and an unused index costs nothing.

All three are reachable from every binding except wasm, which has no
filesystem to map and so omits `DiskIndex`.

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
