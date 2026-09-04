# VaneDB

Embeddable vector database for edge AI, with Rust and header-only C++
implementations maintained under one contract.

## Quick start

### Rust

```rust
use vanedb::{DistanceMetric, HnswIndex};

let index = HnswIndex::builder(768, DistanceMetric::Cosine)
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

index = vanedb.HNSWIndex(768, vanedb.DistanceMetric.COSINE, capacity=100_000)
vecs = np.asarray(embeddings, dtype=np.float32)  # shape (n, 768)
index.add_batch(np.arange(len(vecs), dtype=np.uint64), vecs)
hits = index.search(vecs[0], 10)  # [(id, distance), ...]
```

Vector arguments accept any buffer-protocol object (numpy `float32` arrays,
`array.array`, memoryviews) as well as plain Python lists. `add_batch` is
all-or-nothing and releases the GIL while the index builds. The same batch
API is exposed in the wasm bindings (`Float32Array`/`BigUint64Array`) and
the C ABI (`vanedb_rs_*_add_batch`).

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
