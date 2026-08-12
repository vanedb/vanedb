# VaneDB

Embeddable vector database for edge AI — Rust crate with Python (PyO3) and WASM bindings.

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

index = vanedb.PyHnswIndex(768, vanedb.PyDistanceMetric.Cosine, capacity=100_000)
vecs = np.asarray(embeddings, dtype=np.float32)  # shape (n, 768)
index.add_batch(np.arange(len(vecs), dtype=np.uint64), vecs)
hits = index.search(vecs[0], 10)  # [(id, distance), ...]
```

Vector arguments accept any buffer-protocol object (numpy `float32` arrays,
`array.array`, memoryviews) as well as plain Python lists. `add_batch` is
all-or-nothing and releases the GIL while the index builds. The same batch
API is exposed in the wasm bindings (`Float32Array`/`BigUint64Array`) and
the C ABI (`vanedb_rs_*_add_batch`).

## Implementations

- **Rust** (this repo) — `cargo add vanedb`, Python via `vanedb-py` (PyO3), WASM via `vanedb-wasm` (wasm-bindgen).
- **C++** — [vanedb/vanedb-cpp](https://github.com/vanedb/vanedb-cpp), header-only, no Rust toolchain required.

Both maintained side-by-side under the [@vanedb](https://github.com/vanedb) org. Features generally land in Rust first; core algorithms and persistence formats are synced to C++.

## License

MIT
