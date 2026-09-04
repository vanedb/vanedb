# VaneDB for Python

VaneDB is an embeddable vector database backed by Rust. Store vectors and search
for their nearest neighbors inside your Python process, without a database
server. Supply your own embeddings; VaneDB does not generate them.

The `vanedb` package is the canonical Python implementation. The supplementary
C++ package is named `vanedb-cpp` and imports as `vanedb_cpp`.

## Installation

Requires Python 3.11 or newer. To install a published release:

```sh
python -m pip install vanedb
```

Python lists work without additional dependencies. NumPy is optional; its arrays
can also be used for vector and batch inputs.

## Quick start

Use `VectorStore` for exact search, or `HNSWIndex` for approximate search.
Both return `(id, distance)` pairs, with the nearest results first.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

from vanedb import DistanceMetric, HNSWIndex, VectorStore

ids = [101, 202]
vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
query = [1.0, 0.0, 0.0]

# Exact cosine-distance search.
store = VectorStore(3, DistanceMetric.COSINE)
store.add_batch(ids, vectors)
assert store.search(query, 1) == [(101, 0.0)]

# Approximate search, with a fixed maximum capacity.
index = HNSWIndex(3, DistanceMetric.COSINE, capacity=100, seed=42)
index.add_batch(ids, vectors)
index.ef_search = 100
hits = index.search(query, 1)
assert hits == [(101, 0.0)]

# Save and reload the index without rebuilding it.
with TemporaryDirectory() as directory:
    path = str(Path(directory) / "index.bin")
    index.save(path)
    restored = HNSWIndex.load(path)
    assert restored.search(query, 1) == hits
```

Supported metrics are `DistanceMetric.L2` (squared Euclidean distance),
`DistanceMetric.COSINE`, and `DistanceMetric.DOT` (negative dot product).
For each metric, smaller distances rank first. Vectors must match the store or
index dimension and contain finite values; IDs must be unique unsigned 64-bit
integers.

Persistence formats are currently pre-release and engine-specific. Do not use
the Rust package to load C++ files, or vice versa; a shared format is planned.

## Project

- [Source code and examples](https://github.com/vanedb/vanedb)
- [Issues](https://github.com/vanedb/vanedb/issues)
- [MIT license](https://github.com/vanedb/vanedb/blob/main/LICENSE)
