# VaneDB for Python

VaneDB is an embeddable vector database backed by Rust. Store vectors and search
for their nearest neighbors inside your Python process, without a database
server. Supply your own embeddings; VaneDB does not generate them. It stores only
`(id, vector)` pairs — no metadata or payload storage and no filtered
search — so keep your own id-to-document mapping alongside it.

The `vanedb` package is the shipping Python implementation. C++ bindings are
kept in the repository for reference and local testing, and are not published.

If you do not yet have embeddings, the
[getting started guide](https://github.com/vanedb/vanedb/blob/main/docs/GETTING_STARTED.md)
sets up a provider (local Ollama, sentence-transformers, or OpenAI) and searches
text end to end.

## Installation

Requires Python 3.11 or newer.

```sh
python -m pip install vanedb==0.1.0rc2
```

The version is pinned because pip does not select a prerelease unless asked;
drop the pin once 0.1.0 is out. Building from a checkout is described in the
[repository](https://github.com/vanedb/vanedb/tree/main/vanedb-py).

Python lists work without additional dependencies. NumPy is optional; its arrays
can also be used for vector and batch inputs.

## Quick start

Use `FlatIndex` for exact search, or `ApproxIndex` for approximate search.
Both return `(id, distance)` pairs, with the nearest results first.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

from vanedb import Metric, ApproxIndex, FlatIndex

ids = [101, 202]
vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
query = [1.0, 0.0, 0.0]

# Exact cosine-distance search.
store = FlatIndex(3, Metric.COSINE)
store.add_batch(ids, vectors)
assert store.search(query, 1) == [(101, 0.0)]

# Approximate search; capacity is a reserve hint and the index can grow.
index = ApproxIndex(3, Metric.COSINE, capacity=100, seed=42)
index.add_batch(ids, vectors)
index.ef_search = 100
hits = index.search(query, 1)
assert hits == [(101, 0.0)]

# Save and reload the index without rebuilding it.
with TemporaryDirectory() as directory:
    path = Path(directory) / "index.bin"
    index.save(path)
    restored = ApproxIndex.load(path)
    assert restored.search(query, 1) == hits
```

Every path argument — `ApproxIndex.save`/`load`, `DiskIndexBuilder.save` and
`DiskIndex.open` — takes a `str` or any `os.PathLike`, so a `pathlib.Path`
needs no `str()` around it.

Supported metrics are `Metric.L2` (squared Euclidean distance),
`Metric.COSINE`, and `Metric.DOT` (negative dot product).
For each metric, smaller distances rank first. Vectors must match the store or
index dimension and contain finite values; IDs must be unique unsigned 64-bit
integers.

Exceptions are chosen so a caller can branch on the type rather than parse the
message. A missing id raises `KeyError`; validation errors, including negative
or out-of-range integer sizes and seeds, raise `ValueError`; a corrupt file
raises `ValueError` and an absent one `FileNotFoundError`, so "load it, or build
it if absent" needs no message matching. Other I/O failures raise `OSError`.
Arguments of the wrong type can raise `TypeError`.

`ApproxIndex.search` accepts a keyword-only `ef_search` that applies to that
query alone, leaving the shared `ef_search` property untouched — searches
release the GIL, so raising recall by assigning to the property is visible to
concurrent threads. `m`, `ef_construction` and `seed` are readable on any `ApproxIndex`,
including one from `ApproxIndex.load`, whose graph the caller did not build.
`get` and `get_vector` are the same operation on every index type, so swapping
`FlatIndex` for `ApproxIndex` does not mean renaming call sites.

`ApproxIndex` writes shared VNDB v2 graph files. Both engines preserve their
vectors, links, IDs and deleted slots; further insertions may differ across
engines. Legacy Rust graphs remain readable: load and save to a new path to
migrate. Older readers cannot open VNDB v2. Keep originals and source vectors
while verifying migration. During 0.x, APIs and persistence formats may change in a minor release.
Existing format identifiers will not be reinterpreted; new encodings require
new identifiers and readers for existing files are retained. Older readers
need not accept future formats. Future insertions need not reproduce identical topology. `DiskIndex`
continues to use shared VNDB v1 files. Disk construction buffers vectors in
memory before saving, while the opened index uses a read-only memory mapping.

`DiskIndexBuilder` is the only way to create a file `DiskIndex.open` accepts:

```python
from vanedb import DiskIndex, DiskIndexBuilder, Metric

builder = DiskIndexBuilder(3, Metric.L2)
builder.add(1, [1.0, 0.0, 0.0])
builder.add(2, [0.0, 1.0, 0.0])
builder.save("corpus.vndb")

index = DiskIndex.open("corpus.vndb")
print(len(index))                        # 2
print(index.search([0.9, 0.1, 0.0], 1))  # [(1, ...)]
```

`DiskIndex` has no constructor of its own — `DiskIndex(...)` raises `TypeError`.

For `DiskIndex.open(path)`, keep the underlying file immutable from before the
call until the last reference to the index is released. Prevent every process
from writing or truncating that file, including through other paths or open
handles. A read-only mapping does not prevent those changes; they can corrupt
results or crash Python. To update the data, write a separate file and atomically
replace the path where supported. Existing indexes keep the original mapping;
open a new index to read the replacement.

The C++ Python package has different constructor keywords and returns separate
id and distance arrays. Changing engines requires adapting those calls and
checking feature availability, in addition to changing the import.

## Project

- [Source code and examples](https://github.com/vanedb/vanedb)
- [Issues](https://github.com/vanedb/vanedb/issues)
- [MIT license](https://github.com/vanedb/vanedb/blob/main/LICENSE)
