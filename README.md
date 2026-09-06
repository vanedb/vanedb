# VaneDB

Embeddable vector database for edge AI.

The Rust engine in [`vanedb/`](vanedb) is the one that ships. A header-only
C++ engine in [`cpp/`](cpp) is kept as reference code and as the other arm of
a cross-engine benchmark; it is frozen, and features are not ported to it.

Bring your own embeddings: VaneDB stores and searches vectors; it does not
generate them. This checkout is pre-release.

## Quick start

### Rust

With a Rust toolchain installed, run the example from this repository's root:

```sh
cargo run -p vanedb --example quickstart --locked
```

For a local application, add `vanedb = { path = "/path/to/vanedb/vanedb" }`
to its Cargo dependencies. Add `features = ["disk"]` when using disk indexes.
The complete example is:

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

### Python

Requires Python 3.11 or newer and a Rust toolchain to build from source.
From the repository root, create and activate a virtual environment, then install:

```sh
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install ./vanedb-py
```

```python
from vanedb import ApproxIndex, Metric

index = ApproxIndex(3, Metric.COSINE, capacity=100)
index.add_batch([101, 202], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
hits = index.search([1.0, 0.0, 0.0], 1)
assert hits == [(101, 0.0)]
print(hits)
```

See the [Python guide](vanedb-py/README.md) for exact search and saving an index.
These instructions use the checkout; they do not assume a published 1.0.0 package.

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
results are exact under the selected distance metric. `DiskIndex` maps the file
and pages vectors in as the scan touches them, so a corpus larger than RAM stays
searchable. Building one still buffers its vectors in memory until `save`.
`ApproxIndex` walks an HNSW graph and can miss a true neighbour. `ef_search` trades recall against speed
per query; `m` and `ef_construction` set the graph's quality at build time.

Every type takes a `Metric` (`L2`, cosine, or dot) and returns results nearest
first. The native `ApproxIndex` supports `save`/`load`. `DiskIndex` is written
by `DiskIndexBuilder` and then opened read-only; `FlatIndex` is in-memory only
and is rebuilt on each run.

`ApproxIndex` allocates chunks as vectors arrive, so `capacity` is a reserve
hint rather than a ceiling. Vector storage grows on demand.

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

## Bindings and platforms

| Binding | Available indexes | Metrics and results |
|---|---|---|
| Rust | Flat, Approx, Disk (`disk` feature) | `Metric::Cosine`; `Vec<SearchResult>` |
| Python (`vanedb`) | Flat, Approx, Disk | `Metric.COSINE`; list of `(id, distance)` pairs |
| JavaScript / WebAssembly | Flat, Approx | `"cosine"` string; `SearchResults` with `ids` and `distances` arrays |
| C ABI | Flat, Approx, Disk | `VANEDB_RS_COSINE`; caller-provided id and distance arrays |

WebAssembly currently supports add, batch add, search, lookup methods, and
remove; it does not expose persistence, upsert, or compaction. A single id is a
JavaScript `bigint`; batch ids are a `BigUint64Array` and vectors a row-major
`Float32Array`. Build a browser package from the repository root with
`wasm-pack build vanedb-wasm --target web --release --locked` after installing
`wasm-pack` and the `wasm32-unknown-unknown` Rust target. The generated `pkg/`
directory includes JavaScript, TypeScript declarations, and the wasm module.
The [JavaScript guide](vanedb-wasm/README.md) includes runnable Node and browser
examples. Initialize that module before constructing an index. For example, the approximate
constructor takes `(3, "cosine", 100, 16, 200)`.

Build the native C library with `cargo build -p vanedb-capi --release --locked`.
See the [C guide](vanedb-capi/README.md) for a complete example that links and
runs against the shared library. The library is written to `target/release`; use the generated
[C header](vanedb-capi/include/vanedb_rs_capi.h) for ownership, buffer sizes,
return conventions, and metric constants.

CI runs native Rust tests on Linux x86-64/ARM64, macOS Intel/ARM64, and Windows x86-64,
and WebAssembly tests in Node.js and headless Chrome. Packaged browser acceptance
also runs in Chrome, Firefox and WebKit. Successful runs provide C library archives
and separate Node/browser npm tarballs, each tested as a consumer artifact.
Python release workflows build and test
Linux x86-64/ARM64 (glibc and musl), macOS Intel/ARM64, and Windows x64 wheels
for Python 3.11–3.14. Mobile CI cross-compiles the Rust core and C ABI for
iOS ARM64 and Android ARM64/x86-64, and runs C ABI acceptance on an iOS ARM64
simulator and Android x86-64 emulator. Physical-device and Android ARM64 runtime
verification remain required before claiming 1.0.0 support
on every platform.

The supplementary C++ engine has different constructor arguments, result shapes,
and feature coverage. Moving between Python engines requires adapting the API,
in addition to changing the import.

## Persistence

`DiskIndex` uses the shared **VNDB v1** format: fixed-width little-endian fields,
with [format specification and cross-engine fixtures](conformance/README.md).
Both Rust and C++ can read these files. Building a disk index currently buffers
its vectors in memory before saving; searches use a read-only memory mapping.

`ApproxIndex` still uses an engine-specific pre-release format. Rust graph files
cannot be loaded by C++, or vice versa. Universal graph persistence is unfinished;
there is no public 1.0.0 persistence compatibility promise yet. Keep source vectors
so you can rebuild when migrating formats.

## Repository layout

| Path | Purpose |
|---|---|
| [`vanedb/`](vanedb) | Rust engine; canonical crate and `vanedb` PyPI implementation |
| [`vanedb-py/`](vanedb-py) | PyO3 bindings for `pip install vanedb` |
| [`vanedb-wasm/`](vanedb-wasm) | wasm-bindgen bindings |
| [`vanedb-capi/`](vanedb-capi) | Rust engine C ABI |
| [`cpp/`](cpp) | Frozen C++ reference engine and local Python bindings |
| [`bench/`](bench) | Reproducible cross-engine benchmark harness |
| [`conformance/`](conformance) | Shared behavioral and persistence contract |

Rust is the shipping engine. C++ source tests and shared disk-format fixtures
keep the reference implementation useful for the benchmark comparison.

Release tags are `vanedb-vX.Y.Z` for Python and `vanedb-crate-vX.Y.Z`
for the Rust crate. The C++ reference is not published to PyPI.

## License

MIT
