# VaneDB

Embeddable vector database for edge AI.

Bring your own embeddings: VaneDB stores and searches vectors; it does not
generate them. It holds only `(u64 id, vector)` pairs — no metadata or payload
storage and no filtered search — so keep your own id-to-document mapping
alongside it.

New to embeddings, or unsure where the vectors come from? Start with
[Getting started: from text to search results](docs/GETTING_STARTED.md) — it
sets up a local or hosted embedding provider and searches real sentences in
about ten minutes.

## Quick start

### Rust

```sh
cargo add vanedb                     # add --features disk for disk indexes
```

To run this example from a checkout instead:
`cargo run -p vanedb --example quickstart --locked`.

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

Requires Python 3.11 or newer.

```sh
python -m pip install vanedb
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

For a Rust query with its own recall setting, construct
`let params = vanedb::SearchParams::new().ef_search(100);` and call
`index.search_with(&query, 10, &params)`. These options leave the index's default
unchanged, so concurrent callers can choose different beam widths. The effective
beam is at least `k`; ordinary `search` uses the index's defaults.

Every type accepts a `Metric` (`L2`, cosine, or dot), defaulting to `L2` in the
Python bindings; wasm takes it as a required string argument. Results come back
nearest
first. The native `ApproxIndex` supports `save`/`load`. `DiskIndex` is written
by `DiskIndexBuilder` and then opened read-only; `FlatIndex` is in-memory only
and is rebuilt on each run.

Inspect an index's metric with `metric()` in Rust or WebAssembly, the `metric`
property in Python, or the corresponding `vanedb_rs_*_metric` C accessor.
This is useful after loading a file: queries must use its stored distance
convention. `get` and `get_vector` are the same read under two names, on every
index type in every binding, so swapping one index for another does not mean
renaming call sites.

`ApproxIndex` allocates chunks as vectors arrive, so `capacity` is a reserve
hint rather than a ceiling. Vector storage grows on demand.

`remove` tombstones: the node keeps its graph links, which may be the only
route between live neighbourhoods, and simply stops appearing in results.
`upsert` replaces an entry under one lock, so a concurrent reader never sees
the id missing.

Neither reclaims space — a replaced or removed slot stays allocated, so a
long-running upsert loop grows the index even at constant length. `tombstones`
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

WebAssembly currently supports add, batch add, search, lookup methods, remove,
`tombstones` and `compact`; it does not expose persistence or upsert. A single id is a
JavaScript `bigint`; batch ids are a `BigUint64Array` and vectors a row-major
`Float32Array`. Build a browser package from the repository root with
`wasm-pack build vanedb-wasm --target web --release --locked --out-dir pkg-web`
after installing `wasm-pack` and the `wasm32-unknown-unknown` Rust target. The
`--out-dir` matters: the Node target also defaults to `pkg/`, and whichever
build runs second silently overwrites the first. The generated directory
includes JavaScript, TypeScript declarations, and the wasm module.
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
simulator and Android x86-64 emulator. The 0.1.1 release also passed Android ARM64
acceptance on an Android 15 emulator with 16 KiB pages using the CI-built binary;
see the [release evidence](docs/release/0.1.1-readiness.md). The accepted initial-release mobile
verification scope is simulator/emulator based. Physical-device acceptance
remains a follow-up; these results do not establish behavior on an iPhone or
Android device.

The supplementary C++ engine has different constructor arguments, result shapes,
and feature coverage. Moving between Python engines requires adapting the API,
in addition to changing the import.

## Persistence

`DiskIndex` uses the shared **VNDB v1** format: fixed-width little-endian fields,
with [format specification and cross-engine fixtures](conformance/README.md).
Both Rust and C++ can read these files. Building a disk index currently buffers
its vectors in memory before saving; searches use a read-only memory mapping.

The mapped file must remain immutable from before opening it until every mapped
index using it has been released. This applies to Rust, Python and C consumers:
prevent writes and truncation by all processes, even through another path or
file handle. A read-only mapping does not enforce this requirement; violating
it can corrupt results or crash the process. Rust makes `DiskIndex::open` unsafe
to express this caller obligation. In Python, keep the file unchanged until
the last reference to the index is released; in C, until all reads have finished
and its handle has been freed. To update data, write a separate file and
atomically replace the path where supported. Existing mappings keep the old
file; new opens see the replacement.

`ApproxIndex` now writes the shared [VNDB v2 graph format](conformance/graph/README.md).
Both engines preserve its vectors, links, IDs and deleted slots across load/save.
Further insertions can produce different graphs across engines. Rust still reads
legacy Rust v1/v2 files; C++ still reads legacy C++ v1/v2/v3 files. To migrate,
load a legacy file in its original engine and save to a new path; older readers
cannot open VNDB v2. Keep the original file and source vectors while verifying
the migration. During 0.x, APIs and persistence formats may change in a minor release.
Existing format identifiers will not be reinterpreted; new encodings require
new identifiers and readers for existing files are retained. Older readers
need not accept future formats.
This does not guarantee identical future graph topology after insertions.

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

The Rust engine in [`vanedb/`](vanedb) is the one that ships. The header-only
C++ engine in [`cpp/`](cpp) is frozen reference code and the other arm of the
cross-engine benchmark; features are not ported to it, and its source tests and
shared disk-format fixtures are what keep the comparison honest.

Release tags are scoped to each distribution:

| Tag | Publishes |
|---|---|
| `vanedb-crate-vX.Y.Z` | the `vanedb` crate to crates.io |
| `vanedb-vX.Y.Z` | the `vanedb` wheels to PyPI |
| `vanedb-wasm-vX.Y.Z` | the `@vanedb/wasm` package to npm |

Separate tags allow each distribution to be verified and published independently.
The C++ reference is not published to PyPI. See the [changelog](CHANGELOG.md)
and [security policy](SECURITY.md) for release changes and vulnerability reports.

## Roadmap

CUDA support for NVIDIA GPUs is a required, high-priority follow-up after the initial release.
It is excluded from this release. The [roadmap](docs/ROADMAP.md) records the
implementation, hardware verification and performance requirements, plus mobile
physical-device follow-up. No delivery version or date has been assigned.

## License

MIT
