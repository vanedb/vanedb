# VaneDB

Embeddable vector database for edge AI.

Bring your own embeddings: VaneDB stores and searches vectors; it does not
generate them. It holds only `(u64 id, vector)` pairs — no arbitrary metadata
storage — with support for filtered search via ID allow/deny sets and predicates,
so keep your own id-to-document mapping alongside it.

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

## Demo

[Vane Search](https://github.com/vanedb/obsidian-vane-search) is a local-first
Obsidian plugin that indexes a vault with VaneDB's WASM build and searches by
meaning. Pair it with Ollama + `nomic-embed-text` (or any OpenAI-compatible
embeddings endpoint) once you follow the plugin README. The residual #242
“Try it on a real vault” walkthrough / official 0.2.0 demo slice is still
maintainer-tracked (see
[`docs/launch/0003-demo-update-checklist.md`](docs/launch/0003-demo-update-checklist.md)).
Full #198/#242 closeout (fixture host + dedicated HW tables + demo release):
[`docs/launch/0003-closeout-checklist.md`](docs/launch/0003-closeout-checklist.md).
Competitor methodology (and dedicated-hardware results, when filled) live in
[`bench/COMPARISON.md`](bench/COMPARISON.md).

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

Every binding can set that beam per query, leaving the index's own default
alone so concurrent callers can choose different widths. In Rust, construct
`let params = vanedb::SearchParams::new().ef_search(100);` and call
`index.search_with(&query, 10, &params)`; Python takes `ef_search=` on `search`,
WebAssembly a third argument, and the C ABI a parameter. The effective beam is
at least `k`; ordinary `search` uses the index's defaults.

Filtered search is supported across all indexes and bindings. Rust uses
`SearchParams::new().filter(Filter::Allow(&ids))`; Python accepts `filter=`,
`allow_ids=`, or `deny_ids=`; WebAssembly accepts an options object with `allow`,
`deny`, or `predicate`; and the C ABI exposes `*_search_filtered`. Choose one
filter per query. ID lists must be strictly ascending without duplicates;
an empty allow list matches nothing, and an empty deny list matches everything.
Predicates should consult external metadata and must not call methods on the
index being searched. Python and JavaScript predicate exceptions propagate to
the caller. ID lists avoid crossing the language boundary per candidate.

`FlatIndex` and `DiskIndex` return the exact nearest matching vectors.
`ApproxIndex` still traverses excluded nodes and automatically widens its beam
when fewer than `k` matches are found, up to `max_ef_search` (default four times
the initial beam). A selective filter can still return fewer than `k` matches
or miss a true neighbor; measure recall on your data and raise `ef_search`
and the cap, or use an exact index when needed. Widening stops once `k`
matches are found, so increasing only the cap may not improve their quality.
The cap bounds beam width, not the total number of distance evaluations.

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
renaming call sites. A lookup miss is a value, not an error: Rust returns
`Ok(None)`, Python `None`, JavaScript `undefined`, and the C ABI its
`VANEDB_RS_NOT_FOUND` status; `contains` is the cheaper probe when the vector
is not needed, and `remove` of a missing id stays an error. The count is
`len()` in Rust, `len(index)` in Python (`size()` is kept as an alias), `size()`
in JavaScript and `vanedb_rs_*_len` in C; the search-beam default is
`ef_search()`/`set_ef_search()`, the `ef_search` property, the `efSearch`
property and `vanedb_rs_index_ef_search`/`_set_ef_search` respectively. This
vocabulary is [RFC 0011](docs/rfcs/0011-api-vocabulary-before-1-0.md), tabled
in [`conformance/vocabulary/README.md`](conformance/vocabulary/README.md).

`ApproxIndex` allocates chunks as vectors arrive, so `capacity` is a reserve
hint rather than a ceiling. Vector storage grows on demand. Hard limits,
memory and file-size formulas, and what is not supported are collected in
[`docs/LIMITS.md`](docs/LIMITS.md).

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
`upsert`, `tombstones`, `compact` and persistence (`toBytes` / `fromBytes`,
plus `save(name)` / `load(name)` over IndexedDB in the browser and the
filesystem in Node) on `ApproxIndex`. Approximate search accepts a
per-query beam override as `search(query, k, efSearch)`, leaving the
`efSearch` property unchanged. A single id is a
JavaScript `bigint`; batch ids are a `BigUint64Array` and vectors a row-major
`Float32Array`. Build a browser package from the repository root with
`wasm-pack build vanedb-wasm --target web --release --out-dir pkg-web --locked`
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
| [`bench/`](bench) | Cross-engine (C++/Rust) harness and [`competitor comparison`](bench/COMPARISON.md) |
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

Planned work, including CUDA and physical-device verification, is indexed in
[`docs/ROADMAP.md`](docs/ROADMAP.md); current limits are in
[`docs/LIMITS.md`](docs/LIMITS.md).

## License

MIT
