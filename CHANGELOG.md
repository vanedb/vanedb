# Changelog

Notable changes to the `vanedb` crate and the `vanedb` Python package. The
frozen C++ engine has its own log in [`cpp/CHANGELOG.md`](cpp/CHANGELOG.md).

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). This
project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html); until
1.0.0, breaking changes may land in a minor release.

## [0.1.1] - Unreleased

The first supported stable release. This section records its functionality and
the fixes made during prerelease verification.

### Security

- SIMD distance kernels took their loop bound from one slice while the scalar
  reference truncated to the shorter, so mismatched lengths read past the end
  of the shorter one. The kernels are safe `pub fn` in a public module, making
  this reachable from safe code. Every kernel is now bounded by both lengths.
- `ApproxIndex::load` multiplied file-controlled values without checking, so a
  497-byte file could wrap the product to zero and load with absurd
  parameters, or panic on a path documented to return an error.
- `DiskIndex::open` checked only that the file was not *shorter* than its
  header declared. The expected length is derived from that same header, so a
  header understating the geometry moved the goalpost instead of tripping the
  guard: one flipped bit in `dim` read the payload at the wrong stride and
  `get` returned a vector straddling two stored records. Both engines now
  require exact equality, and both bound `dim` independently so an empty store
  cannot declare an unaddressable dimension. The format carries no checksum, so
  `DiskIndex::open` documents what remains undetectable.
- `DiskIndex::open` accepted files with duplicate ids, after which `size`
  disagreed with lookups, `get` returned a row the id did not name, and a
  single search could return one id twice. Both engines now reject them.
- The `@claude` workflow restricts invocation to repository owners, members
  and collaborators, using GitHub's author associations.
- GitHub Actions are pinned to commits; Rust CI checks dependency advisories,
  licences, bans and sources with `cargo-deny`.

### Changed — public API before the stable release

- `SearchResult` is `#[non_exhaustive]`, so a field can be added later without
  a major version. `PartialEq`/`Ord` stay defined over `(id, distance)`.
- `DiskIndex::get` returns `Cow<'_, [f32]>` rather than `&[f32]`, so a store
  that encodes vectors as anything but native `f32` is still possible. It
  borrows today.
- `ApproxIndex::search_with(query, k, &SearchParams)` takes per-query options.
  `SearchParams` carries an unused lifetime deliberately: one cannot be added
  later, and without it a borrowed filter would be impossible forever.
- The C ABI rejects an unrecognised metric instead of using L2, and its
  `ef_search` argument is per-call — it previously mutated the shared index and
  was written into saved files.
- Python raises `ValueError`, not `OverflowError`, for every negative size, and
  releases the GIL in every method that reaches the index lock.
- **Python raises `KeyError`, not `ValueError`, when no vector is stored under
  an id.** `KeyError` subclasses `LookupError`, so a lookup miss is now
  separable by type from a dimension mismatch or a bad `k`, which previously
  needed message matching. The C ABI already treated a miss as its own status
  code (`VANEDB_RS_NOT_FOUND`); Python was the binding collapsing it into the
  validation bucket. Affects `get`, `get_vector` and `remove` on every index.
- `ApproxIndex.search` takes a keyword-only `ef_search` that applies to that
  query alone. Assigning to the shared property was the only way to raise
  recall for one query, and searches release the GIL, so the mutation was
  visible to concurrent threads — the same defect already fixed in the C ABI.
- Python exposes `m`, `ef_construction` and `seed` on `ApproxIndex`, matching
  the Rust core.
- `get` and `get_vector` now name the same read on **every** index type in
  every binding. Python and the C ABI already carried both spellings
  everywhere; Rust and WebAssembly had the pair on `ApproxIndex` only, so the
  one migration the pair exists to make painless — a graph index swapped for
  an exact one — was the one that stopped compiling. Adds
  `FlatIndex::get_vector` and `DiskIndex::get_vector` in Rust and
  `FlatIndex.get_vector` in WebAssembly.
- `FlatIndex::size`, `DiskIndex::len`/`is_empty`, `DiskIndexBuilder::len`/
  `is_empty` and `ApproxIndex::get` added, so the count and read spellings
  match across index types.
- **Python path arguments accept `os.PathLike`.** `ApproxIndex.save`/`load`,
  `DiskIndexBuilder.save` and `DiskIndex.open` took `str` only, so passing a
  `pathlib.Path` — the ordinary way to spell a path in modern Python — raised
  `TypeError`, and the guide's own example had to wrap it in `str()`.

### Added

- **A search-correctness suite that does not use the engine as its own
  reference.** `disk_tests` compared `DiskIndex` to `FlatIndex` and
  `approx_tests` measured recall against `FlatIndex`, which left the exact
  index — the thing every other cross-check trusts — checked by nothing. A
  systematic ordering or distance error there would have satisfied all of
  them. `tests/search_correctness.rs` computes the ranking in `f64` from the
  metric definitions, shares no code with `vanedb::distance`, and holds
  `FlatIndex`, `DiskIndex` and the graph to it across three metrics and twelve
  dimensions chosen to straddle the AVX2 and NEON tail boundaries. Also pins
  ascending-id tie-breaking on genuinely tied vectors, and that a graph search
  with a corpus-wide beam returns the exact answer.
- Python batch extraction is held to the same reference across Fortran-order,
  transposed, column-strided and reversed-row numpy inputs. The existing
  non-contiguous test used a row-strided slice, whose rows are still
  contiguous — the one layout that cannot expose a transposed read.
- **A real error channel in the C ABI.** Every status function used to return a
  bare `1` for a dimension mismatch, a duplicate id, a corrupt file, an I/O
  failure and a caught panic alike, and a search returning `0` results could
  not be told from an empty store. `vanedb_rs_last_error()` now reports one of
  sixteen `VANEDB_RS_*` codes and `vanedb_rs_last_error_message()` the detail,
  both thread-local. Existing return values are unchanged, so `!= 0` checks
  keep working. `VANEDB_RS_UNKNOWN` is a permanent catch-all because
  `VaneError` is `#[non_exhaustive]`.
- `vanedb_rs_version()` and a `VANEDB_RS_VERSION` header macro, so a consumer
  can check the shared object against the header it compiled against.
- C ABI accessors for a handle a caller did not build: `_index_m`,
  `_ef_construction`, `_seed`, `_capacity`, `_ef_search` and its setter, plus
  `get`/`get_vector` under both spellings on all three handle types.
- `vanedb_rs_index_search` treats `ef_search = 0` as "use the handle's stored
  value" rather than clamping the beam to `k`. The parallel C++ ABI rejects `0`
  instead; the header documents the difference.
- **WebAssembly `ApproxIndex` can measure and reclaim deletions.** It had
  `remove` and neither `tombstones` nor `compact`, so a browser application
  that churned entries grew without bound in the most memory-constrained
  runtime this crate targets. It also gained `get`/`get_vector`, an optional
  construction `seed` (previously hardcoded to 42), and `m`/`ef_construction`/
  `capacity` accessors.
- `DiskIndex` and `DiskIndexBuilder` appear in the published documentation,
  with the feature badge that says they need `disk`.
- Declared MSRV of 1.85, checked in CI on that exact toolchain.
- The generated C header is rebuilt in CI and must match what is committed;
  `build.rs` fails loudly instead of leaving a stale header in place.
- Rust `SearchParams` selects a graph query's beam width without changing
  defaults used by other callers.
- A committed `HNSW` v1 file, loaded as bytes, so a change to the encoder
  cannot silently stop older files from loading.
- `DiskIndex::open` documents that the mapped file must not change while open,
  on the Rust, C and Python surfaces.
- Python `Metric` supports pickling and hashing. Index objects reject pickling:
  save an `ApproxIndex`, build a file with `DiskIndexBuilder`, or rebuild a
  `FlatIndex` from its source vectors.

### Changed

- `DiskIndex` uses VNDB v1. New `ApproxIndex::save` files use the shared VNDB v2
  graph format; legacy HNSW files remain readable. The 0.x format can evolve in
  minor releases; existing identifiers are not reinterpreted and readers for
  existing files are retained. Rust and C++ preserve shared graph files across
  load/save. Keep originals and source vectors while verifying migration.
- Python methods that block — `save`, `load`, `upsert`, `remove`, and the
  `DiskIndexBuilder` methods — release the GIL. `DiskIndexBuilder` takes
  `&self` like every other class and can be shared between threads.
  Accessors that wait on index locks also release the GIL while waiting.
- `Metric::Dot` documents that it is not scale-invariant and not a metric, so
  a vector need not be its own nearest neighbour.
- Rust `SearchResult` is non-exhaustive; use its constructor. Disk lookup
  returns `Cow<[f32]>`, borrowing mapped vectors without copying.
- Rust disk open is unsafe: callers must keep the backing file unchanged
  throughout its mapped lifetime. Python and C consumers have the same
  requirement; atomic path replacement remains supported.
- The published recall figure is annotated as measured on uniform-random
  vectors, the adversarial case for a proximity graph.

### Fixed

- Copyright notices name a person rather than the project name, and the Python
  wheel ships the MIT text rather than a link to it.
- The PyPI description no longer opens on a sentence broken by a rename.
- The README no longer instructs JavaScript users to pass `Metric.COSINE`,
  which the wasm bindings reject.
- Optional Metal compute validates sizes, finite inputs, ID counts and device
  resources; tiny-value calculations use CPU kernels to preserve rankings.
- Both disk loaders reject nonzero reserved VNDB v1 header bytes. Python
  negative or out-of-range integer sizes and seeds consistently raise `ValueError`.
- The cosine "no usable direction" rule is documented and pinned at *both*
  ends. It is decided by the computed squared norm, so a vector whose
  components are below roughly 2.6e-23 squares to a zero norm and is reported
  1.0 from everything, itself included — neither zero nor overflowing, and
  previously described nowhere. Both engines already behaved this way;
  `cosine_scale_invariance.tsv` now has rows that keep them from diverging.
- `Metric::Dot` documents that an inner product large enough to overflow gives
  a distance of negative infinity, which the shared result order ranks *last*
  rather than first.
- Two dead intra-doc links in the AVX2 module. That module is compiled only on
  x86-64, and every local verification ran on macOS ARM64 where it is cfg'd
  away — so the links were broken on exactly the target docs.rs builds, and
  nothing in CI built the documentation at all. `cargo doc` now runs in CI in
  the docs.rs configuration with warnings denied.
- `.gitignore` covers the root `.venv/` the Python guides tell you to create,
  so following the documented workflow leaves nothing for `git add -A` to
  sweep in.
- `cargo test -p vanedb` on the **default** feature set did not compile:
  `debug_impls`, `io_error_paths` and `public_surface` imported
  `DiskIndexBuilder` unconditionally, and every CI leg passes `--features
  disk`, so nothing built it. `cargo check` cannot see this — it does not
  build test targets. The imports are gated and CI now runs the default
  feature set.

### Roadmap

- NVIDIA CUDA support is required after the initial release. The unimplemented Rust CUDA
  feature was removed; implementation and hardware acceptance requirements
  are recorded in the [roadmap](docs/ROADMAP.md).
