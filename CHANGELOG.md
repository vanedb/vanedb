# Changelog

Notable changes to the `vanedb` crate and the `vanedb` Python package. The
frozen C++ engine has its own log in [`cpp/CHANGELOG.md`](cpp/CHANGELOG.md).

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). This
project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html); until
1.0.0, breaking changes may land in a minor release.

## [Unreleased]

Target: **0.2.0**. This is an unreleased candidate; the version bump does not
establish release readiness or publication. The [draft release notes](docs/release/0.2.0-notes.md)
record the remaining integration and evidence dependencies.

### Added

- Filtered search (RFC 0004 / #199): Predicate, Allow, and Deny filters across `FlatIndex`, `ApproxIndex`, and `DiskIndex`.
- `SearchParams::filter` and `SearchParams::max_ef_search` with automatic beam widening for HNSW approximate search.
- Filter options across all bindings:
  - Python (`vanedb-py`): `filter=`, `allow_ids=`, and `deny_ids=` keyword arguments.
  - WebAssembly (`vanedb-wasm`): `{ allow, deny, predicate, efSearch, maxEfSearch }` options.
  - C ABI (`vanedb-capi`): `vanedb_rs_store_search_filtered`, `vanedb_rs_index_search_filtered`, and `vanedb_rs_disk_search_filtered`.
- `VaneError::Validation`, a new public variant for a filter argument that
  fails validation, such as an allow or deny list that is not strictly
  ascending; the C ABI reports it as `VANEDB_RS_INVALID_PARAMETER` and
  Python raises `ValueError`.
- `ApproxIndex::save_to`, `load_from`, `to_bytes` and `from_bytes`. Path
  `save`/`load` wrap these; the bytes are a VNDB v2 file (or a legacy Rust
  graph on load). The corruption suite runs against the byte path.
- WebAssembly `ApproxIndex.toBytes()` / `fromBytes()`, and on the
  `@vanedb/wasm` package asynchronous `save(name)` / `load(name)` over a
  `Storage` adapter. IndexedDB is the browser default; the filesystem is the
  Node default. `load` of an unknown name resolves to `null`.
- Python `ApproxIndex.to_bytes` / `from_bytes` and C ABI
  `vanedb_rs_index_save_to_buffer` / `vanedb_rs_index_load_from_buffer`.

### Changed — breaking

One API vocabulary across the four bindings, settled before 1.0 while the
user count is zero ([RFC 0011](docs/rfcs/0011-api-vocabulary-before-1-0.md),
#206; folds #85 and #86). A lookup miss is a value, not an error; the count
has one canonical spelling per binding; the search-beam default has one shape
per binding. `contains` and `remove` are unchanged everywhere: `remove` of a
missing id stays an error, because a caller that removes what is not there
has a bug. No file-format change. The conformance table is
[`conformance/vocabulary/README.md`](conformance/vocabulary/README.md).

- **Rust**: `get` and `get_vector` return `Result<Option<Vec<f32>>>` on
  `FlatIndex` and `ApproxIndex` and `Result<Option<Cow<[f32]>>>` on
  `DiskIndex`; a missing id is `Ok(None)`, no longer
  `Err(VaneError::NotFound)`. `size()` is removed from `FlatIndex`,
  `ApproxIndex`, `DiskIndex` and `DiskIndexBuilder`; `len()` and `is_empty()`
  stay on all of them. `ApproxIndex::get_ef_search` is renamed `ef_search`.
  Migration: `.size()` → `.len()`, `get_ef_search()` → `ef_search()`, and
  where a miss was an error, `index.get(id)?` →
  `index.get(id)?.ok_or(VaneError::NotFound { id })?`.
- **Python**: `get` and `get_vector` return `None` for a missing id instead
  of raising; `KeyError` is no longer raised by any method, and `remove` of a
  missing id raises `ValueError` like every other invalid argument. `Metric`
  is an `enum.IntEnum` — `.name`, `.value`, `list(Metric)`, `Metric(1)`,
  hashing and pickling all work — with the same integer values, and the
  constructors accept a member or its value. `len(index)` is the count and
  truth testing the emptiness test; `size()` is kept as a documented alias.
  Migration: `try: v = index.get(id) except KeyError: ...` →
  `if (v := index.get(id)) is None: ...`, and `except KeyError` around
  `remove` → `except ValueError`.
- **WebAssembly**: `get` and `get_vector` return `undefined` for a missing id
  instead of throwing; the `ef_search` property is renamed `efSearch`,
  matching the `{ efSearch }` search option, and the number-argument error
  message names it that way. In the TypeScript declarations the return type
  of `get`/`get_vector` becomes `Float32Array | undefined`, which is a
  compile error under strict null checks until the miss is handled, and
  `ApproxIndex.search`'s third parameter is declared `efSearchOrOptions`
  (positional, so no runtime change). Migration:
  `try { index.get(id) } catch { ... }` → `index.get(id) ?? fallback`, and
  `index.ef_search` → `index.efSearch`.
- **C ABI**: unchanged. `vanedb_rs_store_len`, `vanedb_rs_index_len` and
  `vanedb_rs_disk_len` already spell the count, a miss is already the
  `VANEDB_RS_NOT_FOUND` status, and `vanedb_rs_index_ef_search` /
  `_set_ef_search` already exist; the regenerated header is identical.

### Changed

- Internal id maps use an in-crate multiplicative hasher instead of SipHash
  (RFC 0010, #109); `DiskIndexBuilder::add` and the batch adds check
  finiteness block-wise (#77). No format or API change: saved bytes and
  result order are pinned by `vanedb/tests/rfc0010_identity.rs`.
- Platform support is tiered in `docs/PLATFORMS.md` (RFC 0012), the only
  place a tier is asserted. It records what CI proves on each platform, the
  floors (Rust 1.85, glibc 2.34 for C archives, macOS 11.0/10.12, Android
  API 21, Python 3.11 to 3.14; iOS 13 as a declared target and Node.js 22 as
  the tested version over a declared floor of 18) and a dated change log.
  The README summarises and links it, `SECURITY.md` links it, and a test
  fails when the README names a platform the page does not.
- The `gpu-metal` feature is documented as experimental (#208): it exposes the
  standalone `MetalCompute` scan API on macOS and accelerates no index. The
  crate README no longer presents it as a capability; `docs/LIMITS.md` records
  what it does and does not do. The feature, its tests and its CI jobs are
  unchanged.
- `Filter` is `#[non_exhaustive]`: a `match` on it needs a `_` arm, so the
  payload filter variant planned in RFC 0009 will not be a breaking change.
  Unreleased, so no released code is affected.

### Deprecated

- Native x86-64 macOS wheels and the `macos-x86_64` C archive end when
  GitHub's `macos-15-intel` runner retires in August 2027. No cross-compiled
  or `universal2` substitute is published; the sdist and `cargo build` keep
  working on Intel Macs (RFC 0012, #47).

### Fixed

- Filtered beam widening on `ApproxIndex` ended after a single pass at default
  settings: the loop compared the number of nodes a pass visited against
  `max_ef_search`, and one `ef_search = 50` pass on an `M = 16` graph already
  visits far more than 4 x 50 nodes. `max_ef_search` now bounds beam width
  only, as documented. On the pinned 100k embedding fixture a 1%-selectivity
  allow list at default settings returns `k` matches for 99.0% of queries
  instead of 30.8%, and recall@10 at 1% rises from 35.54% to 51.50%. A query
  whose first pass finds fewer than `k` matches now runs the additional
  widening passes the cap allows, so such queries do more work than before;
  queries that fill `k` on the first pass are unchanged. Unreleased, so no
  released behaviour changes.
- Python and WebAssembly predicates that call a method on the index being
  searched (`add` inside the callback, or a nested `search` while another
  thread has a writer queued) deadlocked on that index's read lock. The call
  now raises `RuntimeError` in Python and throws an `Error` with
  `code === "ERR_REENTRANT_SEARCH"` in JavaScript. The guard covers every
  call on the searched index from its predicate, including nested reads
  such as `len`, `contains` or a second `search` that previously happened to
  succeed when no writer was waiting.

## [0.1.1] - 2026-09-10

The first supported release. It is a 0.x release: as stated above, APIs and
persistence formats may still change in a minor release. This section records
its functionality and the fixes made during prerelease verification.

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

### Changed — public API before the first release

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
- A getting-started guide for embedding providers
  ([`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md)), covering local Ollama
  and hosted OpenAI, the document/query prefixes Nomic requires, and storing
  the ids alongside your own documents.
- WebAssembly `ApproxIndex` gains `upsert` and a per-query beam width, closing
  the last two gaps against the other bindings. `upsert` is one operation where
  `remove` then `add` is two, and those two can fail between the halves and
  leave the id deleted; a rejected replacement keeps the existing vector. The
  beam width applies to a single `search` and leaves `ef_search` alone, so
  rescuing one hard query no longer charges every later one. `is_empty` is
  deliberately not ported (#182): Python already says `len(index)` or plain
  truth testing, JavaScript says `index.size() === 0`, and it is absent from
  the C ABI and the C++ engine too, so adding it to two of five surfaces would
  buy no uniformity.
- Python `Metric` pickles and hashes, so a worker pool can be handed one and a
  dict can be keyed by one. The index types still refuse to pickle, at `dumps`
  rather than at `loads`: bytes no unpickler will accept are worse than an
  error. Save an `ApproxIndex`, build a file with `DiskIndexBuilder`, or
  rebuild a `FlatIndex` from its source vectors instead.

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
