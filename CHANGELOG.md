# Changelog

Notable changes to the `vanedb` crate and the `vanedb` Python package. The
frozen C++ engine has its own log in [`cpp/CHANGELOG.md`](cpp/CHANGELOG.md).

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). This
project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html); until
1.0.0, breaking changes may land in a minor release.

## [Unreleased]

Nothing has been published yet, so there is no released version to diff
against. This section records what the first release will contain.

### Security

- SIMD distance kernels took their loop bound from one slice while the scalar
  reference truncated to the shorter, so mismatched lengths read past the end
  of the shorter one. The kernels are safe `pub fn` in a public module, making
  this reachable from safe code. Every kernel is now bounded by both lengths.
- `ApproxIndex::load` multiplied file-controlled values without checking, so a
  497-byte file could wrap the product to zero and load with absurd
  parameters, or panic on a path documented to return an error.
- `DiskIndex::open` accepted files with duplicate ids, after which `size`
  disagreed with lookups, `get` returned a row the id did not name, and a
  single search could return one id twice. Both engines now reject them.
- The `@claude` workflow ran a job holding an OAuth token for any comment
  containing the mention, including from forks. It now requires the commenter
  to be an owner, member or collaborator.
- Every GitHub Action is pinned to a commit; `cargo-deny` gates advisories,
  licences, bans and sources on each change.

### Changed — public API, before anything is published

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
- `FlatIndex::size`, `DiskIndex::len`/`is_empty`, `DiskIndexBuilder::len`/
  `is_empty` and `ApproxIndex::get` added, so the count and read spellings
  match across index types.

### Added

- `DiskIndex` and `DiskIndexBuilder` appear in the published documentation,
  with the feature badge that says they need `disk`.
- Declared MSRV of 1.85, checked in CI on that exact toolchain.
- The generated C header is rebuilt in CI and must match what is committed;
  `build.rs` fails loudly instead of leaving a stale header in place.
- A committed `HNSW` v1 file, loaded as bytes, so a change to the encoder
  cannot silently stop older files from loading.
- `DiskIndex::open` documents that the mapped file must not change while open,
  on the Rust, C and Python surfaces.

### Changed

- The persistence promise is scoped to what is true: `DiskIndex` writes `VNDB`
  v1, a specified, versioned, cross-engine format; `ApproxIndex::save` is
  engine-specific and not yet a public format.
- `Metric::Dot` documents that it is not scale-invariant and not a metric, so
  a vector need not be its own nearest neighbour.
- The published recall figure is annotated as measured on uniform-random
  vectors, the adversarial case for a proximity graph.

### Fixed

- Copyright notices name a person rather than the project name, and the Python
  wheel ships the MIT text rather than a link to it.
- The PyPI description no longer opens on a sentence broken by a rename.
- The README no longer instructs JavaScript users to pass `Metric.COSINE`,
  which the wasm bindings reject.
