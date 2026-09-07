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
- The `@claude` workflow restricts invocation to repository owners, members
  and collaborators, using GitHub's author associations.
- GitHub Actions are pinned to commits; Rust CI checks dependency advisories,
  licences, bans and sources with `cargo-deny`.

### Added

- `DiskIndex` and `DiskIndexBuilder` appear in the published documentation,
  with the feature badge that says they need `disk`.
- Declared MSRV of 1.85, checked in CI on that exact toolchain.
- The generated C header is rebuilt in CI and must match what is committed;
  `build.rs` fails loudly instead of leaving a stale header in place.

### Changed

- `DiskIndex` uses VNDB v1. New `ApproxIndex::save` files use the shared VNDB v2
  graph format; legacy HNSW files remain readable. The graph format is a release
  candidate without a public 1.0.0 compatibility promise yet. Keep originals
  and source vectors while verifying migration.
- Python methods that block — `save`, `load`, `upsert`, `remove`, and the
  `DiskIndexBuilder` methods — release the GIL. `DiskIndexBuilder` takes
  `&self` like every other class and can be shared between threads.
  Accessors that wait on index locks also release the GIL while waiting.
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
- Optional Metal compute validates sizes, finite inputs, ID counts and device
  resources; tiny-value calculations use CPU kernels to preserve rankings.

### Roadmap

- NVIDIA CUDA support is required after 1.0.0. The unimplemented Rust CUDA
  feature was removed; implementation and hardware acceptance requirements
  are recorded in the [roadmap](docs/ROADMAP.md).
