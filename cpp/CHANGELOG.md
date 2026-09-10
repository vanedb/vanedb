# Changelog

All notable changes to VaneDB will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-09-10

### Changed
- **Legacy HNSW load now caps the size a header may declare.** `load()`
  pre-allocated `max_elements * dimension` before reading any payload, with
  overflow as its only bound, so an 80-byte header could request terabytes.
  Both `max_elements` and that product are now capped at `MAX_VEC_SIZE`
  (100,000,000), matching the bound Rust and the VNDB v2 reader already apply.

  This qualifies the backward-compatibility note below. v1/v2 files are
  unaffected — `read_vec` already refused arrays above the cap, and the loader
  requires the array it read to equal the product. A **v3** file stores only
  `count * dimension` and is re-expanded to capacity afterwards, so a sparse v3
  file whose `capacity * dimension` exceeds the cap loaded before and does not
  now: 200,000 slots at dimension 768 is enough. `save` has written VNDB v2
  since before any release and this package has never been published, so such a
  file can only come from a self-built tree. Re-save it with an older build to
  get a VNDB v2 file, which has no such limit.

  The cap is a mitigation, not an elimination: at it, the constructor still
  reserves roughly 4 GB from an 80-byte header.
- The supplementary `vanedb-cpp` distribution is reference-only. CI builds
  and tests its wheels across supported hosts and Python versions; the package
  is not published. The Rust-backed `vanedb` package is the supported product.
- **BREAKING: Project renamed from QuiverDB to VaneDB.** Pre-1.0, so no
  on-disk break — HNSW index files written with the old name still load
  (the `0x51565244` "QVRD" magic is retained for backward compat). What
  *did* change for downstream consumers:
  - C++ namespace: `quiverdb::` → `vanedb::`
  - Python module: `import quiverdb_py` → `import vanedb_cpp`
  - CMake options: `QUIVERDB_BUILD_*` → `VANEDB_BUILD_*`
  - Preprocessor macros injected by callers (e.g. `-DQUIVER_CUDA_ENABLED`)
    → `-DVANE_CUDA_ENABLED`
  - Logging macros (`QUIVERDB_LOG_*` / `QUIVERDB_LOG_LEVEL_*`) → `VANEDB_LOG_*`
  - Repository: `github.com/tsvet01/quiverdb` → `github.com/vanedb/vanedb`
    (GitHub redirects the old URL).

### Added
- Supplementary Python distribution: `vanedb-cpp`, imported as `vanedb_cpp`.
  Built and tested in CI, never published (#100). The Rust bindings remain the
  canonical `vanedb` distribution.
- Comprehensive corruption detection tests for file format validation
  - Invalid magic number, version, metric detection
  - Size overflow protection tests (SIZE_MAX scenarios)
  - Truncated file handling tests
  - Input validation tests (null pointers, invalid parameters)
  - Zero dimension with vectors detection test
  - Combined dim*num_vectors overflow test

### Fixed
- **Pickling a `Metric` aborted the interpreter.** pybind11 gives an enum
  `__getstate__`, which serves protocol 2 and up; protocols 0 and 1
  reconstruct through `copyreg._reconstructor`, which calls `object.__new__`
  on a pybind11 type. That throws a C++ exception with no Python translation,
  so the process died on SIGABRT instead of raising. `Metric` now reduces
  through its constructor, giving every protocol one path. It reduces by
  *value* rather than by name because pybind11 lets an enum hold an integer no
  value names — `Metric(7)` reprs as `<Metric.???: 7>` — and a name-based
  reduce has to pick a named fallback for those. That is worse than lossy:
  `FlatIndex(3, Metric(7))` rejects the invalid value, so by-name turned an
  inert-but-invalid metric into a *valid* `L2` the engine accepts and computes
  wrong distances with.

  A pickled `Metric` is now equal to, but not the same object as, the class
  member, because reconstruction calls the constructor. `Metric(1) is
  Metric.COSINE` was already false, so `is` was never sound on this type — but
  note the `vanedb` package reduces by name and does preserve identity.
  Compare metrics with `==`, which both honour.

  The test runs out-of-process, because an in-process one would have taken the
  whole session down with it, which is why nothing caught the abort. What gives
  it teeth is the unnamed `Metric(7)` case: the three named values round-trip
  correctly even under the buggy implementation.
- A persisted negative level multiplier made the next insertion fail. The
  loader now retains the multiplier derived from `M`, matching Rust, and a
  regression covers negative, infinite and NaN stored values. Recorded here
  because it was previously only in a superseded release record.
- `DiskIndex` accepted a file whose header understated its geometry. The
  expected length is computed from the header, so `file_size_ < expected` could
  not catch a header claiming a smaller `dim` or `count` than the file holds;
  the payload was then read at the wrong stride and `get` returned a vector
  straddling two stored records. Now requires exact equality, matching the Rust
  reader and the VNDB v1 spec. A defect fix in the engine, not a feature port:
  both readers of a shared format must reject the same files, and
  `bench/tests/cross_engine_format.rs` now asserts that over a bit sweep.
- Windows file locking issue in mmap tests (scope store before file removal)
- Type consistency in test file format (uint64_t for dimension field)
- Coverage reporting now excludes test and benchmark files (measures only production code)
- Division by zero in DiskIndex when loading corrupted file with dim=0

## Earlier development

Nothing here was published: these entries record how the engine was built
before 0.1.1, which is the first release of any package. The C++ manifest,
header and CMake version track the core crate because RELEASING.md requires
them to match, not because this package is published — it is not.

### Added
- Core distance functions with SIMD optimization (ARM NEON, x86 AVX2)
  - L2 squared distance
  - Cosine similarity/distance
  - Dot product
- GPU acceleration: Metal (Apple Silicon). CUDA is not included — see the
  roadmap; kernel source in the tree is not wired into the build and is not
  a supported backend
  - Persistent buffer API for zero-copy repeated queries
- In-memory FlatIndex with k-NN brute-force search
- HNSW index for approximate nearest neighbor search
  - Configurable M, ef_construction, ef_search parameters
  - Binary serialization (save/load)
- `DiskIndex`, a memory-mapped store for large datasets
  - Zero-copy file access
  - Atomic save operations
- Python bindings via pybind11
  - NumPy array support
  - All index types and distance metrics
- C++ and Python test suites, run by `ctest` and pytest
- Google Benchmark performance tests
- Multi-platform CI/CD (Linux, macOS, Windows, iOS, Android)
  - GCC, Clang, MSVC compilers
  - Native ARM64 testing
  - iOS arm64 builds (Xcode)
  - Android arm64-v8a and x86_64 builds (NDK)
  - AddressSanitizer and UBSan checks
  - Code coverage reporting

### Performance
- SIMD distance kernels (NEON, AVX2, scalar reference) with runtime dispatch.
  Figures are deliberately not quoted here: they were undated and unsourced,
  and this repo treats a benchmark as meaningful only on dedicated hardware
  with interleaved A-B-A runs. See [`bench/README.md`](../bench/README.md),
  which dates its snapshot and names the machine.
