# vanedb-cpp

## Overview
Embeddable vector database for edge AI. Header-only C++20, SIMD-optimized, cross-platform.

## Role in the VaneDB Project

This is the **C++ header-only** implementation in the `cpp/` directory. The
Rust implementation lives at the repository root and owns the Rust, Python,
C and WASM product. See the root README for source-build instructions.
The C++ Python bindings are built locally for reference and testing; there is
no `vanedb-cpp` publication workflow.

**Why this engine still exists:** it is the other arm of the cross-engine
benchmark, and a header you can drop into a CMake/Bazel project with no Rust
toolchain. It is not a second product, and not a path a new user should be
steered down.

**Alignment policy:** none — features are not synced. Shared regression
vectors and disk fixtures live in `../vanedb/tests/fixtures/conformance/`, with
their contract in `../conformance/`, and those must keep
passing, but that is a constraint on changes rather than a sync obligation.

## Current Status: v0.1.1, frozen

### Features
| Feature | Status |
|---------|--------|
| Distance Functions | L2, Cosine, Dot (ARM NEON, x86 AVX2) |
| FlatIndex | In-memory k-NN, thread-safe |
| ApproxIndex | Approximate NN with save/load |
| DiskIndex | Memory-mapped zero-copy |
| GPU Acceleration | Metal (Apple Silicon). CUDA experimental — unwired, untested |
| Python Bindings | pybind11 + NumPy |
| Mobile | iOS arm64, Android arm64-v8a/x86_64 |

### Test Coverage
- `ctest --test-dir cpp/build` is the count that matters; it is not recorded
  here because every number written down here has gone stale within days.
- Python bindings are covered by `cpp/tests/test_python_bindings.py`.
- Sanitizers: ASan + UBSan clean
- **Not covered:** the Metal path. `test_metal_distance` is built only under
  `-DVANEDB_BUILD_METAL=ON` (default OFF), is never registered with ctest
  because it needs a real Metal device, and so runs in no CI job. Treat it as
  a manual check, not coverage.

### Performance
CPU figures: [`bench/README.md`](../bench/README.md), which dates its snapshot
and names the machine. There is no published Metal figure, and no CI job
measures one.

## Structure
Read `cpp/src/core/`.

## Build
```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build --parallel
ctest --test-dir cpp/build --output-on-failure
```

Run these commands from the repository root.

## CI/CD (10 jobs)
- Linux: GCC, Clang
- macOS: Apple Clang (ARM NEON)
- Windows: MSVC (AVX2)
- ARM64: Native runners
- iOS: Xcode arm64
- Android: NDK arm64-v8a, x86_64
- Python: 3 platforms x Python 3.11–3.14
- Sanitizers: ASan, UBSan
- Coverage: Codecov

## API
`cpp/src/core/{flat_index,approx_index,disk_index}.h` and `gpu/` declare it.
`cpp/README.md` carries usage examples; nothing checks them against the
headers, so read the headers when it matters.


## Maintenance Posture

**This engine is frozen.** The root `AGENTS.md` is authoritative; this section
restates it rather than widening it, because the two disagreed before and an
agent reading only this file drew the wrong conclusion.

Change it only to:
- Fix a defect in the engine itself, including CVEs
- Keep it building and its own tests passing
- Keep shared VNDB disk and graph files loadable both ways, preserving topology
  and loaded tombstones without adding a public delete API
- Keep the benchmark comparison honest — where a Rust change makes a row
  measure something different, annotate the row rather than porting to match

Do **not** port features. `add_batch`, growable capacity and delete are
Rust-only by decision, not by omission. Performance work and new platform
support are not reasons to touch this engine either: an optimisation here
buys nothing that ships, and changes the comparison it exists to provide.

The canonical PyPI distribution (`pip install vanedb`), WASM, and
batch/metadata APIs are owned by the Rust component and tracked in this
repository's issue tracker.

## Known Limitations
- No deletion in ApproxIndex (rebuild required)
- GPU requires dim % 4 == 0
- FlatIndex `get()` pointer invalidated by writes
- Single-file persistence (no sharding)
