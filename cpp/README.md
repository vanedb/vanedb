<div align="center">

# VaneDB

**Embeddable vector database for edge AI**

[![Build](https://github.com/vanedb/vanedb/actions/workflows/ci.yml/badge.svg)](https://github.com/vanedb/vanedb/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/vanedb/vanedb/branch/main/graph/badge.svg)](https://codecov.io/gh/vanedb/vanedb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C.svg)](https://en.cppreference.com/w/cpp/20)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB.svg)](https://www.python.org/)

</div>

---

Header-only C++20 vector database with SIMD acceleration. Runs on Linux, macOS, Windows, iOS, and Android.

> **Two implementations, one repository.** This directory is VaneDB's
> **C++ header-only** implementation — drop a header into any CMake project,
> no Rust toolchain needed. The canonical Rust, Python, and WASM implementation
> lives at the [repository root](..).

## Why VaneDB?

| Feature | VaneDB | FAISS | hnswlib | Pinecone |
|---------|----------|-------|---------|----------|
| Header-only | Yes | No | No | N/A |
| Mobile/Edge | Native | No | Partial | No |
| Dependencies | Zero | Many | Few | Cloud |
| Binary size | <100KB | 200MB+ | ~1MB | N/A |
| GPU (Metal) | Yes | No | No | N/A |

**Perfect for**: Mobile AI apps, Obsidian/Logseq plugins, edge devices, offline-first applications.

## Features

- **SIMD-optimized**: ARM NEON, x86 AVX2 (~100ns for 768d vectors)
- **Multiple indexes**: Brute-force, HNSW, Memory-mapped
- **GPU acceleration**: Metal (Apple Silicon). CUDA (NVIDIA) is experimental — kernel source only, not yet wired into the build
- **Thread-safe**: Concurrent reads with `std::shared_mutex`
- **Python bindings**: NumPy integration, GIL-safe

## Quick Start

```cpp
#include "core/vector_store.h"

vanedb::VectorStore store(768, vanedb::DistanceMetric::COSINE);
store.add(1, embedding);
auto results = store.search(query, 5);  // top-5 nearest neighbors
```

```cpp
#include "core/hnsw_index.h"

vanedb::HNSWIndex index(768, vanedb::DistanceMetric::COSINE, 100000);
index.add(1, embedding);
auto results = index.search(query, 5);
index.save("index.bin");
```

```python
# Supplementary C++ bindings; the canonical Python package is `vanedb`.
# python -m pip install vanedb-cpp
import vanedb_cpp as vanedb
import numpy as np

index = vanedb.HNSWIndex(768, vanedb.DistanceMetric.COSINE)
index.add(1, np.random.rand(768).astype(np.float32))
ids, distances = index.search(query, 10)
```

### Python wheel CPU support

Generic x86-64 wheels target the baseline x86-64 instruction set (SSE2), not
AVX2. The extension and its dispatcher are compiled for that baseline;
AVX2/FMA distance kernels live in a separate object with cross-object LTO
disabled. They are selected only when CPUID advertises AVX2 and FMA and the OS
has enabled saving both XMM and YMM register state. Other x86-64 CPUs use the
scalar kernel; arm64 wheels use NEON. Python and NumPy distributions have their
own CPU requirements: the complete application must also meet those. CI checks
the module's baseline import separately and uses no-AVX Nehalem for full NumPy
search tests, not a claim that every NumPy wheel works on an SSE2-only CPU.

```python
import vanedb_cpp
print(vanedb_cpp.simd_backend())  # "scalar", "avx2_fma", or "neon"
```

Both PR CI and the release artifact gate install the already-built Linux wheel
and execute its Python tests under QEMU with baseline, AVX-only, AVX2-without-FMA,
and AVX2/FMA CPU profiles. CPU compatibility does not override the Python/OS
requirements encoded in wheel tags. Publishing still requires explicit approval.

Header-only C++ consumers and the C API benchmark harness retain their existing
compile-time specialization/inlining. When building those with `-mavx2 -mfma`
or `/arch:AVX2`, the resulting binary still requires a suitable CPU; the portable
runtime dispatcher described above is enabled for the Python distribution.

## Build

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build --parallel
ctest --test-dir cpp/build --output-on-failure
```

Run these commands from the repository root.

## Documentation

- [Full API Guide](docs/GUIDE.md) - Detailed usage, Python bindings, mobile builds
- [CHANGELOG](CHANGELOG.md) - Version history

## License

MIT License - see [LICENSE](LICENSE) for details.
