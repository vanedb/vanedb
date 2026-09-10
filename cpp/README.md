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

This frozen C++20 engine is reference code and the other arm of VaneDB's
[cross-engine benchmark](../bench). The [Rust engine](..) is the product that
ships. C++ source builds and tests are maintained; features are not ported and
the `vanedb-cpp` Python package is not published.

## Features

- **SIMD-optimized**: ARM NEON, x86 AVX2 (figures in [`bench/README.md`](../bench/README.md))
- **Multiple indexes**: Brute-force, HNSW, Memory-mapped
- **GPU acceleration**: Metal (Apple Silicon). CUDA (NVIDIA) is experimental — kernel source only, not yet wired into the build
- **Thread-safe**: Concurrent reads with `std::shared_mutex`
- **Python bindings**: NumPy integration, GIL-safe

## Quick Start

```cpp
#include "core/flat_index.h"

vanedb::FlatIndex store(768, vanedb::Metric::COSINE);
store.add(1, embedding);
auto results = store.search(query, 5);  // top-5 nearest neighbors
```

```cpp
#include "core/approx_index.h"

vanedb::ApproxIndex index(768, vanedb::Metric::COSINE, 100000);
index.add(1, embedding);
auto results = index.search(query, 5);
index.save("index.bin");
```

```python
# Reference bindings, built locally with `python -m pip install ./cpp`.
import vanedb_cpp as vanedb
import numpy as np

index = vanedb.ApproxIndex(768, vanedb.Metric.COSINE)
index.add(1, np.random.rand(768).astype(np.float32))
ids, distances = index.search(query, 10)
```

Local Python builds dispatch between scalar, AVX2/FMA, and NEON kernels.
Header-only C++ consumers specialize at compile time: binaries built with
`-mavx2 -mfma` or `/arch:AVX2` require a matching CPU.

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
