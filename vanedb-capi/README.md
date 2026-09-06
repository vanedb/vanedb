# VaneDB from C

Embed the Rust engine in a C or C++ application. Requires a Rust toolchain,
C compiler and CMake 3.20 or newer to build from this repository.

From the repository root:

```sh
cargo build -p vanedb-capi --release --locked
cmake -S vanedb-capi/examples -B target/c-example -DCMAKE_BUILD_TYPE=Release
cmake --build target/c-example --config Release
ctest --test-dir target/c-example --build-config Release --output-on-failure
```

The [complete example](examples/quickstart.c) inserts two vectors, searches for
the nearest one, checks its ID and distance, and frees its owning handle. Its
[CMake project](examples/CMakeLists.txt) links the shared library on Linux,
macOS and Windows; use `VANEDB_LIBRARY_DIR` for a different artifact directory.

Include [vanedb_rs_capi.h](include/vanedb_rs_capi.h) in your application and
link the matching `vanedb_capi` library from `target/release`. Keep the shared
library available to the operating system's library loader when distributing
your application.

Use `vanedb_rs_store_*` for exact in-memory search, `vanedb_rs_index_*` for an
approximate graph, and `vanedb_rs_disk_*` for a read-only mapped file. Metrics
are `VANEDB_RS_L2` (squared distance), `VANEDB_RS_COSINE` and `VANEDB_RS_DOT`
(negative dot product). Lower distances rank first. Supply finite vectors with
the configured dimension and unique unsigned 64-bit IDs.

Constructors return null on failure. Status functions return zero on success;
searches return the number of results written, with zero also representing an
error. Allocate output buffers for at least `k` IDs and distances. Free each
handle once with its matching `*_free` function. The generated header documents
pointer validity, ownership and buffer requirements for every function.

The graph search's `ef_search` argument applies only to that call, so concurrent
queries can choose different recall/speed settings. See the
[repository guide](..) for persistence limitations and platform verification.
