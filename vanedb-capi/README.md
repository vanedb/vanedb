# VaneDB from C

Embed the Rust engine in a C or C++ application. You need a C compiler and
CMake 3.20 or newer for the example, plus Rust when building from source.

CI produces platform archives named `vanedb-capi-<version>-<platform>.zip`,
with a SHA-256 checksum. Choose the archive matching your OS and architecture
from a successful [CI run](https://github.com/vanedb/vanedb/actions/workflows/ci.yml).
These are development artifacts until a release is tagged. Check the archive's
`compatibility.json` for binary requirements, imported libraries and the OS used
for consumer acceptance:

| Archive | Binary requirements |
|---|---|
| Linux x86-64 / ARM64 | glibc, not musl; current libraries require glibc 2.34 and system libraries listed in the metadata |
| macOS Intel | Declared deployment target 10.12; system dependencies listed in the metadata |
| macOS ARM64 | Declared deployment target 11.0; system dependencies listed in the metadata |
| Windows x64 | Imported DLLs listed in the metadata, including VCRUNTIME140 and Universal CRT; a minimum Windows release has not been verified |

CI rejects an increased glibc requirement or a changed macOS deployment target.
These are inspected binary requirements, not proof of runtime support on the
oldest OS. Consumer acceptance runs on the recorded CI host. In particular, the
Windows PE subsystem version does not establish the application's minimum OS.
Build from source and verify on your deployment if it differs from those hosts.

Extract the archive,
then run these commands inside its top-level directory:

```sh
cmake -S examples -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build --build-config Release --output-on-failure
```

The archive includes `include/`, `lib/`, complete examples and the license.
CI tests a consumer against each extracted archive before uploading it.

To build from source, run from the repository root:

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
searches return the number of results written, where zero alone cannot tell a
failure from an empty store. After any call, `vanedb_rs_last_error()` gives the
reason as a `VANEDB_RS_*` code and `vanedb_rs_last_error_message()` the detail —
both thread-local, and both reset by the next call on that thread — except
the `*_free` functions, which deliberately preserve them so the ordinary C
path of fail, clean up, then report does not lose the reason. The message
pointer is also freed when its thread exits. Branch on the
code rather than the return value: `VANEDB_RS_IO` is worth retrying,
`VANEDB_RS_CORRUPT` is not, and `VANEDB_RS_FILE_NOT_FOUND` on a load means
build the file instead — while on a save it means the destination directory
does not exist. Treat an unrecognized code as a failure; the set grows in minor
releases.

Allocate output buffers for at least `k` IDs and distances. Free each handle
once with its matching `*_free` function. The generated header documents
pointer validity, ownership and buffer requirements for every function.

`vanedb_rs_index_save_to_buffer` / `vanedb_rs_index_load_from_buffer` write and
read the same VNDB file as the path functions, without a filesystem. Passing
a null buffer and a zero capacity queries the required size; a short buffer
fails and still reports that size so the caller can allocate and retry.
[`examples/ctypes_quickstart.py`](examples/ctypes_quickstart.py) shows the
round trip.

The graph search's `ef_search` argument applies only to that call, so concurrent
queries can choose different recall/speed settings. Pass `0` to search at the
handle's own setting, which `vanedb_rs_index_ef_search()` reports. Note that
`vanedb_cpp_index_search` rejects `0` rather than resolving it — the two ABIs
are otherwise callable through one uniform FFI.

### Filtered search

`vanedb_rs_store_search_filtered`, `vanedb_rs_index_search_filtered` and
`vanedb_rs_disk_search_filtered` take the same query, `k` and output buffers
as the plain searches, plus at most one filter: a callback with its
`user_data`, an allow list, or a deny list. A callback combined with a list,
or both lists together, fails with `VANEDB_RS_INVALID_PARAMETER`. With a null
callback and null list pointers the call is an ordinary unfiltered search.

A list is selected by pointer presence, not by length. A non-null pointer with
a length of zero is an empty list: an empty allow list matches nothing and an
empty deny list matches everything. A null pointer must come with a zero
length; null with a nonzero length fails with `VANEDB_RS_NULL_ARGUMENT`. Lists
must be sorted strictly ascending with no duplicates — an unsorted or
duplicated id fails with `VANEDB_RS_INVALID_PARAMETER`, as does a length that
cannot fit in the platform's address space. Every rejected call returns zero
and leaves `out_ids` and `out_dists` untouched, so as with the other searches,
branch on `vanedb_rs_last_error()` to tell a rejected call from a search that
found no matches.

A callback has the type

```c
typedef bool (*vanedb_rs_filter_fn)(uint64_t id, void *user_data);
```

and returns true to accept an id. It runs synchronously on the calling thread
while the searched handle's read lock is held, and the graph search may call
it more than once for the same id as it widens its beam. It must not access,
modify or free the handle being searched — same-index re-entrancy under that
lock — and must not modify or free the search buffers. Calls on other handles
are allowed; an error such a call records does not replace the outer search's
result. The callback and everything `user_data` points to must stay valid
until the search returns. No foreign exception or `longjmp` may cross the
callback. A Rust callback declared `extern "C-unwind"` may panic: the panic is
caught at the boundary, the search returns zero with `VANEDB_RS_PANIC`, the
result buffers are untouched, and the handle remains usable afterwards. ID
lists are the fast path because they never cross the language boundary per
candidate.

`vanedb_rs_index_search_filtered` takes `ef_search` under the same rule as the
plain graph search: `0` uses the handle's setting. The C ABI omits the beam
cap (`max_ef_search` in Rust, Python and WebAssembly) by design: a filtered
graph search widens its beam up to the core's default of four times the
initial beam, capped at the stored slot count. Raise `ef_search` on the call
to improve recall under a selective filter — on the order of `k` divided by
the fraction of ids the filter accepts — since the cap alone does not improve
results that already fill `k`. Measured recall at several selectivities is in
[the 0.2.0 validation record](https://github.com/vanedb/vanedb/blob/main/docs/release/0.2.0-filtered-search-validation.md#recall-on-real-embeddings).

### Calling from Python with ctypes

Declare `restype` and `argtypes` for **every** function before calling it.
ctypes defaults an undeclared return to a C `int`, which truncates a 64-bit
pointer to 32 bits; the next dereference reads a garbage address and the
process dies with SIGSEGV and no Python traceback, because the fault happens
inside libffi. A crash whose stack reads `PyCFuncPtr_call` →
`_ctypes_callproc` → `ffi_call` is almost always a missing `restype` rather
than a fault in this library. Undeclared pointer *arguments* truncate the same
way on the way in.

Use `c_void_p` rather than `c_char_p` for `vanedb_rs_last_error_message`:
`c_char_p` copies the bytes into a Python object at the boundary, which is
convenient but conceals the lifetime — the pointer is valid only until the next
`vanedb_rs_*` call on that thread, or until the thread exits.

[`examples/ctypes_quickstart.py`](https://github.com/vanedb/vanedb/blob/main/vanedb-capi/examples/ctypes_quickstart.py)
is a complete
working consumer; CI runs it against a built library on every change.

`vanedb_rs_version()` returns the library's version so a consumer can check the
shared object matches the header it compiled against. This is a 0.x ABI: it may
change in a minor release. See the
[repository guide](https://github.com/vanedb/vanedb#persistence) for persistence
limitations and platform verification.
