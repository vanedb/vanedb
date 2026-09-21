# VaneDB from C

Embed the Rust engine in a C or C++ application. You need a C compiler and
CMake 3.20 or newer (or pkg-config and make), plus Rust when building from
source.

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
Support tiers and floors for every platform are in
[`docs/PLATFORMS.md`](../docs/PLATFORMS.md), which also records that the
macOS Intel archive ends when GitHub's last Intel runner retires in August 2027.
These are inspected binary requirements, not proof of runtime support on the
oldest OS. Consumer acceptance runs on the recorded CI host. In particular, the
Windows PE subsystem version does not establish the application's minimum OS.
Build from source and verify on your deployment if it differs from those hosts.

## What the archive contains

```
include/vanedb_rs_capi.h        the header; VANEDB_RS_ABI_VERSION and VANEDB_RS_VERSION
lib/libvanedb_capi.{so,dylib}   shared library, stripped, exports only vanedb_rs_*
lib/libvanedb_capi.a            static library (vanedb_capi.lib and vanedb_capi.dll.lib on Windows)
lib/cmake/vanedb/               find_package(vanedb): targets vanedb::shared and vanedb::static
lib/pkgconfig/vanedb.pc         pkg-config --cflags --libs vanedb
examples/                       quickstart.c, its CMake project, the ctypes example
consumers/cmake, consumers/pkgconfig   the consumer projects CI runs from this layout
tests/acceptance.c              the acceptance program every consumer builds
compatibility.json              inspected requirements and the static link line
```

Both package files are relocatable: they resolve every path relative to
themselves, so the extracted directory can live anywhere.

### CMake

```cmake
find_package(vanedb REQUIRED CONFIG)   # -DCMAKE_PREFIX_PATH=/path/to/vanedb-capi-<version>-<platform>
target_link_libraries(app PRIVATE vanedb::shared)   # or vanedb::static
```

`vanedb::static` carries the system libraries the Rust standard library needs
in its `INTERFACE_LINK_LIBRARIES`, recorded at package time from
`cargo rustc --print native-static-libs` (`compatibility.json` lists them too).
`vanedbConfigVersion.cmake` treats a 0.x request as compatible only with the
same major and minor. Run the consumer CI runs, from the extracted archive:

```sh
cmake -S consumers/cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$PWD"
cmake --build build --config Release
ctest --test-dir build --build-config Release --output-on-failure
```

### pkg-config

```sh
PKG_CONFIG_PATH="$PWD/lib/pkgconfig" pkg-config --cflags --libs vanedb
PKG_CONFIG_PATH="$PWD/lib/pkgconfig" make -C consumers/pkgconfig test
```

`Libs` links the shared library; `Libs.private` (shown by `--static`) is the
static link line. Not available on Windows.

### Without either

Run the examples from the top-level directory of the archive:

```sh
cmake -S examples -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build --build-config Release --output-on-failure
```

CI tests every consumer above against each extracted archive before uploading it.

## Building from source

The shipped libraries are built under the `capi` Cargo profile (fat LTO, one
codegen unit, `opt-level = 3`, `panic = "unwind"` so the panic boundary keeps
working). Run from the repository root:

```sh
cargo build -p vanedb-capi --profile capi --locked      # target/capi/
cmake -S vanedb-capi/examples -B target/c-example -DCMAKE_BUILD_TYPE=Release
cmake --build target/c-example --config Release
ctest --test-dir target/c-example --build-config Release --output-on-failure
```

A plain `--release` build still works and lands in `target/release/`; the
examples look for `target/capi` first. `scripts/package_capi.py --platform
<platform>` produces the archive above from the `capi` build and runs every
consumer against it.

Shared-library size on the machine the profile was introduced on (Linux
x86-64, GNU ld 2.42, Rust 1.94.1), in bytes, from a local build. These are
file sizes, not performance claims:

| Build | `libvanedb_capi.so` |
|---|---|
| `--release` (before) | 749,120 |
| `--profile capi`, unstripped | 719,704 |
| `--profile capi`, stripped as packaged | 594,368 |

The static library is packaged unstripped.

## Using the ABI

The [complete example](examples/quickstart.c) inserts two vectors, searches for
the nearest one, checks its ID and distance, and frees its handle. Its
[CMake project](examples/CMakeLists.txt) links the shared library on Linux,
macOS and Windows; use `VANEDB_LIBRARY_DIR` for a different artifact directory.

Include [vanedb_rs_capi.h](include/vanedb_rs_capi.h) in your application and
link the matching `vanedb_capi` library. Keep the shared library available to
the operating system's library loader when distributing your application.

**Check the ABI first.** `VANEDB_RS_ABI_VERSION` is the integer ABI version of
the header and `vanedb_rs_abi_version()` the loaded library's; they must
agree before anything else is called. It is bumped only on an incompatible
change. The rule that avoids one, written into the header: a declared
signature never changes, new behaviour arrives as a new `_ex` or `_v2`
function, no struct crosses the boundary (and if one ever does, its first
field is `size_t size`), and error codes are only added. `VANEDB_RS_VERSION`
and `vanedb_rs_version()` carry the semver string alongside.

**Handles are 64-bit ids, not pointers.** Every `vanedb_rs_store`,
`vanedb_rs_index` and `vanedb_rs_disk` is a `uint64_t` into a table owned by
the library. An id that was never issued, was freed, was truncated on the way
through an FFI that guessed its type, or belongs to another handle type fails
the call with `VANEDB_RS_INVALID_HANDLE`; nothing is dereferenced and a freed
id is never reissued, so a double free or use after free is a reported error.
`VANEDB_RS_NULL_HANDLE` (0) is what a failed constructor returns: passing it
is `VANEDB_RS_NULL_ARGUMENT` and freeing it is a no-op, so the usual C
cleanup idiom holds. `vanedb_rs_handle_count()` reports the live handles for
a leak test. Each call does one lookup under a sharded lock and then runs on
its own reference, so a search never holds the lock and freeing a handle
another thread is using is safe (that call completes; later calls fail).

Use `vanedb_rs_store_*` for exact in-memory search, `vanedb_rs_index_*` for an
approximate graph, and `vanedb_rs_disk_*` for a read-only mapped file. Metrics
are `VANEDB_RS_L2` (squared distance), `VANEDB_RS_COSINE` and `VANEDB_RS_DOT`
(negative dot product). Lower distances rank first. Supply finite vectors with
the configured dimension and unique unsigned 64-bit IDs.

Constructors return `VANEDB_RS_NULL_HANDLE` on failure. Status functions
return zero on success; searches return the number of results written, where
zero alone cannot tell a failure from an empty store. After any call,
`vanedb_rs_last_error()` gives the reason as a `VANEDB_RS_*` code and
`vanedb_rs_last_error_message()` the detail — both thread-local, and both
reset by the next call on that thread — except the `*_free` functions, which
deliberately preserve them on success so the ordinary C path of fail, clean
up, then report does not lose the reason. The message pointer is also freed
when its thread exits. Branch on the code rather than the return value:
`VANEDB_RS_IO` is worth retrying, `VANEDB_RS_CORRUPT` is not, and
`VANEDB_RS_FILE_NOT_FOUND` on a load means build the file instead — while on a
save it means the destination directory does not exist. Treat an unrecognized
code as a failure; the set grows in minor releases.

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
are otherwise callable through one uniform FFI, except that the C++ engine's
handles remain pointers.

### Calling from Python with ctypes

Declare `restype` and `argtypes` for **every** function before calling it.
ctypes defaults an undeclared return to a C `int`, which truncates a 64-bit
value to 32 bits. For a handle that is no longer a crash — the library refuses
the truncated id with `VANEDB_RS_INVALID_HANDLE` — but every call made with it
fails. `vanedb_rs_last_error_message` still returns a pointer, and a
truncated pointer still dies with SIGSEGV inside libffi, with no Python
traceback. Bind handles as `ctypes.c_uint64`, never `c_void_p` (which turns 0
into `None` and truncates on a 32-bit Python):

```python
import ctypes
lib = ctypes.CDLL("libvanedb_capi.so")
HANDLE, usize, f32p = ctypes.c_uint64, ctypes.c_size_t, ctypes.POINTER(ctypes.c_float)
lib.vanedb_rs_abi_version.restype = ctypes.c_uint32
lib.vanedb_rs_abi_version.argtypes = []
lib.vanedb_rs_store_new.restype = HANDLE
lib.vanedb_rs_store_new.argtypes = [usize, ctypes.c_uint32]
lib.vanedb_rs_store_add.restype = ctypes.c_int32
lib.vanedb_rs_store_add.argtypes = [HANDLE, ctypes.c_uint64, f32p]
lib.vanedb_rs_store_search.restype = usize
lib.vanedb_rs_store_search.argtypes = [HANDLE, f32p, usize, ctypes.POINTER(ctypes.c_uint64), f32p]
lib.vanedb_rs_store_free.restype = None
lib.vanedb_rs_store_free.argtypes = [HANDLE]
lib.vanedb_rs_last_error.restype = ctypes.c_uint32
lib.vanedb_rs_last_error.argtypes = []
lib.vanedb_rs_last_error_message.restype = ctypes.c_void_p   # see below
lib.vanedb_rs_last_error_message.argtypes = []
assert lib.vanedb_rs_abi_version() == 1                      # VANEDB_RS_ABI_VERSION
```

Use `c_void_p` rather than `c_char_p` for `vanedb_rs_last_error_message`:
`c_char_p` copies the bytes into a Python object at the boundary, which is
convenient but conceals the lifetime — the pointer is valid only until the next
`vanedb_rs_*` call on that thread, or until the thread exits.

[`examples/ctypes_quickstart.py`](https://github.com/vanedb/vanedb/blob/main/vanedb-capi/examples/ctypes_quickstart.py)
is a complete working consumer; CI runs it against a built library on every
change.

## Exported symbols

The shared library exports exactly the functions the header declares and
nothing else, so two Rust-built libraries in one process cannot collide on
standard-library symbols. The allowlist lives under [`exports/`](exports/)
in the three spellings linkers take (GNU version script, Apple
`-exported_symbols_list`, Windows `.def`) plus a plain list, all generated
from the header by `scripts/capi_exports.py generate`; `build.rs` passes the
version script to ELF linkers and the exported-symbols list to Apple's.
On MSVC, rustc's own `/DEF` already exports exactly the `#[no_mangle]` set
and link.exe takes one definition file, so the generated `.def` is used
there by the check rather than by the link. CI asserts the built library's
export set with `nm -D --defined-only` on Linux, `nm -gU` on macOS and
`dumpbin /EXPORTS` on Windows, and a test holds the lists to the header.

The static library is post-processed on Linux (`ld -r` then `objcopy
--keep-global-symbols`) and macOS (`ld -r -exported_symbols_list`) so that
only `vanedb_rs_*` is global; the system libraries from `native-static-libs`
stay as undefined references. There is no equivalent of `objcopy` for COFF
archives in the MSVC toolset, so the Windows static library is packaged as
rustc produced it, with every Rust symbol global. Linking it next to another
Rust-built static library on Windows can therefore collide; use the DLL there.

## ABI compatibility gate

The `abidiff` CI job (libabigail) compares the freshly built Linux x86-64
shared library against the previous tagged release's: a removed or changed
symbol fails, an added symbol passes. The baseline procedure:

1. The job reads the version from `vanedb-capi/Cargo.toml` and lists GitHub
   Releases with `gh`. The baseline is the newest non-draft release tagged
   `vanedb-v<version>` (or `vanedb-crate-v<version>`) whose version is older
   than the crate's.
2. It downloads that release's `vanedb-capi-<version>-linux-x86_64.zip`,
   extracts `lib/libvanedb_capi.so`, and runs
   `abidiff --no-added-syms --headers-dir2 vanedb-capi/include <baseline> <current>`.
3. If no older release exists, or the release carries no such asset, the job
   prints a notice and passes without comparing. Releases before RFC 0002
   stage 5 attached no C ABI archives, so the first baseline is the first
   release that does — 0.2.0 itself. From then on every release is the next
   one's baseline automatically.

To compare locally against any shared object:
`python3 scripts/capi_abidiff.py target/capi/libvanedb_capi.so --baseline /path/to/old/libvanedb_capi.so`.

This is a 0.x ABI: it may change in a minor release, and
`VANEDB_RS_ABI_VERSION` is bumped when it does. See the
[repository guide](https://github.com/vanedb/vanedb#persistence) for persistence
limitations, and [`docs/PLATFORMS.md`](../docs/PLATFORMS.md) for platform
verification.
