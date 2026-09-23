# VaneDB from C

Embed the Rust engine in a C or C++ application. You need a C compiler and
CMake 3.20 or newer (or pkg-config and make), plus Rust when building from
source.

Each release attaches platform archives named
`vanedb-capi-<version>-<platform>.zip`, with a SHA-256 checksum, to the
`vanedb-crate-v<version>` [GitHub Release](https://github.com/vanedb/vanedb/releases);
the first is 0.1.1. Choose the archive matching your OS and architecture. A
successful [CI run](https://github.com/vanedb/vanedb/actions/workflows/ci.yml)
carries the same archives for the commit it built, as development artifacts.
Check the archive's
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
[`docs/PLATFORMS.md`](https://github.com/vanedb/vanedb/blob/main/docs/PLATFORMS.md), which also records that the
macOS Intel archive ends when GitHub's last Intel runner retires in August 2027.
These are inspected binary requirements, not proof of runtime support on the
oldest OS. Consumer acceptance runs on the recorded CI host. In particular, the
Windows PE subsystem version does not establish the application's minimum OS.
Build from source and verify on your deployment if it differs from those hosts.

## Signed release verification (0.2.0 onward)

The [`vanedb-crate-v<version>` release](https://github.com/vanedb/vanedb/releases)
carries all five desktop archives, a target-specific CycloneDX SBOM beside
each archive, `CAPI-RELEASE.json` (source commit and artifact inventory),
`CAPI-VERIFYING.md`, and `SHA256SUMS`. Every payload has a matching
`.sigstore.json` keyless signature bundle. The bundle binds the exact bytes
to this repository's release workflow through GitHub OIDC and Sigstore's
transparency log. These signatures start with 0.2.0; 0.1.1 assets are unsigned.

Download the files into an empty directory. With
[cosign v3.1.3](https://github.com/sigstore/cosign/releases/tag/v3.1.3),
verify the checksum file's exact release identity **before** trusting its hashes
(substitute the desired version):

```sh
version=0.2.0
cosign verify-blob SHA256SUMS --bundle SHA256SUMS.sigstore.json \
  --certificate-identity "https://github.com/vanedb/vanedb/.github/workflows/publish-capi.yml@refs/tags/vanedb-crate-v$version" \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com
sha256sum --check SHA256SUMS  # macOS: shasum -a 256 --check SHA256SUMS
```

The release notes and `CAPI-VERIFYING.md` contain the complete verification
command with the approved source commit and its matching OIDC certificate SHA
claim. The full verifier enforces both. To verify one archive independently,
replace `SHA256SUMS` and its bundle with that archive and its bundle, retaining
the exact certificate identity and issuer. Do not accept a branch identity for
a tagged release. The SBOM inventories Cargo dependencies (including build
dependencies) for the recorded target and default C ABI features; system
library requirements remain in the archive's `compatibility.json`.

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

`Libs` is `-L<libdir> -lvanedb_capi`, which links the shared library, and a
program linked that way needs the library on its loader path at run time:
an rpath (`-Wl,-rpath,$(pkg-config --variable=libdir vanedb)`, which the
shipped Makefile adds) or `LD_LIBRARY_PATH` / `DYLD_LIBRARY_PATH`. Note that
`pkg-config --static --libs vanedb` still links the shared library: `-l`
prefers the `.so` when both exist. To link statically, name the archive by
path and take only the system libraries from `Libs.private`, as the shipped
Makefile does:

```sh
cc app.c $(pkg-config --cflags vanedb) \
   "$(pkg-config --variable=libdir vanedb)/libvanedb_capi.a" \
   $(pkg-config --static --libs-only-l --libs-only-other vanedb | sed 's/-lvanedb_capi//')
```

Not available on Windows.

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

A plain `--release` build still works and lands in `target/release/`. On
macOS, packaging also needs `rustup component add llvm-tools-preview`
for the active toolchain (or `llvm-objcopy` on `PATH`). The CMake
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

The static library keeps its symbols and debug information; only the embedded
LLVM bitcode is removed (see "Exported symbols"), which on the same build
takes `libvanedb_capi.a` from 22,782,298 bytes as cargo wrote it to
12,588,258 bytes as packaged. The shared library carries no bitcode, so its
figures above are unaffected.

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

**Mapped files must outlive in-flight calls.** From the start of
`vanedb_rs_disk_open`, keep the underlying file unmodified and untruncated
until its handle has been freed **and every in-flight call using it has
returned**. `vanedb_rs_disk_free` does not wait for these calls; each retains
its own mapping. Synchronize with all calling threads before modifying or
truncating that file. Replacing its path with a newly built file is allowed;
modifying the mapped file in place is not.

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
while a store or index handle's read lock is held (a disk handle has no
lock), and the graph search may call it more than once for the same id as it
widens its beam. It must not access, modify or free the handle being searched
(re-entering the same index under that lock) and must not modify or free the
search buffers. Calls on other handles are allowed; an error such a call
records does not replace the outer search's result. The callback and
everything `user_data` points to must stay valid
until the search returns. Every external callback must contain its own panics
and exceptions; neither unwinding nor `longjmp` may leave it. This includes
Rust callbacks declared `extern "C-unwind"`: a panic from a separately linked
Rust runtime is a foreign exception and can abort the process. The library's
panic boundary contains engine panics; it cannot promise recovery from another
runtime's exception. A Rust callback can use its own `catch_unwind` and return
false on a locally handled failure. ID
lists are the fast path because they never cross the language boundary per
candidate.

For example, a graph search restricted to two ids:

```c
const uint64_t allow[] = {101, 202};          /* strictly ascending */
uint64_t ids[10]; float dists[10];
size_t n = vanedb_rs_index_search_filtered(index, query, 10, 0,
    NULL, NULL,                                /* no callback */
    allow, 2, NULL, 0, ids, dists);            /* allow list; no deny list */
```

The store and disk variants take the same arguments without `ef_search`.
[`examples/ctypes_quickstart.py`](examples/ctypes_quickstart.py) makes the
store call from Python, including the `CFUNCTYPE` declaration for a callback.

`vanedb_rs_index_search_filtered` takes `ef_search` under the same rule as the
plain graph search: `0` uses the handle's setting. The C ABI omits the beam
cap (`max_ef_search` in Rust, Python and WebAssembly) by design: a filtered
graph search widens its beam up to the core's default of four times the
initial beam. Raise `ef_search` on the call to improve recall under a
selective filter — on the order of `k` divided by the fraction of ids the
filter accepts — since the cap alone does not improve results that already
fill `k`. Measured recall at several selectivities is in
[the 0.2.0 validation record](https://github.com/vanedb/vanedb/blob/main/docs/release/0.2.0-filtered-search-validation.md#recall-on-real-embeddings).

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
standard-library symbols. rustc already restricts a `cdylib`'s exports to
its `#[no_mangle]` set on every platform, through an anonymous version
script on ELF, an exported-symbols list on Apple and a `/DEF` on MSVC. GNU
ld refuses a second version script beside rustc's anonymous one ("anonymous
version tag cannot be combined with other version tags"; lld does not, which
is why only the ARM64 leg caught it), so on Linux no extra list is passed
and the CI assertion is the gate. The allowlist under [`exports/`](exports/)
is generated from the header by `scripts/capi_exports.py generate`: the
Apple `-exported_symbols_list`, which `build.rs` adds beside rustc's; the
Windows `.def`, used by the check rather than by the link since link.exe
takes one definition file; and a plain list, which localizes the static
library. CI asserts the built library's export set with
`nm -D --defined-only` on Linux, `nm -gU` on macOS and `dumpbin /EXPORTS`
on Windows, and a test holds the lists to the header.

The static library is post-processed on Linux (`ld -r` then `objcopy
--keep-global-symbols`) and macOS (`clang -r -Wl,-exported_symbols_list`,
driven through the compiler so the arch and deployment target are supplied
from what rustc built) so that only `vanedb_rs_*` is global; the system
libraries from `native-static-libs` stay as undefined references. The same
step removes the LLVM bitcode that fat LTO embeds in every staticlib object
(`.llvmbc`/`.llvmcmd` on ELF, `__LLVM,__bitcode` on Mach-O): nothing links
against it, Apple's `nm` cannot read rustc's newer bitcode, and it is most of
the archive's size.

On Windows, packaging performs a separate staticlib-only fat-LTO build and
localizes its single implementation object with GNU COFF `objcopy`.
`llvm-nm` requires exactly the header's API functions on that object. The
archive also retains rustc's unchanged native import members for kernel32,
bcryptprimitives and the synchronization API set; their individually checked
import descriptors and thunks remain global. No Rust implementation globals
are allowed. A build that still needs implementation definitions from omitted
archive members fails packaging. This avoids GNU COFF partial linking, which
can corrupt COMDAT and weak-symbol metadata.

Windows packaging needs `llvm-tools-preview` for the active Rust toolchain
and GNU COFF `objcopy` (the hosted Windows image provides binutils). Set
`VANEDB_COFF_OBJCOPY` to its executable if it is outside the usual MinGW paths.
LLVM's `objcopy` does not implement this COFF localization operation. The
plain C consumer requires only a C toolchain. Packaging explicitly enables
`-DVANEDB_TEST_RUST_COEXISTENCE=ON` to additionally link an independently
compiled Rust static library and exercise allocation, threads, TLS and
locally caught unwinding alongside vanedb's full acceptance lifecycle. This
optional check requires rustc; it is off for normal consumers. Successful native
Windows CI is required to establish that runtime result.

## ABI compatibility gate

The `C ABI gate` CI job compares this branch against the previous release in
two layers, because neither alone sees enough:

1. **Prototypes.** `scripts/capi_exports.py` parses the header into one
   normalised prototype per function (return type, name, parameter types;
   parameter names dropped) and commits the list as
   `exports/vanedb_capi.sigs`. The job downloads the baseline release's
   `vanedb-capi-<version>-linux-x86_64.zip`, parses the header it carries
   the same way, and fails on any removed or changed prototype; an added one
   passes, which is the header's rule. This is the layer that sees a changed
   signature.
2. **abidiff** (libabigail) on `lib/libvanedb_capi.so`, `--no-added-syms`.
   The shipped library carries no DWARF (the `capi` profile inherits
   `release`, and the packaged copy is stripped), so abidiff compares the
   dynamic symbol tables only: it detects a removed symbol and nothing else,
   and `--no-added-syms` hides additions too. Measured on this branch
   against the 0.1.1 library: stripped, "0 Removed, 0 Changed" although
   every handle parameter changed from a pointer to `uint64_t`; with debug
   info on both sides, "0 Removed, 47 Changed", exit 4. It stays as the
   binary-level check that a symbol the header declares has not vanished
   from the library.

`VANEDB_RS_ABI_VERSION` keys the verdict. A baseline whose header carries a
different version, or none (0.1.1 predates the macro and counts as 0), is an
intentional incompatible release: both layers run and print what they found,
and the job passes with a notice. Under the same version both layers must
pass. So ABI 1 legitimately breaks against 0.1.1, and every later build under
ABI 1 must stay compatible with the first release that ships it.

The baseline procedure:

1. The job reads the version from `vanedb-capi/Cargo.toml` and lists GitHub
   Releases with `gh`. Candidates are non-draft releases tagged
   `vanedb-crate-v<version>` (the crate release carries the C ABI archives)
   or, second at the same version, `vanedb-v<version>` (the Python tag),
   whose version is at most the crate's — so after a release the next builds
   compare against that release itself.
2. Candidates are tried newest-first; the first whose assets list the Linux
   x86-64 archive is the baseline. A listed asset that fails to download
   fails the job. Only when no release lists the archive does the job pass
   with a notice.
3. 0.1.1 is the first baseline (`vanedb-crate-v0.1.1` attaches all five
   archives). From then on every release is the next one's baseline.

To compare locally against any archive or shared object:
`python3 scripts/capi_abidiff.py target/capi/libvanedb_capi.so --baseline /path/to/vanedb-capi-<version>-linux-x86_64.zip`
(a bare `.so` runs the abidiff layer only; `--skip-abidiff` runs the
prototype layer without libabigail).

This is a 0.x ABI: it may change in a minor release, and
`VANEDB_RS_ABI_VERSION` is bumped when it does. See the
[repository guide](https://github.com/vanedb/vanedb#persistence) for persistence
limitations, and [`docs/PLATFORMS.md`](https://github.com/vanedb/vanedb/blob/main/docs/PLATFORMS.md) for platform
verification.
