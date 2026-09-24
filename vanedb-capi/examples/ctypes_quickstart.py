#!/usr/bin/env python3
"""Consume the vanedb C ABI from Python through ctypes.

Run against a built library:

    cargo build -p vanedb-capi --profile capi --locked
    python3 vanedb-capi/examples/ctypes_quickstart.py target/capi/libvanedb_capi.dylib

**Declare `restype` and `argtypes` for every function you call.** ctypes
defaults an undeclared return to a C `int`, which truncates a 64-bit value to
32 bits. Handles are 64-bit ids, not pointers: a truncated handle is refused
by the library with `VANEDB_RS_INVALID_HANDLE` instead of being dereferenced,
so the failure shows up as an error code rather than a SIGSEGV inside libffi
-- but every call made with it fails, so declare the types. The same applies
to arguments: an undeclared 64-bit argument is passed as an int and arrives
truncated.

`vanedb_rs_last_error_message` still returns a pointer, and a truncated
pointer still crashes, so its `restype` matters most of all.
"""

import ctypes
import sys
from pathlib import Path

# Metric values, from vanedb_rs_capi.h. COSINE and DOT are unused here but
# named so the mapping is visible.
L2, COSINE, DOT = 0, 1, 2
# The VANEDB_RS_* codes this example asserts on. The full set is in the header.
OK, NOT_FOUND, DUPLICATE_ID, INVALID_HANDLE = 0, 5, 6, 16
# VANEDB_RS_ABI_VERSION from the header this example was written against. A
# library reporting another value is not the one these bindings describe.
ABI_VERSION = 1

# Every handle -- vanedb_rs_store, vanedb_rs_index, vanedb_rs_disk -- is a
# uint64_t. Not c_void_p: that would truncate on 32-bit Python and, on any
# Python, converts 0 to None.
HANDLE = ctypes.c_uint64
NULL_HANDLE = 0

FILTER_FN = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_uint64, ctypes.c_void_p)


def bind(lib: ctypes.CDLL) -> None:
    """Bind the remaining signatures after the library passes the ABI check."""
    u64 = ctypes.c_uint64
    usize = ctypes.c_size_t
    f32p = ctypes.POINTER(ctypes.c_float)

    lib.vanedb_rs_version.restype = ctypes.c_char_p
    lib.vanedb_rs_version.argtypes = []
    lib.vanedb_rs_handle_count.restype = usize
    lib.vanedb_rs_handle_count.argtypes = []

    lib.vanedb_rs_last_error.restype = ctypes.c_uint32
    lib.vanedb_rs_last_error.argtypes = []

    # c_void_p, not c_char_p. With c_char_p ctypes copies the bytes into a
    # Python object at the boundary, which is convenient but hides the
    # lifetime: the pointer is only valid until the next vanedb_rs_* call on
    # this thread, or until this thread exits. Reading it explicitly keeps that
    # visible.
    lib.vanedb_rs_last_error_message.restype = ctypes.c_void_p
    lib.vanedb_rs_last_error_message.argtypes = []

    lib.vanedb_rs_store_new.restype = HANDLE
    lib.vanedb_rs_store_new.argtypes = [usize, ctypes.c_uint32]
    lib.vanedb_rs_store_free.restype = None
    lib.vanedb_rs_store_free.argtypes = [HANDLE]
    lib.vanedb_rs_store_add.restype = ctypes.c_int32
    lib.vanedb_rs_store_add.argtypes = [HANDLE, u64, f32p]
    lib.vanedb_rs_store_len.restype = usize
    lib.vanedb_rs_store_len.argtypes = [HANDLE]
    lib.vanedb_rs_store_get.restype = ctypes.c_int32
    lib.vanedb_rs_store_get.argtypes = [HANDLE, u64, f32p]
    lib.vanedb_rs_store_search.restype = usize
    lib.vanedb_rs_store_search.argtypes = [
        HANDLE, f32p, usize, ctypes.POINTER(u64), f32p
    ]
    u64p = ctypes.POINTER(u64)
    lib.vanedb_rs_store_search_filtered.restype = usize
    lib.vanedb_rs_store_search_filtered.argtypes = [
        HANDLE,
        f32p,
        usize,
        FILTER_FN,
        ctypes.c_void_p,
        u64p,
        usize,
        u64p,
        usize,
        u64p,
        f32p,
    ]

    lib.vanedb_rs_index_new.restype = HANDLE
    lib.vanedb_rs_index_new.argtypes = [
        usize, ctypes.c_uint32, usize, usize, usize, u64
    ]
    lib.vanedb_rs_index_free.restype = None
    lib.vanedb_rs_index_free.argtypes = [HANDLE]
    lib.vanedb_rs_index_add.restype = ctypes.c_int32
    lib.vanedb_rs_index_add.argtypes = [HANDLE, u64, f32p]
    lib.vanedb_rs_index_save_to_buffer.restype = ctypes.c_int32
    lib.vanedb_rs_index_save_to_buffer.argtypes = [
        HANDLE, ctypes.POINTER(ctypes.c_uint8), usize,
        ctypes.POINTER(usize),
    ]
    lib.vanedb_rs_index_load_from_buffer.restype = HANDLE
    lib.vanedb_rs_index_load_from_buffer.argtypes = [
        ctypes.POINTER(ctypes.c_uint8), usize,
    ]
    lib.vanedb_rs_index_search.restype = usize
    lib.vanedb_rs_index_search.argtypes = [
        HANDLE, f32p, usize, usize, ctypes.POINTER(u64), f32p
    ]
    lib.vanedb_rs_index_len.restype = usize
    lib.vanedb_rs_index_len.argtypes = [HANDLE]


def message(lib: ctypes.CDLL) -> str:
    pointer = lib.vanedb_rs_last_error_message()
    return ctypes.string_at(pointer).decode() if pointer else ""


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    library = Path(sys.argv[1])
    if not library.exists():
        print(f"no library at {library}; build it with "
              f"`cargo build -p vanedb-capi --profile capi --locked`")
        return 2

    lib = ctypes.CDLL(str(library))
    lib.vanedb_rs_abi_version.restype = ctypes.c_uint32
    lib.vanedb_rs_abi_version.argtypes = []
    library_abi = lib.vanedb_rs_abi_version()
    if library_abi != ABI_VERSION:
        print(f"library speaks ABI {library_abi}, "
              f"these bindings speak {ABI_VERSION}")
        return 1
    bind(lib)
    print("vanedb", lib.vanedb_rs_version().decode())

    dim = 3
    floats = ctypes.c_float * dim
    store = lib.vanedb_rs_store_new(dim, L2)
    if store == NULL_HANDLE:
        print(f"construction failed: code {lib.vanedb_rs_last_error()} "
              f"{message(lib)!r}")
        return 1
    try:
        floats = ctypes.c_float * dim
        assert lib.vanedb_rs_store_add(store, 1, floats(1.0, 0.0, 0.0)) == OK
        assert lib.vanedb_rs_store_add(store, 2, floats(0.0, 1.0, 0.0)) == OK
        print("stored", lib.vanedb_rs_store_len(store), "vectors")

        # Failures report a reason. The return value alone cannot: every status
        # function returns 1, and a search returns 0 results for both an empty
        # store and a rejected query.
        if lib.vanedb_rs_store_add(store, 1, floats(9.0, 9.0, 9.0)) != OK:
            code = lib.vanedb_rs_last_error()
            print(f"duplicate rejected: code {code} {message(lib)!r}")
            assert code == DUPLICATE_ID

        out = floats()
        if lib.vanedb_rs_store_get(store, 99, out) != OK:
            assert lib.vanedb_rs_last_error() == NOT_FOUND
            print(f"absent id: code {NOT_FOUND} {message(lib)!r}")

        # What a caller who forgot `restype` would have passed: the handle
        # truncated to 32 bits. It is refused, not dereferenced.
        truncated = store & 0xFFFF_FFFF
        assert lib.vanedb_rs_store_len(truncated) == 0
        assert lib.vanedb_rs_last_error() == INVALID_HANDLE
        print(f"truncated handle refused: code {INVALID_HANDLE} {message(lib)!r}")

        k = 2
        ids = (ctypes.c_uint64 * k)()
        distances = (ctypes.c_float * k)()
        found = lib.vanedb_rs_store_search(store, floats(1.0, 0.0, 0.0), k, ids, distances)
        assert lib.vanedb_rs_last_error() == OK, "a real failure would set a code"
        print("nearest:", [(ids[i], round(distances[i], 4)) for i in range(found)])

        allow_ids = (ctypes.c_uint64 * 1)(2)
        found_filtered = lib.vanedb_rs_store_search_filtered(
            store,
            floats(1.0, 0.0, 0.0),
            k,
            ctypes.cast(None, FILTER_FN),
            None,
            allow_ids,
            1,
            None,
            0,
            ids,
            distances,
        )
        assert lib.vanedb_rs_last_error() == OK
        assert found_filtered == 1
        assert ids[0] == 2
        print("filtered nearest:", [(ids[i], round(distances[i], 4)) for i in range(found_filtered)])
    finally:
        lib.vanedb_rs_store_free(store)

    # The same VNDB file as path save/load, without a filesystem.
    index = lib.vanedb_rs_index_new(dim, L2, 16, 4, 16, 42)
    if index == NULL_HANDLE:
        print(f"index construction failed: code {lib.vanedb_rs_last_error()} "
              f"{message(lib)!r}")
        return 1
    try:
        floats = ctypes.c_float * dim
        assert lib.vanedb_rs_index_add(index, 1, floats(1.0, 0.0, 0.0)) == OK
        needed = ctypes.c_size_t(0)
        assert lib.vanedb_rs_index_save_to_buffer(index, None, 0, ctypes.byref(needed)) == OK
        buf = (ctypes.c_uint8 * needed.value)()
        wrote = ctypes.c_size_t(needed.value)
        assert lib.vanedb_rs_index_save_to_buffer(
            index, buf, needed.value, ctypes.byref(wrote)
        ) == OK
        print("serialized", wrote.value, "bytes")
    finally:
        lib.vanedb_rs_index_free(index)

    loaded = lib.vanedb_rs_index_load_from_buffer(buf, wrote.value)
    if loaded == NULL_HANDLE:
        print(f"buffer load failed: code {lib.vanedb_rs_last_error()} "
              f"{message(lib)!r}")
        return 1
    try:
        assert lib.vanedb_rs_index_len(loaded) == 1
        ids = (ctypes.c_uint64 * 1)()
        distances = (ctypes.c_float * 1)()
        found = lib.vanedb_rs_index_search(
            loaded, floats(1.0, 0.0, 0.0), 1, 0, ids, distances
        )
        assert found == 1 and ids[0] == 1
        print("loaded from buffer; nearest:", ids[0])
    finally:
        lib.vanedb_rs_index_free(loaded)

    # Every handle above was freed; a consumer's test suite can assert this.
    assert lib.vanedb_rs_handle_count() == 0
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
