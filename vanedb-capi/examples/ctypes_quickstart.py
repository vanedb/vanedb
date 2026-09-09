#!/usr/bin/env python3
"""Consume the vanedb C ABI from Python through ctypes.

Run against a built library:

    cargo build -p vanedb-capi --release
    python3 vanedb-capi/examples/ctypes_quickstart.py target/release/libvanedb_capi.dylib

**Declare `restype` and `argtypes` for every function you call.** ctypes
defaults an undeclared return to a C `int`, which truncates a 64-bit pointer to
32 bits. The next dereference reads a garbage address and the process dies with
SIGSEGV — no Python traceback, because the fault happens inside libffi. A crash
whose stack is `PyCFuncPtr_call -> _ctypes_callproc -> ffi_call` is almost
always this and not a bug in the library.

The same applies to arguments: an undeclared pointer argument is passed as an
int, so a 64-bit handle arrives at the callee truncated.
"""

import ctypes
import sys
from pathlib import Path

L2, COSINE, DOT = 0, 1, 2
OK, NULL_ARGUMENT, NOT_FOUND, DUPLICATE_ID = 0, 1, 5, 6


def bind(lib: ctypes.CDLL) -> None:
    """Every signature this example uses. Nothing is called before it is bound."""
    u64 = ctypes.c_uint64
    usize = ctypes.c_size_t
    f32p = ctypes.POINTER(ctypes.c_float)

    lib.vanedb_rs_version.restype = ctypes.c_char_p
    lib.vanedb_rs_version.argtypes = []

    lib.vanedb_rs_last_error.restype = ctypes.c_uint32
    lib.vanedb_rs_last_error.argtypes = []

    # c_void_p, not c_char_p. With c_char_p ctypes copies the bytes into a
    # Python object at the boundary, which is convenient but hides the
    # lifetime: the pointer is only valid until the next vanedb_rs_* call on
    # this thread, or until this thread exits. Reading it explicitly keeps that
    # visible.
    lib.vanedb_rs_last_error_message.restype = ctypes.c_void_p
    lib.vanedb_rs_last_error_message.argtypes = []

    lib.vanedb_rs_store_new.restype = ctypes.c_void_p
    lib.vanedb_rs_store_new.argtypes = [usize, ctypes.c_uint32]
    lib.vanedb_rs_store_free.restype = None
    lib.vanedb_rs_store_free.argtypes = [ctypes.c_void_p]
    lib.vanedb_rs_store_add.restype = ctypes.c_int32
    lib.vanedb_rs_store_add.argtypes = [ctypes.c_void_p, u64, f32p]
    lib.vanedb_rs_store_len.restype = usize
    lib.vanedb_rs_store_len.argtypes = [ctypes.c_void_p]
    lib.vanedb_rs_store_get.restype = ctypes.c_int32
    lib.vanedb_rs_store_get.argtypes = [ctypes.c_void_p, u64, f32p]
    lib.vanedb_rs_store_search.restype = usize
    lib.vanedb_rs_store_search.argtypes = [
        ctypes.c_void_p, f32p, usize, ctypes.POINTER(u64), f32p
    ]


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
              f"`cargo build -p vanedb-capi --release`")
        return 2

    lib = ctypes.CDLL(str(library))
    bind(lib)
    print("vanedb", lib.vanedb_rs_version().decode())

    dim = 3
    store = lib.vanedb_rs_store_new(dim, L2)
    if not store:
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

        k = 2
        ids = (ctypes.c_uint64 * k)()
        distances = floats(*([0.0] * k)) if k == dim else (ctypes.c_float * k)()
        found = lib.vanedb_rs_store_search(store, floats(1.0, 0.0, 0.0), k, ids, distances)
        assert lib.vanedb_rs_last_error() == OK, "a real failure would set a code"
        print("nearest:", [(ids[i], round(distances[i], 4)) for i in range(found)])
    finally:
        lib.vanedb_rs_store_free(store)
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
