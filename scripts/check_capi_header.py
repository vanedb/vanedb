#!/usr/bin/env python3
"""Compile a translation unit that includes the C ABI header as C99, C11 and
C++17 with every warning on and warnings as errors (RFC 0002 stage 1).

GCC and Clang: `-Wall -Wextra -pedantic -Werror`. MSVC: `/W4 /WX`. Every
compiler found on the host is used; at least one must be, or this fails, so a
CI leg cannot pass by having nothing to run. `vanedb-capi/tests/
header_compiles_as_c.rs` does the same from `cargo test` for GCC and Clang;
this script is what the Windows leg runs, and the explicit step elsewhere.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ROOT / "vanedb-capi/include"

SOURCE = r'''#include "vanedb_rs_capi.h"
#include <stdio.h>
static bool accepts(uint64_t id, void *data) {
    return id == *(const uint64_t *)data;
}
int main(void) {
    vanedb_rs_store s = VANEDB_RS_NULL_HANDLE;
    vanedb_rs_index h = VANEDB_RS_NULL_HANDLE;
    vanedb_rs_disk d = VANEDB_RS_NULL_HANDLE;
    float query = 0.0f, distance;
    uint64_t selected = 42, id;
    vanedb_rs_filter_fn filter = accepts;
    if (vanedb_rs_abi_version() != VANEDB_RS_ABI_VERSION) {
        return 1;
    }
    (void)vanedb_rs_store_len(s);
    (void)vanedb_rs_index_len(h);
    (void)vanedb_rs_disk_len(d);
    (void)vanedb_rs_store_search_filtered(s, &query, 1, filter, &selected,
        0, 0, 0, 0, &id, &distance);
    (void)vanedb_rs_index_search_filtered(h, &query, 1, 0, filter, &selected,
        0, 0, 0, 0, &id, &distance);
    (void)vanedb_rs_disk_search_filtered(d, &query, 1, filter, &selected,
        0, 0, 0, 0, &id, &distance);
    printf("%s %u\n", VANEDB_RS_VERSION, (unsigned)VANEDB_RS_INVALID_HANDLE);
    return 0;
}
'''

GNU_C = ["-std=c99", "-std=c11"]
GNU_CXX = ["-std=c++17"]
GNU_FLAGS = ["-Wall", "-Wextra", "-pedantic", "-Werror", "-fsyntax-only"]


def available(command):
    try:
        return subprocess.run(command, capture_output=True).returncode == 0
    except OSError:
        return False


def gnu_compilers():
    found = []
    for candidate in [os.environ.get("CC", "cc"), "gcc", "clang"]:
        if candidate and candidate not in [c for c, _ in found] and available([candidate, "--version"]):
            found.append((candidate, "c"))
    for candidate in [os.environ.get("CXX", "c++"), "g++", "clang++"]:
        if candidate and candidate not in [c for c, _ in found] and available([candidate, "--version"]):
            found.append((candidate, "c++"))
    return found


def run_gnu(compiler, language, work):
    standards = GNU_C if language == "c" else GNU_CXX
    source = work / ("tu.c" if language == "c" else "tu.cpp")
    source.write_text(SOURCE)
    for standard in standards:
        command = [compiler, standard, *GNU_FLAGS, "-I", str(INCLUDE), str(source)]
        print("+", " ".join(command), flush=True)
        subprocess.run(command, check=True)


def vcvars():
    vswhere = Path(os.environ["ProgramFiles(x86)"]) / "Microsoft Visual Studio/Installer/vswhere.exe"
    matches = subprocess.check_output([str(vswhere), "-latest", "-products", "*", "-find",
                                       r"VC\Auxiliary\Build\vcvars64.bat"], text=True).splitlines()
    if not matches:
        raise SystemExit("MSVC vcvars64.bat was not found; install the C++ build tools")
    return matches[0]


def run_msvc(work):
    source_c = work / "tu.c"
    source_cpp = work / "tu.cpp"
    source_c.write_text(SOURCE)
    source_cpp.write_text(SOURCE)
    batch = vcvars()
    # /Za would reject the Windows headers; C99-only mode has no switch, so
    # the default C mode stands in for it and /std:c11 is the C11 leg.
    legs = [
        ["/W4", "/WX", "/c", "/Tc", str(source_c)],
        ["/W4", "/WX", "/std:c11", "/c", "/Tc", str(source_c)],
        ["/W4", "/WX", "/std:c++17", "/c", "/Tp", str(source_cpp)],
    ]
    for leg in legs:
        cl = " ".join(["cl", "/nologo", f"/I{INCLUDE}", f"/Fo{work}\\", *leg])
        command = ["cmd", "/c", f'call "{batch}" >nul && {cl}']
        print("+", cl, flush=True)
        subprocess.run(command, check=True)


def main():
    with tempfile.TemporaryDirectory(prefix="vanedb-header-") as temporary:
        work = Path(temporary)
        if sys.platform == "win32":
            run_msvc(work)
            print("MSVC /W4 /WX: C (default), C11 and C++17 accepted the header")
            return
        compilers = gnu_compilers()
        if not compilers:
            raise SystemExit("no C or C++ compiler found")
        for compiler, language in compilers:
            run_gnu(compiler, language, work)
        print(f"{len(compilers)} compilers accepted the header under -Wall -Wextra -pedantic -Werror")


if __name__ == "__main__":
    main()
