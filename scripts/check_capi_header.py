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
import shutil
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
    """(name, language) per distinct toolchain. `cc` and `c++` are usually
    aliases of gcc/g++ or clang/clang++; a compiler is counted once per
    resolved binary, under the first name that reached it."""
    found = []
    seen = set()
    for language, candidates in [("c", [os.environ.get("CC", "cc"), "gcc", "clang"]),
                                 ("c++", [os.environ.get("CXX", "c++"), "g++", "clang++"])]:
        for candidate in candidates:
            path = shutil.which(candidate) if candidate else None
            if not path or not available([candidate, "--version"]):
                continue
            identity = (language, os.path.realpath(path))
            if identity in seen:
                continue
            seen.add(identity)
            found.append((candidate, language))
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


MSVC_LEGS = [
    # /Za would reject the Windows headers and MSVC has no C99-only switch,
    # so its default C mode stands in for C99; /std:c11 is the C11 leg.
    ("C (default mode)", ["/W4", "/WX", "/c", "/Tc", "tu.c"]),
    ("C11", ["/W4", "/WX", "/std:c11", "/c", "/Tc", "tu.c"]),
    ("C++17", ["/W4", "/WX", "/std:c++17", "/c", "/Tp", "tu.cpp"]),
]


def msvc_batch(vcvars_path, include, work):
    """The batch file that runs every MSVC leg.

    A `cmd /c "call \"...vcvars64.bat\" && cl ..."` list element does not
    survive subprocess: list2cmdline escapes the inner quotes to \" and cmd
    then looks for a program literally named \"C:\...\". A batch file
    carries the quoted path verbatim.
    """
    # Paths are joined with a backslash by hand so the text is the same on
    # every host (the unit test runs on Linux). /Fo is not quoted: cl reads a
    # trailing `\"` as an escaped quote, and the temp dir has no spaces.
    lines = ["@echo off", f'call "{vcvars_path}" >nul || exit /b 1']
    for _, leg in MSVC_LEGS:
        arguments = " ".join(f"{work}\\{a}" if a.startswith("tu.") else a for a in leg)
        lines.append(f'cl /nologo /I"{include}" /Fo{work}\\ {arguments} || exit /b 1')
    return "\r\n".join(lines) + "\r\n"


def run_msvc(work):
    (work / "tu.c").write_text(SOURCE)
    (work / "tu.cpp").write_text(SOURCE)
    batch = work / "check_header.bat"
    text = msvc_batch(vcvars(), INCLUDE, work)
    batch.write_text(text, newline="")
    print(text, flush=True)
    subprocess.run(["cmd", "/c", str(batch)], check=True)


def main():
    with tempfile.TemporaryDirectory(prefix="vanedb-header-") as temporary:
        work = Path(temporary)
        if sys.platform == "win32":
            run_msvc(work)
            print("MSVC /W4 /WX: " + ", ".join(name for name, _ in MSVC_LEGS) + " accepted the header")
            return
        compilers = gnu_compilers()
        if not compilers:
            raise SystemExit("no C or C++ compiler found")
        for compiler, language in compilers:
            run_gnu(compiler, language, work)
        names = ", ".join(name for name, _ in compilers)
        print(f"{len(compilers)} distinct toolchains ({names}) accepted the header "
              f"under -Wall -Wextra -pedantic -Werror")


if __name__ == "__main__":
    main()
