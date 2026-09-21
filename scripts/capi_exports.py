#!/usr/bin/env python3
"""The C ABI's exported-symbol allowlist, derived from its header (RFC 0002).

`generate` reads every `vanedb_rs_*` function declared in
`vanedb-capi/include/vanedb_rs_capi.h` and writes the same set in the three
spellings linkers take, plus a plain list for `objcopy`:

    vanedb-capi/exports/vanedb_capi.map   GNU linker version script (ELF)
    vanedb-capi/exports/vanedb_capi.exp   Apple -exported_symbols_list
    vanedb-capi/exports/vanedb_capi.def   Windows module-definition file
    vanedb-capi/exports/vanedb_capi.syms  one symbol per line, no decoration

`check <library>` lists the defined, external symbols of a built shared
library with the platform's tool (`nm -D --defined-only`, `nm -gU`, or
`dumpbin /EXPORTS`) and fails unless that set is exactly the header's.
`--verify` regenerates into memory and fails if the committed files differ,
which is what CI runs alongside the header-in-sync check.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "vanedb-capi/include/vanedb_rs_capi.h"
EXPORTS = ROOT / "vanedb-capi/exports"
PREFIX = "vanedb_rs_"

# Declarations, once comments are gone: a name followed by its parameter
# list. The function-pointer typedef `bool (*vanedb_rs_filter_fn)(...)` has
# its name followed by `)`, so it does not match, and the handle typedefs
# have no parenthesis at all.
DECLARATION = re.compile(r"\b(vanedb_rs_[a-z0-9_]+)\s*\(")
COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)


def functions(header_text):
    """Every exported function name, sorted, from the header's declarations."""
    stripped = COMMENT.sub("", header_text)
    names = sorted(set(DECLARATION.findall(stripped)))
    if not names:
        raise SystemExit("no vanedb_rs_* declarations found; is the header generated?")
    return names


def render(names):
    version_script = "{\n  global:\n" + "".join(f"    {n};\n" for n in names) + "  local:\n    *;\n};\n"
    apple = "".join(f"_{n}\n" for n in names)
    module_definition = "EXPORTS\n" + "".join(f"    {n}\n" for n in names)
    plain = "".join(f"{n}\n" for n in names)
    return {
        "vanedb_capi.map": version_script,
        "vanedb_capi.exp": apple,
        "vanedb_capi.def": module_definition,
        "vanedb_capi.syms": plain,
    }


def generate(verify=False):
    rendered = render(functions(HEADER.read_text(encoding="utf-8")))
    if verify:
        stale = [name for name, text in rendered.items()
                 if not (EXPORTS / name).exists() or (EXPORTS / name).read_text(encoding="utf-8") != text]
        if stale:
            raise SystemExit("export lists are out of date with the header; run "
                             f"scripts/capi_exports.py generate: {', '.join(stale)}")
        print(f"{len(rendered)} export lists match {HEADER.relative_to(ROOT)}")
        return
    EXPORTS.mkdir(parents=True, exist_ok=True)
    for name, text in rendered.items():
        (EXPORTS / name).write_text(text, encoding="utf-8")
    print(f"wrote {len(rendered)} export lists to {EXPORTS.relative_to(ROOT)}")


def dumpbin():
    found = shutil.which("dumpbin")
    if found:
        return found
    vswhere = Path(os.environ["ProgramFiles(x86)"]) / "Microsoft Visual Studio/Installer/vswhere.exe"
    matches = subprocess.check_output([
        str(vswhere), "-latest", "-products", "*", "-find",
        r"VC\Tools\MSVC\**\bin\Hostx64\x64\dumpbin.exe",
    ], text=True).splitlines()
    if not matches:
        raise SystemExit("MSVC dumpbin was not found; install the C++ build tools")
    return matches[0]


def exported_symbols(library):
    """Defined, externally visible symbols of a shared library, per platform."""
    library = str(library)
    if sys.platform == "win32":
        listing = subprocess.check_output([dumpbin(), "/EXPORTS", library], text=True)
        # Export rows: ordinal, hint, RVA, name. Anything without an RVA is a
        # forwarder or the table header, neither of which this library has.
        return sorted(re.findall(r"^\s+\d+\s+[0-9A-Fa-f]+\s+[0-9A-Fa-f]{8}\s+(\S+)", listing, re.MULTILINE))
    if sys.platform == "darwin":
        listing = subprocess.check_output(["nm", "-gU", library], text=True)
        names = [line.split()[-1] for line in listing.splitlines() if line.strip()]
        return sorted(name[1:] if name.startswith("_") else name for name in names)
    listing = subprocess.check_output(["nm", "-D", "--defined-only", library], text=True)
    return sorted(line.split()[-1] for line in listing.splitlines() if line.strip())


def check(library):
    expected = functions(HEADER.read_text(encoding="utf-8"))
    actual = exported_symbols(library)
    # Symbol versioning would print `name@@VERSION`; the version script is
    # anonymous so it does not, but strip defensively so a versioned build
    # fails on the set, not on the decoration.
    actual = sorted({name.split("@")[0] for name in actual})
    extra = sorted(set(actual) - set(expected))
    missing = sorted(set(expected) - set(actual))
    if extra or missing:
        raise SystemExit(
            f"{library} does not export exactly the header's {len(expected)} functions\n"
            f"  unexpected exports: {extra}\n  missing exports: {missing}"
        )
    print(f"{library} exports exactly the {len(expected)} vanedb_rs_* functions in the header")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    generate_command = commands.add_parser("generate", help="write the export lists from the header")
    generate_command.add_argument("--verify", action="store_true",
                                  help="fail if the committed lists differ instead of writing")
    check_command = commands.add_parser("check", help="assert a built shared library's exports")
    check_command.add_argument("library", type=Path)
    args = parser.parse_args()
    if args.command == "generate":
        generate(verify=args.verify)
    else:
        check(args.library)


if __name__ == "__main__":
    main()
