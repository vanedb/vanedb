#!/usr/bin/env python3
"""The C ABI's exported-symbol allowlist, derived from its header (RFC 0002).

`generate` reads every `vanedb_rs_*` function declared in
`vanedb-capi/include/vanedb_rs_capi.h` and writes the same set in the two
spellings linkers take, a plain list for `objcopy`, and the canonical
signature list the ABI gate diffs against a baseline release's header:

    vanedb-capi/exports/vanedb_capi.exp   Apple -exported_symbols_list
    vanedb-capi/exports/vanedb_capi.def   Windows module-definition file
    vanedb-capi/exports/vanedb_capi.syms  one symbol per line, no decoration
    vanedb-capi/exports/vanedb_capi.sigs  one normalised prototype per line:
                                          return type, name, parameter types
                                          (typedefs expanded, parameter
                                          names dropped), sorted

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
COMMENT = re.compile(r"/\*.*?\*/|//[^\n]*", re.DOTALL)


def functions(header_text):
    """Every exported function name, sorted, from the header's declarations."""
    stripped = COMMENT.sub("", header_text)
    names = sorted(set(DECLARATION.findall(stripped)))
    if not names:
        raise SystemExit("no vanedb_rs_* declarations found; is the header generated?")
    return names


PREPROCESSOR = re.compile(r"^[ \t]*#.*$", re.MULTILINE)
BRACES = re.compile(r'extern "C" \{|^\s*\}\s*$', re.MULTILINE)
STATEMENT = re.compile(r"\s*(.*?)\s*\b(vanedb_rs_[a-z0-9_]+)\s*\((.*)\)\s*", re.DOTALL)
ABI_VERSION = re.compile(r"^#define VANEDB_RS_ABI_VERSION (\d+)$", re.MULTILINE)


# Deliberately support the generated header's small C declaration subset,
# including the opaque struct aliases in the 0.1.1 baseline. Unknown types
# or syntax fail closed: the stripped binary cannot verify what we skip.
IDENTIFIER = r"[A-Za-z_][A-Za-z0-9_]*"
SCALAR_TYPES = {
    "void", "bool", "char", "signed char", "unsigned char", "short",
    "short int", "unsigned short", "unsigned short int", "int", "unsigned",
    "unsigned int", "long", "long int", "unsigned long", "unsigned long int",
    "long long", "long long int", "unsigned long long", "unsigned long long int",
    "float", "double", "long double", "size_t", "ptrdiff_t", "intptr_t", "uintptr_t",
    *(f"{prefix}int{width}_t" for prefix in ("", "u") for width in (8, 16, 32, 64)),
}
TYPEDEF = re.compile(rf"typedef\s+(.+?)\s+({IDENTIFIER})", re.DOTALL)
CALLBACK = re.compile(rf"typedef\s+(.+?)\(\s*\*\s*({IDENTIFIER})\s*\)\s*\((.*)\)", re.DOTALL)


def prototypes(header_text):
    """Function prototypes with names removed and typedefs fully expanded.

    Scalar aliases, alias chains, opaque struct aliases and callback typedefs
    cover the current generated header and the legacy release. This is not a
    general C parser: arrays, inline callbacks, struct bodies and attributes
    require an explicit extension before the gate can accept them.
    """
    text = BRACES.sub("", PREPROCESSOR.sub("", COMMENT.sub("", header_text)))
    aliases, declarations = {}, []
    for raw in text.split(";"):
        statement = raw.strip()
        if not statement:
            continue
        if statement.startswith("typedef"):
            callback = CALLBACK.fullmatch(statement)
            alias = TYPEDEF.fullmatch(statement)
            if callback:
                returns, name, parameters = callback.groups()
                definition = (returns, parameters)
            elif alias:
                definition, name = alias.groups()
            else:
                raise SystemExit(f"unsupported C ABI typedef: {statement}")
            if name in aliases:
                raise SystemExit(f"duplicate C ABI typedef: {name}")
            aliases[name] = definition
        else:
            match = STATEMENT.fullmatch(statement)
            if not match:
                raise SystemExit(f"unsupported C ABI declaration: {statement}")
            declarations.append(match.groups())

    resolved = {}

    def resolve_alias(name, visiting):
        if name in visiting:
            raise ValueError(f"cyclic typedef: {name}")
        if name not in resolved:
            definition = aliases[name]
            visiting = visiting | {name}
            if isinstance(definition, tuple):
                returns, parameters = definition
                resolved[name] = f"{normalise_type(returns, visiting)} (*)({parameter_types(parameters, visiting)})"
            else:
                resolved[name] = normalise_type(definition, visiting)
        return resolved[name]

    def normalise_type(value, visiting):
        value = " ".join(value.split())
        match = re.fullmatch(r"(const )?([A-Za-z_][A-Za-z0-9_ ]*?)(\s*\*(?:\s*const)?)*", value)
        if not match:
            raise ValueError(f"unsupported type: {value}")
        # Split at the first '*' to retain every pointer in e.g. char **.
        base, *pointers = value.split("*")
        base = base.strip()
        qualifier = "const " if base.startswith("const ") else ""
        base = base.removeprefix("const ")
        if base in aliases:
            base = resolve_alias(base, visiting)
            if ("(*)" in base and pointers) or ("*" in base and qualifier):
                raise ValueError(f"unsupported qualified alias type: {value}")
        elif base not in SCALAR_TYPES and not re.fullmatch(rf"struct {IDENTIFIER}", base):
            raise ValueError(f"unknown type: {base}")
        return qualifier + base + "".join(" *" + (" const" if p.strip() else "") for p in pointers)

    def normalise_parameter(value, visiting):
        value = " ".join(value.split())
        try:
            return normalise_type(value, visiting)
        except ValueError as unnamed_error:
            # A named parameter ends in an identifier after whitespace or '*'.
            match = re.fullmatch(rf"(.+[\s*])({IDENTIFIER})", value)
            if not match or match.group(2) in {
                "const", "volatile", "restrict", "signed", "unsigned", "short",
                "long", "void", "char", "int", "float", "double", "struct",
            }:
                raise unnamed_error
            return normalise_type(match.group(1).strip(), visiting)

    def parameter_types(parameters, visiting):
        if not parameters.strip():
            raise ValueError("unspecified argument list; use void for no arguments")
        return ", ".join(normalise_parameter(p, visiting) for p in parameters.split(","))

    found = {}
    try:
        # Validate even unused aliases so adding unsupported public syntax
        # cannot silently weaken the header comparison.
        for name in aliases:
            resolve_alias(name, set())
        for returns, name, parameters in declarations:
            if name in found:
                raise ValueError(f"duplicate function: {name}")
            returns = normalise_type(returns, set())
            parameters = parameter_types(parameters, set())
            found[name] = re.sub(r"\*\s+vanedb", "*vanedb", f"{returns} {name}({parameters})")
    except ValueError as error:
        raise SystemExit(f"unsupported C ABI declaration: {error}") from None
    if not found:
        raise SystemExit("no vanedb_rs_* prototypes found; is the header generated?")
    return found


def abi_version(header_text):
    """VANEDB_RS_ABI_VERSION, or None for a header that predates it."""
    match = ABI_VERSION.search(header_text)
    return int(match.group(1)) if match else None


def render(names, signatures):
    apple = "".join(f"_{n}\n" for n in names)
    module_definition = "EXPORTS\n" + "".join(f"    {n}\n" for n in names)
    plain = "".join(f"{n}\n" for n in names)
    sigs = "".join(f"{signatures[n]}\n" for n in names)
    return {
        "vanedb_capi.exp": apple,
        "vanedb_capi.def": module_definition,
        "vanedb_capi.syms": plain,
        "vanedb_capi.sigs": sigs,
    }


def generate(verify=False):
    header_text = HEADER.read_text(encoding="utf-8")
    names, signatures = functions(header_text), prototypes(header_text)
    if set(names) != set(signatures):
        raise SystemExit(f"prototype parse disagrees with the name parse: "
                         f"{sorted(set(names) ^ set(signatures))}")
    rendered = render(names, signatures)
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
        # LF on every OS: the files are compared byte for byte and pinned to
        # LF in .gitattributes.
        (EXPORTS / name).write_text(text, encoding="utf-8", newline="\n")
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
    # Symbol versioning would print `name@@VERSION`; rustc's version script
    # is anonymous so it does not, but strip defensively so a versioned build
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
