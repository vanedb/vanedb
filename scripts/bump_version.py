#!/usr/bin/env python3
"""Set the release version everywhere it is declared.

Ten sites state the version across four languages. Eight can spell a
prerelease; `cpp/CMakeLists.txt`'s `project(VERSION)` and `version.h`'s three
numeric components cannot, because CMake accepts numeric components only --
those track the release core. `vanedb-capi/include/vanedb_rs_capi.h` is not
edited here: `vanedb-capi/build.rs` stamps it from `CARGO_PKG_VERSION`.

Four of those sites live in one file, `cpp/src/core/version.h`. An earlier
version of this script planned every edit against the text it read at plan
time, then wrote each independently -- so the last write to that file discarded
the other three, and a bump left `VERSION_MAJOR` untouched while reporting it
changed. Edits are grouped per file and applied to one buffer, highest offset
first, so the spans stay valid as the text shifts.

`bench/tests/release_identity.rs` detects a partial bump. This performs one and
then runs that test, rather than claiming success.
"""

import argparse
import pathlib
import re
import subprocess
import sys
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent

# `\Z`, not `$`: `$` matches before a trailing newline, so `0.2.0\n` from a
# `$(cat version.txt)` wrapper was accepted and written into every Cargo.toml.
VERSION = re.compile(r"\A(\d+)\.(\d+)\.(\d+)(?:-[0-9A-Za-z.-]+)?\Z")

# (path, regex with one capture group around the version, kind)
SITES = [
    ("vanedb/Cargo.toml",        r'(?m)^version = "([^"]+)"',          "full"),
    ("vanedb-py/Cargo.toml",     r'(?m)^version = "([^"]+)"',          "full"),
    ("vanedb-capi/Cargo.toml",   r'(?m)^version = "([^"]+)"',          "full"),
    ("vanedb-wasm/Cargo.toml",   r'(?m)^version = "([^"]+)"',          "full"),
    ("vanedb-py/pyproject.toml", r'(?m)^version = "([^"]+)"',          "full"),
    ("cpp/pyproject.toml",       r'(?m)^version = "([^"]+)"',          "full"),
    ("cpp/src/core/version.h",   r'VERSION_STRING = "([^"]+)"',        "full"),
    ("cpp/Doxyfile",             r'PROJECT_NUMBER\s+= "([^"]+)"',      "full"),
    ("cpp/CMakeLists.txt",       r'project\(vanedb VERSION ([0-9.]+)', "core"),
    ("cpp/src/core/version.h",   r"VERSION_MAJOR = (\d+)",             "major"),
    ("cpp/src/core/version.h",   r"VERSION_MINOR = (\d+)",             "minor"),
    ("cpp/src/core/version.h",   r"VERSION_PATCH = (\d+)",             "patch"),
]
# Stamped by build.rs, not here, but checked by the identity test -- so it
# belongs in the coverage comparison below.
STAMPED = "vanedb-capi/include/vanedb_rs_capi.h"

IDENTITY = "bench/tests/release_identity.rs"


def core_of(version):
    return version.split("-", 1)[0]


def unchecked_sites(root, paths):
    """Paths edited here that the release-identity test never looks at.

    Comment lines are stripped first: the test mentions
    `target/npm/vanedb-wasm/package.json` only inside a comment explaining that
    it is deliberately *not* checked, and a plain substring search over the
    whole source would count that as coverage.
    """
    source = (root / IDENTITY).read_text(encoding="utf-8")
    code = "\n".join(l for l in source.splitlines() if not l.lstrip().startswith("//"))
    return sorted(p for p in paths if p not in code)


def plan(root, version):
    """Every edit, grouped per file. Returns (edits, error)."""
    if not VERSION.match(version):
        return None, (f"{version!r} is not a version: expected MAJOR.MINOR.PATCH "
                      f"with an optional -prerelease, e.g. 0.1.0 or 0.1.0-rc.1")

    unchecked = unchecked_sites(root, {p for p, _, _ in SITES} | {STAMPED})
    if unchecked:
        return None, ("these sites are edited here but not checked by "
                      + IDENTITY + ":\n  " + "\n  ".join(unchecked))

    major, minor, patch = core_of(version).split(".")
    replacement = {"full": version, "core": core_of(version),
                   "major": major, "minor": minor, "patch": patch}

    per_file = defaultdict(list)
    for path, pattern, kind in SITES:
        text = (root / path).read_text(encoding="utf-8", newline="")
        matches = list(re.finditer(pattern, text))
        if len(matches) != 1:
            return None, (f"{path}: {pattern!r} matched {len(matches)} times, "
                          f"expected 1\nnothing was written; the tree is unchanged.")
        per_file[path].append((matches[0].span(1), matches[0].group(1), replacement[kind], kind))
    return per_file, None


def bump(root, version, quiet=False):
    per_file, error = plan(root, version)
    if error:
        print(error)
        return 1
    for path, edits in per_file.items():
        file = root / path
        text = file.read_text(encoding="utf-8", newline="")
        # Highest offset first, so earlier spans stay valid as the text shifts.
        for (start, end), was, now, kind in sorted(edits, key=lambda e: -e[0][0]):
            text = text[:start] + now + text[end:]
            if not quiet:
                print(f"  {was:>12} -> {now:<12} {path}  ({kind})")
        file.write_text(text, encoding="utf-8", newline="")
    return 0


def run(command, root):
    print(f"\n$ {' '.join(command)}")
    return subprocess.run(command, cwd=root, stdout=subprocess.DEVNULL).returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="e.g. 0.1.0 or 0.1.0-rc.1")
    parser.add_argument("--root", type=pathlib.Path, default=ROOT)
    parser.add_argument("--no-verify", action="store_true",
                        help="skip the lockfile refresh, header restamp and test")
    args = parser.parse_args()

    if bump(args.root, args.version):
        sys.exit(1)

    if not args.no_verify:
        for command in (
            ["cargo", "update", "-w", "--offline"],
            ["cargo", "update", "--manifest-path", "bench/Cargo.toml", "-w", "--offline"],
            ["cargo", "build", "-p", "vanedb-capi"],
            ["cargo", "test", "--manifest-path", "bench/Cargo.toml", "--test", "release_identity"],
        ):
            if run(command, args.root):
                print(f"\nFAILED: {' '.join(command)}")
                sys.exit(1)

    print(f"""
Version set to {args.version}. Remaining release tasks:

  1. vanedb/README.md, vanedb-py/README.md and vanedb-wasm/README.md must carry
     the published-install form -- in the release commit, not earlier
     (RELEASING.md step 5). All three are packaged and rendered on a registry
     that forbids re-upload.
     Check with: scripts/check_release_readmes.py <tag>
  2. CHANGELOG.md: move `## [Unreleased]` to `## [{args.version}] - <date>`.
  3. docs/release/{args.version}-readiness.md: re-designate, with a fresh CI run.
""")
