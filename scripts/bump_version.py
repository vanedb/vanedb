#!/usr/bin/env python3
"""Set the release version everywhere it is declared.

Ten sites state the version across four languages. Eight can spell a
prerelease; `cpp/CMakeLists.txt`'s `project(VERSION)` and `version.h`'s numeric
components cannot, because CMake accepts numeric components only -- those track
the release core. `vanedb-capi/include/vanedb_rs_capi.h` is not edited here at
all: `vanedb-capi/build.rs` stamps it from `CARGO_PKG_VERSION`.

`bench/tests/release_identity.rs` already detects a partial bump. This performs
one, and then runs that test rather than claiming success -- so an incomplete
edit fails here instead of at a tag.

The site list below is a second copy of the test's, which is the kind of
hand-maintained mirror that drifts. Two cheap guards, in place of a clever one:
every path edited here must appear somewhere in the test's source, and the test
is run at the end -- so a site the test gained and this script lacks fails the
run. An earlier version tried to parse the paths back out of the test; it
silently found four of ten, because the test contains `split('"')` and naive
quote pairing goes wrong from there on.
"""

import argparse
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
IDENTITY = ROOT / "bench/tests/release_identity.rs"

# (path, regex with one capture group around the version, kind)
#   full -> takes a prerelease suffix;  core -> numeric components only
SITES = [
    ("vanedb/Cargo.toml",        r'(?m)^version = "([^"]+)"',            "full"),
    ("vanedb-py/Cargo.toml",     r'(?m)^version = "([^"]+)"',            "full"),
    ("vanedb-capi/Cargo.toml",   r'(?m)^version = "([^"]+)"',            "full"),
    ("vanedb-wasm/Cargo.toml",   r'(?m)^version = "([^"]+)"',            "full"),
    ("vanedb-py/pyproject.toml", r'(?m)^version = "([^"]+)"',            "full"),
    ("cpp/pyproject.toml",       r'(?m)^version = "([^"]+)"',            "full"),
    ("cpp/src/core/version.h",   r'VERSION_STRING = "([^"]+)"',          "full"),
    ("cpp/Doxyfile",             r'PROJECT_NUMBER\s+= "([^"]+)"',        "full"),
    ("cpp/CMakeLists.txt",       r'project\(vanedb VERSION ([0-9.]+)',   "core"),
    ("cpp/src/core/version.h",   r'VERSION_MAJOR = (\d+)',               "major"),
    ("cpp/src/core/version.h",   r'VERSION_MINOR = (\d+)',               "minor"),
    ("cpp/src/core/version.h",   r'VERSION_PATCH = (\d+)',               "patch"),
]
# Edited by build.rs, not here, but the test checks it -- so it belongs in the
# comparison below or the two sets would look different for the wrong reason.
STAMPED = "vanedb-capi/include/vanedb_rs_capi.h"


def unchecked_sites(paths):
    """Paths this script edits that the release-identity test never looks at."""
    identity = IDENTITY.read_text()
    return sorted(p for p in paths if p not in identity)


def core_of(version):
    return version.split("-", 1)[0]


def bump(version):
    unchecked = unchecked_sites({path for path, _, _ in SITES} | {STAMPED})
    if unchecked:
        print("these sites are edited here but not checked by release_identity.rs:")
        for path in unchecked:
            print(f"  {path}")
        print("add them to the test, or a wrong value here ships unnoticed.")
        return 1

    major, minor, patch = core_of(version).split(".")
    replacement = {"full": version, "core": core_of(version),
                   "major": major, "minor": minor, "patch": patch}

    for path, pattern, kind in SITES:
        file = ROOT / path
        text = file.read_text()
        matches = list(re.finditer(pattern, text))
        if len(matches) != 1:
            print(f"{path}: {pattern!r} matched {len(matches)} times, expected 1")
            return 1
        span = matches[0].span(1)
        new = replacement[kind]
        print(f"  {matches[0].group(1):>12} -> {new:<12} {path}  ({kind})")
        file.write_text(text[: span[0]] + new + text[span[1] :])
    return 0


def run(command, **kwargs):
    print(f"\n$ {' '.join(command)}")
    return subprocess.run(command, cwd=ROOT, **kwargs).returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help='e.g. 0.1.0 or 0.1.0-rc.1')
    parser.add_argument("--no-verify", action="store_true",
                        help="skip the lockfile refresh, header restamp and test")
    args = parser.parse_args()

    if bump(args.version):
        sys.exit(1)

    if not args.no_verify:
        for command in (
            ["cargo", "update", "-w", "--offline"],
            ["cargo", "update", "--manifest-path", "bench/Cargo.toml", "-w", "--offline"],
            # Restamps vanedb_rs_capi.h from CARGO_PKG_VERSION.
            ["cargo", "build", "-p", "vanedb-capi"],
            ["cargo", "test", "--manifest-path", "bench/Cargo.toml", "--test", "release_identity"],
        ):
            if run(command, stdout=subprocess.DEVNULL):
                print(f"\nFAILED: {' '.join(command)}")
                sys.exit(1)

    print(f"""
Version set to {args.version}. Still manual, and none of it is checked by a test:

  1. vanedb/README.md and vanedb-py/README.md must carry the published-install
     form -- in the release commit, not earlier (RELEASING.md step 5). Both are
     packaged and rendered on a registry that forbids re-upload.
     Check with: scripts/check_release_readmes.py <tag>
  2. CHANGELOG.md: move `## [Unreleased]` to `## [{args.version}] - <date>`.
  3. docs/release/0.1.0-readiness.md: re-designate, with a fresh CI run and
     fresh publication-disabled rehearsals at the new head.
""")
