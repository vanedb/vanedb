#!/usr/bin/env python3
"""Tests for the release scripts, run by CI's workflow-lint job.

`check_release_readmes.py` previously ran for the first time in its life during
an irreversible publish. Every case below came from a review that defeated the
first version of it by rewording, or that found it rejecting a correct README.
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from check_release_readmes import TARGETS, problems_in  # noqa: E402

CRATE, PY, WASM = TARGETS["vanedb-crate-v"], TARGETS["vanedb-v"], TARGETS["vanedb-wasm-v"]

# (target, text, should_be_ready, why)
CASES = [
    # --- the mistake this exists to catch: the pre-publication README ---
    (CRATE, 'Add `vanedb = { path = "/path/to/vanedb" }` to your dependencies.',
     False, "path dependency is the unconverted crate README"),
    (PY, "python -m pip install ./vanedb-py\nThese instructions do not assume a published release.",
     False, "local pip install is the unconverted Python README"),
    (WASM, "Nothing is published yet; until then, build from a checkout.",
     False, "the sentence that actually shipped on npm"),

    # --- reworded evasions that defeated the first version ---
    (CRATE, "VaneDB is not yet on crates.io. Point Cargo at the checkout. "
            "Once published you will be able to run `cargo add vanedb`.",
     False, "reworded: claims unpublished while naming the published command"),
    (PY, "There is no release yet on PyPI; `pip install vanedb` will work when 0.1.0 ships.",
     False, "reworded: future-tense install instruction"),
    (WASM, "npm install @vanedb/wasm  <!-- not yet on npm, build from a checkout -->",
     False, "reworded: install line with an unpublished disclaimer"),

    # --- correct published forms the first version wrongly rejected ---
    (CRATE, 'vanedb = { version = "0.1.0", features = ["disk"] }',
     True, "the table form a crate with features needs"),
    (CRATE, "Run `cargo add vanedb@0.1.0-rc.1` to try the release candidate.",
     True, "a version-pinned cargo add, which is the honest text during a bootstrap window"),
    (PY, "python -m pip install --upgrade vanedb", True, "upgrade form"),

    # --- a different package must not satisfy the check on its own ---
    (CRATE, "Install the C bindings with `cargo add vanedb-capi`.",
     False, "vanedb-capi is a different package"),
    (PY, "The C++ bindings install with `pip install vanedb-cpp`.",
     False, "vanedb-cpp is a different package"),

    # --- each fails on ONE pattern only, so deleting that pattern turns the
    # --- suite red. Without these, a review found two patterns that could be
    # --- removed with 13/13 still passing, because every other fixture also
    # --- failed the positive install-line requirement.
    (WASM, "npm install @vanedb/wasm\n\nNothing is published yet.",
     False, "isolates 'nothing is published': the install line is present"),
    (CRATE, 'Run `cargo add vanedb@0.1.0-rc.1`.\n\nOr `vanedb = { path = "../vanedb" }`.',
     False, "isolates the crate forbidden list: the install line is present"),
    (PY, "pip install vanedb\n\nThere is no release yet on PyPI.",
     False, "isolates the registry-qualified no-release pattern"),
    (PY, "pip install vanedb\n\nThese instructions do not assume a published release.",
     False, "isolates 'do not assume a published': the install line is present"),
    (CRATE, 'Add `vanedb = "0.1.0-rc.1"` to your dependencies.',
     True, "isolates the plain version-requirement published form"),

    # --- legitimate prose a broad class wrongly rejected ---
    (PY, "pip install vanedb\n\nThe C++ bindings are kept for reference and are not published.",
     True, "a true statement about another package"),
    (WASM, "npm install @vanedb/wasm\n\nPersistence is not available in WebAssembly.\n"
           "## Building from source\nSee the repository.",
     True, "an API limitation and a build section are not publication claims"),
]


PIN_CASES = [
    # (prefix, text, version, should_be_ready, why)
    ("vanedb-crate-v", "cargo add vanedb@0.1.0-rc.2", "0.1.0-rc.2", True,
     "a pin naming the version being published"),
    ("vanedb-crate-v", "cargo add vanedb@0.1.0-rc.1", "0.1.0-rc.2", False,
     "a pin left over from the previous release candidate"),
    ("vanedb-crate-v", 'vanedb = { version = "0.1.0-rc.1", features = ["disk"] }',
     "0.1.0-rc.2", False, "a stale pin in the features table form"),
    ("vanedb-v", "pip install vanedb==0.1.0rc2", "0.1.0-rc.2", True,
     "pip spells the same version differently and must be accepted"),
    ("vanedb-v", "pip install vanedb==0.1.0rc1", "0.1.0-rc.2", False,
     "a stale pip pin"),
    ("vanedb-crate-v", "cargo add vanedb", "0.1.0-rc.2", True,
     "an unpinned line has nothing to check"),
    ("vanedb-crate-v", 'vanedb = { version = "0.1", features = ["gpu-metal"] }',
     "0.1.0", True, "a caret range the published version satisfies is not a stale pin"),
    ("vanedb-crate-v", 'vanedb = { version = "0.1", features = ["gpu-metal"] }',
     "0.1.0-rc.2", False, "a bare range does not resolve to a prerelease, so it is wrong then"),
    ("vanedb-crate-v", 'vanedb = "0.2"', "0.1.0", False,
     "a range the published version does not satisfy"),
    ("vanedb-crate-v", 'vanedb = "0.1"', "0.10.0", False,
     "0.1 must not match 0.10.0 -- the minor boundary, which two surviving "
     "mutants of satisfied_by both crossed"),
    ("vanedb-crate-v", 'vanedb = "1"', "0.1.0", False,
     "a range must be a PREFIX of the version, not a substring of it: '1.' "
     "occurs inside '0.1.0', so relaxing startswith to `in` accepts a README "
     "pinning a 1.x that does not exist"),
]


def check_pins(failures):
    from check_release_readmes import TARGETS, problems_in
    for prefix, text, version, should_be_ready, why in PIN_CASES:
        problems = problems_in(TARGETS[prefix], text, prefix, version)
        ready = not problems
        if ready != should_be_ready:
            failures.append(
                f"pin case: expected {'ready' if should_be_ready else 'NOT ready'} "
                f"({why})\n    text: {text!r}\n    problems: {problems}")


def main():
    failures = []
    check_pins(failures)
    for target, text, should_be_ready, why in CASES:
        problems = problems_in(target, text)
        ready = not problems
        if ready != should_be_ready:
            failures.append(
                f"{target['registry']}: expected {'ready' if should_be_ready else 'NOT ready'} "
                f"({why})\n    text: {text[:70]!r}\n    problems: {problems}"
            )
    for failure in failures:
        print(f"FAIL {failure}")
    total = len(CASES) + len(PIN_CASES)
    print(f"{total - len(failures)}/{total} release-README cases pass")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
