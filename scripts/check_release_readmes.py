#!/usr/bin/env python3
"""Refuse a release tag whose packaged README still describes the checkout.

Each published artifact carries a README that the registry renders, and no
registry here permits a re-upload: a "not published yet" line is permanent for
that version. It has already happened once — `@vanedb/wasm@0.1.0` shipped
saying "Nothing is published yet", which is what this exists to stop.

**This is a tripwire, not a proof.** It matches prose, and prose can be
reworded past it. It reliably catches the specific mistake of tagging a release
with the pre-publication README still in place, which is the mistake that has
actually occurred. It cannot certify that a rewritten README is honest. Run it
locally before pushing a tag rather than discovering a false positive after.
"""

import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Language asserting the package is not available. Checked in every target: the
# earlier version listed one literal sentence per registry, so any rewording
# disarmed it silently. This is a class, deliberately broad -- a false positive
# costs a re-tag, a false negative is permanent on a registry page.
UNPUBLISHED = [
    # Publication-specific phrasings only. Bare "not published" / "not
    # available" match legitimate prose about other things -- the Python README
    # says the C++ bindings "are not published" (true, different package) and
    # the wasm README says features "are not available in WebAssembly" (an API
    # limit). Both are correct sentences a broad class would reject.
    r"not (?:yet )?(?:on|published to|available on) (?:crates\.io|PyPI|npm)",
    r"nothing is published",
    r"no (?:release|published version) (?:yet|on)",
    r"do not assume a published",
    r"until (?:then|it is published|release)",
    r"is not published (?:yet|to)",
]

# `(?![-\w])` so `vanedb` does not match inside `vanedb-capi` or `vanedb-cpp`,
# which are different packages that must not satisfy this on their own.
NAME = r"(?![-\w])"

TARGETS = {
    "vanedb-crate-v": {
        "readme": "vanedb/README.md",
        "registry": "crates.io",
        # `cargo add`, a plain version requirement, or the table form a crate
        # with features needs. The table form was rejected by the first version.
        "published": [
            rf"cargo add vanedb{NAME}",
            rf"vanedb{NAME}\s*=\s*[\"']",
            rf"vanedb{NAME}\s*=\s*\{{[^}}]*version\s*=",
        ],
        "forbidden": [r"path\s*=\s*[\"'][^\"']*vanedb"],
    },
    "vanedb-v": {
        "readme": "vanedb-py/README.md",
        "registry": "PyPI",
        "published": [rf"pip install (?:--upgrade )?[\"']?vanedb{NAME}"],
        "forbidden": [r"pip install \.", r"pip install [^\s\"']*/"],
    },
    # Packaged by `scripts/build_npm_package.py` and rendered by npmjs.com.
    "vanedb-wasm-v": {
        "readme": "vanedb-wasm/README.md",
        "registry": "npm",
        "published": [r"npm (?:install|i) @vanedb/wasm"],
        "forbidden": [],
    },
}


def problems_in(target, text):
    """Why `text` is not ready to ship for `target`; empty means ready.

    Split out so the tests can drive it with fixtures instead of the real
    READMEs, which change under it.
    """
    problems = []

    if not any(re.search(p, text, re.I) for p in target["published"]):
        problems.append(
            "no published-install line. One of these must appear: "
            + ", ".join(repr(p) for p in target["published"])
        )
    for pattern in UNPUBLISHED + target["forbidden"]:
        match = re.search(pattern, text, re.I)
        if match:
            problems.append(f"still says the package is unpublished: {match.group(0)!r}")

    return problems


def check(tag):
    target = next((t for p, t in TARGETS.items() if tag.startswith(p)), None)
    if target is None:
        print(f"{tag}: no packaged README is tied to this tag; nothing to check")
        return 0

    problems = problems_in(target, (ROOT / target["readme"]).read_text(encoding="utf-8"))
    if problems:
        print(f"{target['readme']} is not ready to ship to {target['registry']}:")
        for problem in problems:
            print(f"  - {problem}")
        print(
            f"\nThis file is packaged into the artifact and rendered on "
            f"{target['registry']}, which does not permit a re-upload. See "
            f"RELEASING.md step 5."
        )
        return 1

    print(f"{target['readme']}: carries the published-install form for {target['registry']}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag", help="the release tag, e.g. vanedb-crate-v0.1.0")
    sys.exit(check(parser.parse_args().tag))
