#!/usr/bin/env python3
"""Refuse a release tag whose packaged README still describes the checkout.

`vanedb/Cargo.toml` declares `readme = "README.md"`, so that file is inside the
`.crate` and is what crates.io renders. `vanedb-py/pyproject.toml` does the
same, making its README the core-metadata description of all the wheels and the
sdist, and the PyPI project page.

Neither registry permits a re-upload. A checkout-only install line in either
file is therefore permanent for that version: for as long as it is the newest
release, the registry page tells a visitor nothing is published while they are
standing on the published package.

RELEASING.md step 5 says to make this edit in the release commit. Nothing
enforced it, which is why this exists -- it is the only step in the release
whose mistake cannot be corrected afterwards.
"""

import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Per tag prefix: the packaged file, a pattern proving the published form is
# present, and patterns proving the checkout-only form is gone. Both directions
# matter: adding the published line without removing the checkout line leaves
# the misleading sentence on the registry page, which is the actual harm.
TARGETS = {
    "vanedb-crate-v": {
        "readme": "vanedb/README.md",
        "registry": "crates.io",
        "published": [r"cargo add vanedb\b", r'vanedb\s*=\s*"'],
        "checkout_only": [r'vanedb\s*=\s*\{\s*path\s*='],
    },
    "vanedb-v": {
        "readme": "vanedb-py/README.md",
        "registry": "PyPI",
        "published": [r"pip install vanedb\b"],
        "checkout_only": [r"pip install \./vanedb-py", r"do not assume a published"],
    },
}


def target_for(tag):
    """Longest prefix wins: `vanedb-crate-v` also starts with `vanedb-`."""
    matches = [p for p in TARGETS if tag.startswith(p)]
    if not matches:
        return None
    return TARGETS[max(matches, key=len)]


def check(tag):
    target = target_for(tag)
    if target is None:
        print(f"{tag}: no packaged README is tied to this tag; nothing to check")
        return 0

    path = ROOT / target["readme"]
    text = path.read_text()
    problems = []

    if not any(re.search(p, text) for p in target["published"]):
        problems.append(
            f"no published-install line found. One of these must appear: "
            + ", ".join(repr(p) for p in target["published"])
        )
    for pattern in target["checkout_only"]:
        if re.search(pattern, text):
            problems.append(f"still describes the checkout: matched {pattern!r}")

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
