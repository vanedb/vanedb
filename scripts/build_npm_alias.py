#!/usr/bin/env python3
"""Assemble the unscoped `vanedb` npm package.

npm's scoped and unscoped namespaces are separate: owning the `vanedb`
organisation reserves `@vanedb/*` and leaves the bare word open to anyone. This
package holds it, and makes `npm install vanedb` — the name a reader is most
likely to type — resolve to our code rather than to whatever a stranger might
publish there.

It is a real re-export, not a placeholder: npm's policy is fine with a
functional alias and objects to empty name-holding. It depends on
`@vanedb/wasm` at an exact matching version, so the two can never drift.

`init` stays in the public contract even though it is a no-op on Node. That is
what lets this entry point later dispatch to a native binding without breaking
anyone who wrote `await init()`.
"""

import argparse
import json
import pathlib
import re
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SOURCE = ROOT / "npm/vanedb"
WASM_MANIFEST = ROOT / "vanedb-wasm/Cargo.toml"


def crate_version():
    text = WASM_MANIFEST.read_text(encoding="utf-8")
    match = re.search(r'(?m)^version = "([^"]+)"', text)
    if not match:
        raise SystemExit(f"no version in {WASM_MANIFEST}")
    return match.group(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, default=ROOT / "target/npm/vanedb")
    parser.add_argument(
        "--alias-version",
        help="version for this package. Defaults to the wasm crate's, which is "
        "the lockstep case. Override only for the bootstrap publish that "
        "reserves the name, so the version people install is never one "
        "published by hand.",
    )
    args = parser.parse_args()

    wasm_version = crate_version()
    version = args.alias_version or wasm_version

    out = args.output
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    for name in ("index.mjs", "index.cjs", "index.d.ts", "README.md"):
        shutil.copy2(SOURCE / name, out / name)
    shutil.copy2(ROOT / "LICENSE", out / "LICENSE")

    package = {
        "name": "vanedb",
        "version": version,
        "description": "Vector search in Node.js and the browser. Re-exports @vanedb/wasm.",
        "license": "MIT",
        "repository": {"type": "git", "url": "git+https://github.com/vanedb/vanedb.git"},
        "keywords": ["vector", "search", "embeddings", "wasm", "hnsw", "database"],
        "homepage": "https://github.com/vanedb/vanedb#readme",
        "type": "module",
        "types": "./index.d.ts",
        "exports": {
            ".": {
                "types": "./index.d.ts",
                "import": "./index.mjs",
                "require": "./index.cjs",
            },
            "./package.json": "./package.json",
        },
        # Exact, not a range: the alias exists to be indistinguishable from the
        # package it forwards to, and a range would let them drift.
        "dependencies": {"@vanedb/wasm": wasm_version},
        "files": ["index.mjs", "index.cjs", "index.d.ts", "README.md", "LICENSE"],
    }
    (out / "package.json").write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")

    print(f"vanedb {version} -> depends on @vanedb/wasm {wasm_version}")
    print(f"npm alias assembled at {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
