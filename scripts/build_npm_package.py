#!/usr/bin/env python3
"""Merge the two wasm-pack outputs into one publishable npm package.

wasm-pack emits a separate package per target and names them both
`vanedb-wasm`, so they cannot both go to npm under that name. It also emits
two *different contracts*: the Node build is CommonJS that loads the module
synchronously, while the web build is ESM whose default export is an async
`init` that must resolve before any class is touched.

Conditional `exports` alone would therefore route one specifier to two
different APIs. This adds a small wrapper so `init()` exists in both — a
no-op on Node, the real loader in a browser — and the same source works
either way:

    import init, { FlatIndex } from '@vanedb/wasm';
    await init();                  // no-op on Node
    const index = new FlatIndex(3, 'l2');
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CRATE = ROOT / "vanedb-wasm"

NODE_CJS = """// Node, `require`: wasm-pack's nodejs target loads the module synchronously,
// so there is nothing to await. `init` exists only so browser and Node source
// can be identical.
const bindings = require('./vanedb_wasm.js');

module.exports = Object.assign({}, bindings, {
  default: async function init() { return bindings; },
  initSync: function initSync() { return bindings; },
});
"""

# Node, `import`. An ESM file re-exporting a CommonJS module cannot rely on
# Node's named-export detection — `import { ApproxIndex } from '@vanedb/wasm'`
# fails with "Named export not found" when the CJS module assigns its exports
# dynamically, which wasm-pack's output does. Each name is therefore re-bound
# explicitly, and the list is generated from the build rather than hand-written
# so a new class cannot be silently omitted.
NODE_MJS_HEADER = """import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const bindings = require('./vanedb_wasm.js');

export default async function init() { return bindings; }
export function initSync() { return bindings; }
"""

WEB_SHIM = """// Browser: re-export wasm-pack's web target unchanged. Its default export is
// the async loader, which must resolve before any class is constructed.
export { default, initSync } from './vanedb_wasm.js';
export * from './vanedb_wasm.js';
"""


def exported_names(node_js: Path) -> list[str]:
    """The names wasm-pack's nodejs target assigns to `exports`."""
    names = []
    for line in node_js.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("exports.") and "=" in stripped:
            name = stripped[len("exports."):].split("=", 1)[0].strip()
            if name.isidentifier() and name not in names:
                names.append(name)
    return names


def build(target: str, out: str) -> Path:
    subprocess.run(
        ["wasm-pack", "build", "--target", target, "--out-dir", out, "--locked"],
        cwd=CRATE, check=True,
    )
    return CRATE / out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # `target/npm/vanedb-wasm` is a filesystem path, not the package name.
    # Both workflows glob `target/npm/vanedb-wasm-*.tgz`, which keeps matching
    # because `npm pack` flattens a scope: `@vanedb/wasm` packs to
    # `vanedb-wasm-0.1.0.tgz`. Do not "fix" either to look scoped.
    parser.add_argument("--output", type=Path, default=ROOT / "target/npm/vanedb-wasm")
    parser.add_argument("--skip-build", action="store_true",
                        help="reuse existing pkg/ and pkg-web/ directories")
    args = parser.parse_args()

    node_dir = CRATE / "pkg" if args.skip_build else build("nodejs", "pkg")
    web_dir = CRATE / "pkg-web" if args.skip_build else build("web", "pkg-web")

    out = args.output.resolve()
    if out.exists():
        shutil.rmtree(out)
    (out / "node").mkdir(parents=True)
    (out / "web").mkdir(parents=True)

    payload = ["vanedb_wasm.js", "vanedb_wasm.d.ts", "vanedb_wasm_bg.wasm"]
    for name in payload:
        shutil.copy2(node_dir / name, out / "node" / name)
        shutil.copy2(web_dir / name, out / "web" / name)
    # The Node target also emits a .d.ts for the raw module.
    for extra in ("vanedb_wasm_bg.wasm.d.ts",):
        if (node_dir / extra).exists():
            shutil.copy2(node_dir / extra, out / "node" / extra)
        if (web_dir / extra).exists():
            shutil.copy2(web_dir / extra, out / "web" / extra)

    names = exported_names(node_dir / "vanedb_wasm.js")
    if not names:
        raise SystemExit("no exports found in the Node build; the shim would be empty")

    # The two targets must expose the same surface, or one specifier would give
    # a browser and Node different APIs.
    web_names = {
        line.split("class ")[1].split()[0].rstrip("{")
        for line in (web_dir / "vanedb_wasm.js").read_text().splitlines()
        if line.startswith("export class ")
    }
    web_names |= {
        line.split("function ")[1].split("(")[0]
        for line in (web_dir / "vanedb_wasm.js").read_text().splitlines()
        if line.startswith("export function ")
    }
    missing = set(names) - web_names
    if missing:
        raise SystemExit(f"the web build is missing {sorted(missing)}, exported by Node")

    # The root package.json declares "type": "module", which would make Node
    # parse wasm-pack's CommonJS output in node/ as ESM and fail on its first
    # `exports.` assignment. A nested package.json scopes that directory back
    # to CommonJS; index.mjs keeps its explicit extension and stays ESM.
    (out / "node" / "package.json").write_text('{ "type": "commonjs" }\n')
    (out / "node" / "index.cjs").write_text(NODE_CJS)
    (out / "node" / "index.mjs").write_text(
        NODE_MJS_HEADER
        + "".join(f"export const {n} = bindings.{n};\n" for n in names)
    )
    (out / "web" / "index.js").write_text(WEB_SHIM)
    print(f"  re-exported from the Node build: {', '.join(names)}")
    # One .d.ts serves both: the classes are identical, and the web build's
    # extra init types are what the shim gives Node as well.
    shutil.copy2(web_dir / "vanedb_wasm.d.ts", out / "index.d.ts")

    for doc in ("README.md", "LICENSE"):
        shutil.copy2(CRATE / doc, out / doc)

    generated = json.loads((node_dir / "package.json").read_text())
    package = {
        # Scoped, so a future native binding can take `@vanedb/node` without
        # competing with this one for a bare name. `--access public` in the
        # publish workflow is required: npm defaults scoped packages private.
        "name": "@vanedb/wasm",
        "version": generated["version"],
        "description": generated["description"],
        "license": generated["license"],
        "repository": generated["repository"],
        "keywords": ["vector", "search", "embeddings", "wasm", "hnsw", "database"],
        "homepage": "https://github.com/vanedb/vanedb#readme",
        "type": "module",
        "types": "./index.d.ts",
        # `node` before `browser` and `default`: Node honours the first match,
        # bundlers honour `browser`, and everything else falls through to the
        # web build, which is the safe default for an unknown runtime.
        "exports": {
            ".": {
                "types": "./index.d.ts",
                "node": {
                    "import": "./node/index.mjs",
                    "require": "./node/index.cjs",
                },
                "browser": "./web/index.js",
                "default": "./web/index.js",
            },
            "./package.json": "./package.json",
        },
        "main": "./node/index.cjs",
        "module": "./web/index.js",
        "files": ["node/", "web/", "index.d.ts", "README.md", "LICENSE"],
        "sideEffects": ["./web/index.js", "./web/vanedb_wasm.js"],
        "engines": {"node": ">=18"},
    }
    (out / "package.json").write_text(json.dumps(package, indent=2) + "\n")
    print(f"npm package assembled at {out}")


if __name__ == "__main__":
    main()
