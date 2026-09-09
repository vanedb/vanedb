#!/usr/bin/env python3
"""Install the alias into a throwaway project and use it through both module systems.

The package is three lines of re-export, which is exactly the kind of thing that
looks obviously correct and silently forwards nothing. This installs the packed
tarball alongside the real `@vanedb/wasm` and runs a search through `vanedb`.
"""

import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
ALIAS = ROOT / "target/npm/vanedb"
WASM = ROOT / "target/npm/vanedb-wasm"

ESM = """
import init, { FlatIndex, version } from 'vanedb';
await init();
const index = new FlatIndex(3, 'l2');
index.add(7n, new Float32Array([1, 0, 0]));
const hits = index.search(new Float32Array([1, 0, 0]), 1);
if (hits.ids[0] !== 7n) throw new Error(`ESM: got id ${hits.ids[0]}`);
if (typeof version() !== 'string') throw new Error('ESM: version() is not a string');
hits.free(); index.free();
console.log('ESM ok');
"""

CJS = """
const vanedb = require('vanedb');
const index = new vanedb.FlatIndex(3, 'l2');
index.add(9n, new Float32Array([0, 1, 0]));
const hits = index.search(new Float32Array([0, 1, 0]), 1);
if (hits.ids[0] !== 9n) throw new Error(`CJS: got id ${hits.ids[0]}`);
hits.free(); index.free();
console.log('CJS ok');
"""


def run(command, cwd):
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    if result.returncode:
        print(result.stdout[-2000:])
        print(result.stderr[-2000:], file=sys.stderr)
        raise SystemExit(f"failed: {' '.join(command)}")
    return result.stdout


def main():
    for path in (ALIAS, WASM):
        if not path.is_dir():
            raise SystemExit(f"{path} is missing; run the build scripts first")

    with tempfile.TemporaryDirectory() as tmp:
        project = pathlib.Path(tmp)
        (project / "package.json").write_text(
            json.dumps({"name": "alias-check", "private": True, "type": "module"}) + "\n")
        # Pack first. `npm install <dir>` links the directory, so the alias
        # would resolve `@vanedb/wasm` from its real path in target/ and never
        # reach this project's node_modules -- passing or failing for reasons
        # that have nothing to do with the published package.
        tarballs = []
        for source in (WASM, ALIAS):
            out = run(["npm", "pack", str(source), "--pack-destination", str(project),
                       "--ignore-scripts"], project)
            tarballs.append(str(project / out.strip().splitlines()[-1]))
        # The alias depends on @vanedb/wasm by exact version, so install the
        # local build of it too: this must test the tarballs about to be
        # published, not whatever the registry currently holds.
        run(["npm", "install", "--no-audit", "--no-fund", *tarballs], project)

        (project / "esm.mjs").write_text(ESM)
        (project / "cjs.cjs").write_text(CJS)
        print(" ", run(["node", "esm.mjs"], project).strip())
        print(" ", run(["node", "cjs.cjs"], project).strip())

    print("npm alias vanedb: ESM and CommonJS consumers both OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
