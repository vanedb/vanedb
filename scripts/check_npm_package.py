#!/usr/bin/env python3
"""Assemble, pack and consume the npm package exactly as a user would.

`npm pack` on the assembled directory, `npm install` of that tarball into a
throwaway project, then the consumer test running from inside it — so the bare
specifier resolves through the package's own `exports` conditions rather than a
path. Every packaging failure found while building this was invisible until the
package was installed and imported.
"""

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "vanedb-wasm/tests/npm_package.mjs"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path, nargs="?",
                        default=ROOT / "target/npm/vanedb-wasm",
                        help="assembled package directory")
    args = parser.parse_args()
    package = args.package.resolve()
    manifest = json.loads((package / "package.json").read_text())

    with tempfile.TemporaryDirectory(prefix="vanedb-npm-check-") as tmp:
        consumer = Path(tmp)
        (consumer / "package.json").write_text(
            json.dumps({"name": "consumer", "version": "1.0.0",
                        "type": "module", "private": True}) + "\n"
        )
        env_cache = consumer / "cache"
        env = {"npm_config_cache": str(env_cache), "PATH": __import__("os").environ["PATH"],
               "HOME": str(consumer)}
        packed = subprocess.run(
            ["npm", "pack", str(package), "--pack-destination", str(consumer),
             "--ignore-scripts", "--json"],
            cwd=consumer, env=env, check=True, text=True, capture_output=True,
        )
        tarball = consumer / json.loads(packed.stdout)[0]["filename"]
        subprocess.run(
            ["npm", "install", str(tarball), "--no-audit", "--no-fund", "--ignore-scripts"],
            cwd=consumer, env=env, check=True, capture_output=True,
        )
        shutil.copy2(TEST, consumer / "npm_package.mjs")
        subprocess.run(["node", "npm_package.mjs"], cwd=consumer, env=env, check=True)

    print(f"npm package {manifest['name']}@{manifest['version']}: consumable")


if __name__ == "__main__":
    main()
