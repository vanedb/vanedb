#!/usr/bin/env python3
"""Pack generated WASM distributions and test the installed Node artifact."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=["nodejs", "web"])
    parser.add_argument("package", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "target/wasm-artifacts")
    args = parser.parse_args()
    output = args.output.resolve() / args.target
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vanedb-wasm-package-") as temporary:
        env = {**os.environ, "npm_config_cache": str(Path(temporary) / "cache")}
        packed = subprocess.run(
            ["npm", "pack", "--json", "--ignore-scripts", "--pack-destination", str(output)],
            cwd=args.package, env=env, check=True, text=True, capture_output=True,
        )
        archive = output / json.loads(packed.stdout)[0]["filename"]
        with tarfile.open(archive) as package:
            required = {f"package/{name}" for name in [
                "package.json", "README.md", "LICENSE", "vanedb_wasm.js",
                "vanedb_wasm.d.ts", "vanedb_wasm_bg.wasm",
            ]}
            missing = required - set(package.getnames())
            if missing:
                raise SystemExit(f"Incomplete {args.target} package: {sorted(missing)}")
        if args.target == "nodejs":
            subprocess.run(
                ["npm", "install", "--offline", "--ignore-scripts", "--no-audit", "--no-fund",
                 "--package-lock=false", "--prefix", temporary, str(archive)],
                cwd=temporary, env=env, check=True,
            )
            installed = Path(temporary) / "node_modules/vanedb-wasm"
            subprocess.run(["node", str(ROOT / "vanedb-wasm/tests/node.cjs"), str(installed)],
                           cwd=temporary, check=True)
        print(f"Verified {args.target} package: {archive}")


if __name__ == "__main__":
    main()
