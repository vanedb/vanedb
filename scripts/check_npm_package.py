#!/usr/bin/env python3
"""Assemble, pack and consume the npm package exactly as a user would.

`npm pack` on the assembled directory, `npm install` of that tarball into a
throwaway project, then the consumer test running from inside it — so the bare
specifier resolves through the package's own `exports` conditions rather than a
path. Every packaging failure found while building this was invisible until the
package was installed and imported.
"""

import argparse
import gzip
import json
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "vanedb-wasm/tests/npm_package.mjs"
FIXTURE = ROOT / "vanedb/tests/fixtures/vndb_graph/l2_rng1.vndb"


def gzipped_len(data: bytes) -> int:
    return len(gzip.compress(data, compresslevel=9))


def report_sizes(package: Path) -> None:
    """Print gzipped and unpacked sizes of the wasm module and JS layer."""
    if package.is_dir():
        root = package
        wasm = (root / "web/vanedb_wasm_bg.wasm").read_bytes()
        js_files = sorted(
            p for p in root.rglob("*")
            if p.suffix in {".js", ".mjs", ".cjs"} and p.is_file()
        )
        js = b"".join(p.read_bytes() for p in js_files)
        unpacked = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    else:
        with tarfile.open(package) as archive:
            names = archive.getnames()
            wasm = archive.extractfile("package/web/vanedb_wasm_bg.wasm").read()
            js_parts = []
            unpacked = 0
            for info in archive.getmembers():
                if not info.isfile():
                    continue
                unpacked += info.size
                if info.name.endswith((".js", ".mjs", ".cjs")):
                    js_parts.append(archive.extractfile(info).read())
            js = b"".join(js_parts)
            if "package/web/vanedb_wasm_bg.wasm" not in names:
                raise SystemExit("packed tarball is missing the web wasm module")

    print(
        "bundle sizes: "
        f"wasm {gzipped_len(wasm)} gzipped / {len(wasm)} unpacked; "
        f"js {gzipped_len(js)} gzipped / {len(js)} unpacked; "
        f"package files {unpacked} unpacked"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path, nargs="?",
                        default=ROOT / "target/npm/vanedb-wasm",
                        help="assembled package directory or packed .tgz")
    args = parser.parse_args()
    package = args.package.resolve()
    if package.is_dir():
        manifest = json.loads((package / "package.json").read_text())
    else:
        with tarfile.open(package) as archive:
            manifest = json.load(archive.extractfile("package/package.json"))

    report_sizes(package)

    with tempfile.TemporaryDirectory(prefix="vanedb-npm-check-") as tmp:
        consumer = Path(tmp)
        (consumer / "package.json").write_text(
            json.dumps({"name": "consumer", "version": "1.0.0",
                        "type": "module", "private": True}) + "\n"
        )
        env_cache = consumer / "cache"
        env = {"npm_config_cache": str(env_cache), "PATH": __import__("os").environ["PATH"],
               "HOME": str(consumer)}
        tarball = package
        if package.is_dir():
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
        shutil.copy2(TEST.with_name("atomic_write_child.cjs"), consumer / "atomic_write_child.cjs")
        shutil.copy2(FIXTURE, consumer / "l2_rng1.vndb")
        subprocess.run(["node", "npm_package.mjs"], cwd=consumer, env=env, check=True)

    print(f"npm package {manifest['name']}@{manifest['version']}: consumable")


if __name__ == "__main__":
    main()
