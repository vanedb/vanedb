#!/usr/bin/env python3
"""Build the crate archive, run its isolated tests, and retain release artifacts."""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import tempfile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-dirty", action="store_true", help="verify local uncommitted work")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    metadata = json.loads(subprocess.check_output(
        ["cargo", "metadata", "--no-deps", "--format-version", "1", "--locked"],
        cwd=root, text=True,
    ))
    package = next(item for item in metadata["packages"] if item["name"] == "vanedb")
    target = Path(metadata["target_directory"])
    name = f"vanedb-{package['version']}"
    command = ["cargo", "package", "-p", "vanedb", "--locked", "--no-verify"]
    if args.allow_dirty:
        command.append("--allow-dirty")
    subprocess.run(command, cwd=root, check=True)
    archive = target / "package" / f"{name}.crate"
    with tempfile.TemporaryDirectory(prefix="vanedb-package-") as directory:
        with tarfile.open(archive) as source:
            contents = sorted(str(PurePosixPath(member.name).relative_to(name))
                              for member in source.getmembers() if member.isfile())
            source.extractall(directory, filter="data")
        unpacked = Path(directory) / name
        subprocess.run(
            ["cargo", "test", "--locked", "--features", "disk"],
            cwd=unpacked, env=dict(os.environ, CARGO_TARGET_DIR=str(target)), check=True,
        )
    artifacts = target / "crate-artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    shutil.copy2(archive, artifacts / archive.name)
    (artifacts / "package-contents.txt").write_text("\n".join(contents) + "\n")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (artifacts / "SHA256SUMS").write_text(f"{digest}  {archive.name}\n")
    print(f"Verified {archive.name}; artifacts: {artifacts}")


if __name__ == "__main__":
    main()
