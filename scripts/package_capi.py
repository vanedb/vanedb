#!/usr/bin/env python3
"""Package the native C shared library and test an extracted consumer."""

import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import tomllib
import zipfile

ROOT = Path(__file__).resolve().parents[1]
LIBRARIES = {
    "linux": ["libvanedb_capi.so"],
    "darwin": ["libvanedb_capi.dylib"],
    "win32": ["vanedb_capi.dll", "vanedb_capi.dll.lib"],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", required=True, choices=[
        "linux-x86_64", "linux-aarch64", "macos-x86_64", "macos-aarch64", "windows-x86_64"
    ])
    parser.add_argument("--library-dir", type=Path, default=ROOT / "target/release")
    parser.add_argument("--output", type=Path, default=ROOT / "target/c-artifacts")
    args = parser.parse_args()
    version = tomllib.loads((ROOT / "vanedb-capi/Cargo.toml").read_text())["package"]["version"]
    name = f"vanedb-capi-{version}-{args.platform}"
    args.output.mkdir(parents=True, exist_ok=True)
    archive = args.output.resolve() / f"{name}.zip"
    with tempfile.TemporaryDirectory(prefix="vanedb-c-package-") as temporary:
        directory = Path(temporary)
        package = directory / "stage" / name
        (package / "lib").mkdir(parents=True)
        for library in LIBRARIES[sys.platform]:
            shutil.copy2(args.library_dir / library, package / "lib" / library)
        for relative in ["README.md", "include/vanedb_rs_capi.h", "examples/CMakeLists.txt",
                         "examples/quickstart.c", "tests/acceptance.c"]:
            destination = package / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / "vanedb-capi" / relative, destination)
        shutil.copy2(ROOT / "LICENSE", package / "LICENSE")
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
            for source in sorted(package.rglob("*")):
                if source.is_file():
                    output.write(source, source.relative_to(package.parent))
        # Compile and execute using only the archive's files and libraries.
        shutil.rmtree(package.parent)
        with zipfile.ZipFile(archive) as packaged:
            packaged.extractall(directory / "extracted")
        extracted = directory / "extracted" / name
        build = directory / "build"
        for command in [
            ["cmake", "-S", str(extracted / "examples"), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release"],
            ["cmake", "--build", str(build), "--config", "Release"],
            ["ctest", "--test-dir", str(build), "--build-config", "Release", "--output-on-failure"],
        ]:
            subprocess.run(command, cwd=directory, check=True)
    with archive.open("rb") as packaged:
        digest = hashlib.file_digest(packaged, "sha256").hexdigest()
    archive.with_suffix(".zip.sha256").write_text(f"{digest}  {archive.name}\n")
    print(f"Verified {archive}")


if __name__ == "__main__":
    main()
