#!/usr/bin/env python3
"""Package the native C shared library and test an extracted consumer."""

import argparse
import hashlib
import json
import os
import platform
import re
import struct
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


def version_tuple(value):
    return tuple(int(part) for part in value.split("."))


def inspect_elf(library, architecture, objdump="objdump"):
    header = library.read_bytes()[:20]
    expected = 62 if architecture == "x86_64" else 183
    if header[:6] != b"\x7fELF\x02\x01" or struct.unpack_from("<H", header, 18)[0] != expected:
        raise ValueError("ELF architecture does not match archive name")
    details = subprocess.check_output([objdump, "-p", str(library)], text=True)
    versions = set(re.findall(r"\bGLIBC_([0-9.]+)\b", details))
    if not versions or "GLIBC_PRIVATE" in details:
        raise ValueError("Expected public glibc symbol requirements")
    minimum = max(versions, key=version_tuple)
    if version_tuple(minimum) > (2, 34):
        raise ValueError(f"glibc requirement increased beyond 2.34: {minimum}")
    dependencies = re.findall(r"^\s*NEEDED\s+(\S+)", details, re.MULTILINE)
    if not dependencies or any("/" in name for name in dependencies):
        raise ValueError("Missing or nonportable ELF dependencies")
    return {"minimum_glibc": minimum, "dependencies": dependencies}


def inspect_macho(library, architecture):
    header = subprocess.check_output(["otool", "-hv", str(library)], text=True)
    if not re.search(r"\b" + ("X86_64" if architecture == "x86_64" else "ARM64") + r"\b", header):
        raise ValueError("Mach-O architecture does not match archive name")
    details = subprocess.check_output(["otool", "-l", str(library)], text=True)
    versions = re.findall(r"cmd LC_(?:BUILD_VERSION|VERSION_MIN_MACOSX)\n(?:(?!\nLoad command).)*?\n\s+(?:minos|version) ([0-9.]+)", details, re.DOTALL)
    expected = "10.12" if architecture == "x86_64" else "11.0"
    if len(versions) != 1 or version_tuple(versions[0]) != version_tuple(expected):
        raise ValueError(f"Expected macOS deployment target {expected}, got {versions}")
    links = subprocess.check_output(["otool", "-L", str(library)], text=True)
    dependencies = [line.strip().split(" (", 1)[0] for line in links.splitlines()[2:]]
    if not dependencies or any(not name.startswith(("/usr/lib/", "/System/Library/")) for name in dependencies):
        raise ValueError("Unexpected non-system macOS library dependency")
    return {"minimum_macos_load_command": versions[0], "dependencies": dependencies}


def inspect_pe(library):
    dumpbin = shutil.which("dumpbin")
    if not dumpbin:
        vswhere = Path(os.environ["ProgramFiles(x86)"]) / "Microsoft Visual Studio/Installer/vswhere.exe"
        matches = subprocess.check_output([
            str(vswhere), "-latest", "-products", "*", "-find",
            r"VC\Tools\MSVC\**\bin\Hostx64\x64\dumpbin.exe",
        ], text=True).splitlines()
        if not matches:
            raise ValueError("MSVC dumpbin was not found; install the C++ build tools")
        dumpbin = matches[0]
    header = subprocess.check_output([dumpbin, "/headers", str(library)], text=True)
    if not re.search(r"\b8664 machine", header):
        raise ValueError("Expected x86-64 PE library")
    subsystem = re.search(r"([0-9.]+) subsystem version", header)
    imports = subprocess.check_output([dumpbin, "/dependents", str(library)], text=True)
    dependencies = re.findall(r"^\s+(\S+\.dll)\s*$", imports, re.MULTILINE | re.IGNORECASE)
    if not subsystem or not dependencies or any("/" in name or "\\" in name for name in dependencies):
        raise ValueError("Missing or nonportable Windows DLL metadata")
    return {"pe_subsystem_version": subsystem[1], "dependencies": sorted(set(dependencies)),
            "minimum_windows": None}


def compatibility(library, target):
    family, architecture = target.split("-", 1)
    if family == "linux":
        requirements = inspect_elf(library, architecture)
    elif family == "macos":
        requirements = inspect_macho(library, architecture)
    else:
        requirements = inspect_pe(library)
    return {"platform": target, "requirements": requirements,
            "tested_host": platform.platform(),
            "runtime_scope": "Consumer acceptance runs on the recorded host only. Binary load requirements are not oldest-OS runtime verification."}


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
        if sys.platform == "darwin":
            library = package / "lib/libvanedb_capi.dylib"
            # Rust's default install name points into the build checkout.
            # A distributed consumer must load this archive's copy instead.
            subprocess.run(["install_name_tool", "-id", "@rpath/libvanedb_capi.dylib", str(library)], check=True)
            subprocess.run(["codesign", "--force", "--sign", "-", str(library)], check=True)
        for relative in ["README.md", "include/vanedb_rs_capi.h", "examples/CMakeLists.txt",
                         "examples/quickstart.c", "tests/acceptance.c"]:
            destination = package / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / "vanedb-capi" / relative, destination)
        shutil.copy2(ROOT / "LICENSE", package / "LICENSE")
        requirements = compatibility(package / "lib" / LIBRARIES[sys.platform][0], args.platform)
        (package / "compatibility.json").write_text(json.dumps(requirements, indent=2) + "\n")
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
        if sys.platform == "darwin":
            links = subprocess.check_output(["otool", "-L", str(build / "acceptance")], text=True)
            if "@rpath/libvanedb_capi.dylib" not in links or str(ROOT) in links:
                raise SystemExit(f"C consumer links outside the extracted archive:\n{links}")
    with archive.open("rb") as packaged:
        digest = hashlib.file_digest(packaged, "sha256").hexdigest()
    archive.with_suffix(".zip.sha256").write_text(f"{digest}  {archive.name}\n")
    print(f"Verified {archive}")


if __name__ == "__main__":
    main()
