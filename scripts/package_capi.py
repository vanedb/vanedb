#!/usr/bin/env python3
"""Package the native C libraries and test consumers against the extracted archive.

The archive (RFC 0002 stage 1) carries, for one platform:

    include/vanedb_rs_capi.h          the generated header
    lib/<shared library>              stripped; exports exactly vanedb_rs_*
    lib/<static library>              isolated API globals plus native import glue
    lib/cmake/vanedb/*.cmake          find_package(vanedb): vanedb::shared, vanedb::static
    lib/pkgconfig/vanedb.pc           pkg-config --cflags --libs vanedb
    examples/, consumers/, tests/     the consumer projects CI runs
    compatibility.json                inspected binary requirements

The static target's link line and Libs.private come from
`cargo rustc --print native-static-libs`, run here unless --native-static-libs
is given. After zipping, the archive is extracted somewhere else and the
examples, the CMake consumer and (Linux, macOS) the pkg-config consumer are
built and run from it alone.
"""

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
sys.path.insert(0, str(ROOT / "scripts"))
import capi_exports  # noqa: E402
import capi_windows_static  # noqa: E402

PROFILE = "capi"
LIBRARIES = {
    "linux": {"shared": "libvanedb_capi.so", "static": "libvanedb_capi.a", "import": None},
    "darwin": {"shared": "libvanedb_capi.dylib", "static": "libvanedb_capi.a", "import": None},
    "win32": {"shared": "vanedb_capi.dll", "static": "vanedb_capi.lib", "import": "vanedb_capi.dll.lib"},
}
PACKAGED_SOURCES = [
    "README.md", "include/vanedb_rs_capi.h",
    "examples/CMakeLists.txt", "examples/quickstart.c", "examples/ctypes_quickstart.py",
    "tests/acceptance.c",
    "consumers/cmake/CMakeLists.txt", "consumers/pkgconfig/Makefile",
    "consumers/cmake/independent.rs",
]


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
    dumpbin = capi_exports.dumpbin()
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


def compatibility(library, target, native_static_libs):
    family, architecture = target.split("-", 1)
    if family == "linux":
        requirements = inspect_elf(library, architecture)
    elif family == "macos":
        requirements = inspect_macho(library, architecture)
    else:
        requirements = inspect_pe(library)
    return {"platform": target, "requirements": requirements,
            "native_static_libs": native_static_libs,
            "tested_host": platform.platform(),
            "runtime_scope": "Consumer acceptance runs on the recorded host only. Binary load requirements are not oldest-OS runtime verification."}


# --- the static target's link line -----------------------------------------

def native_static_libs():
    """What `cargo rustc --print native-static-libs` reports for the staticlib.

    rustc prints the note while producing the staticlib, so this relinks the
    crate under the shipped profile with the extra flag; the artifacts are
    the same ones packaged below.
    """
    command = ["cargo", "rustc", "-p", "vanedb-capi", "--profile", PROFILE, "--locked",
               "--color", "never", "--", "--print", "native-static-libs"]
    env = dict(os.environ, CARGO_TERM_COLOR="never")
    completed = subprocess.run(command, cwd=ROOT, check=True, text=True, capture_output=True, env=env)
    line = parse_native_static_libs(completed.stderr)
    if line is None:
        raise SystemExit("cargo rustc did not report native-static-libs:\n" + completed.stderr)
    return line


ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def parse_native_static_libs(stderr):
    """The link line from rustc's `note: native-static-libs: ...`, or None.

    CI sets CARGO_TERM_COLOR=always, under which the note ends in a colour
    reset and the last library became `c\x1b[0m`, which no linker can find.
    The command above asks for no colour; the escapes are stripped anyway so
    a wrapper that recolours the output cannot bring the defect back.
    """
    plain = ANSI_ESCAPE.sub("", stderr)
    match = re.search(r"native-static-libs:\s*(.*)", plain)
    return match.group(1).strip() if match else None


def link_tokens(native):
    """`-framework X` is one item; everything else splits on whitespace."""
    words = native.split()
    tokens = []
    while words:
        word = words.pop(0)
        if word == "-framework" and words:
            tokens.append(f"-framework {words.pop(0)}")
        else:
            tokens.append(word)
    # rustc repeats libraries; the order is what matters, not the count.
    seen = []
    for token in tokens:
        if token not in seen:
            seen.append(token)
    return seen


def cmake_link_items(native):
    """INTERFACE_LINK_LIBRARIES items. `-lX` becomes `X`; MSVC's
    `/defaultlib:msvcrt` is dropped because CMake selects the C runtime
    itself and a slash-prefixed item would be read as a file path."""
    items = []
    for token in link_tokens(native):
        if token.startswith("-l"):
            items.append(token[2:])
        elif token.lower().startswith("/defaultlib:"):
            continue
        else:
            items.append(token)
    return ";".join(items)


def pkgconfig_libs_private(native):
    return " ".join(t for t in link_tokens(native) if not t.lower().startswith("/defaultlib:"))


def render(template, **values):
    text = (ROOT / "vanedb-capi/packaging" / template).read_text(encoding="utf-8")
    for key, value in values.items():
        text = text.replace(f"@{key}@", str(value))
    leftover = re.findall(r"@[A-Z_]+@", text)
    if leftover:
        raise SystemExit(f"{template}: unfilled placeholders {leftover}")
    return text


def write_package_files(package, version, native):
    header = (ROOT / "vanedb-capi/include/vanedb_rs_capi.h").read_text(encoding="utf-8")
    abi_version = re.search(r"^#define VANEDB_RS_ABI_VERSION (\d+)$", header, re.MULTILINE)
    if not abi_version:
        raise SystemExit("the header does not define VANEDB_RS_ABI_VERSION")
    core = version.split("-", 1)[0]
    major, minor, _patch = core.split(".")
    names = LIBRARIES[sys.platform]
    import_property = ""
    if names["import"]:
        import_property = ('set_target_properties(vanedb::shared PROPERTIES\n'
                           f'  IMPORTED_IMPLIB "${{_vanedb_prefix}}/lib/{names["import"]}")\n')
    cmake_dir = package / "lib/cmake/vanedb"
    cmake_dir.mkdir(parents=True)
    (cmake_dir / "vanedbConfig.cmake").write_text(
        render("vanedbConfig.cmake.in", VERSION=version, ABI_VERSION=abi_version.group(1)), encoding="utf-8")
    (cmake_dir / "vanedbConfigVersion.cmake").write_text(
        render("vanedbConfigVersion.cmake.in", VERSION_CORE=core, VERSION_MAJOR=major, VERSION_MINOR=minor),
        encoding="utf-8")
    (cmake_dir / "vanedbTargets.cmake").write_text(
        render("vanedbTargets.cmake.in", SHARED_LIBRARY=names["shared"], STATIC_LIBRARY=names["static"],
               IMPORT_LIBRARY_PROPERTY=import_property, STATIC_LINK_LIBRARIES=cmake_link_items(native)),
        encoding="utf-8")
    pc_dir = package / "lib/pkgconfig"
    pc_dir.mkdir(parents=True)
    (pc_dir / "vanedb.pc").write_text(
        render("vanedb.pc.in", VERSION=version, LIBS_PRIVATE=pkgconfig_libs_private(native)), encoding="utf-8")


# --- symbol hygiene -----------------------------------------------------------

def strip_shared(library):
    """Debug info and local symbols go; the dynamic export table stays.
    MSVC DLLs carry their symbols in the separate .pdb, so nothing to do."""
    if sys.platform == "linux":
        subprocess.run(["strip", "--strip-unneeded", str(library)], check=True)
    elif sys.platform == "darwin":
        subprocess.run(["strip", "-x", str(library)], check=True)


APPLE_ARCH = {"macos-aarch64": "arm64", "macos-x86_64": "x86_64"}
APPLE_TRIPLE = {"macos-aarch64": "aarch64-apple-darwin", "macos-x86_64": "x86_64-apple-darwin"}


def parse_deployment_target(text):
    """`MACOSX_DEPLOYMENT_TARGET=11.0` as rustc prints it -> `11.0`. Older
    toolchains spelt it `deployment_target=`; both are accepted."""
    match = re.search(r"(?:MACOSX_DEPLOYMENT_TARGET|deployment_target)=([0-9.]+)", text)
    if not match:
        raise SystemExit(f"rustc did not report a deployment target: {text!r}")
    return match.group(1)


def apple_deployment_target(platform):
    """The minimum macOS the library itself was built for, from rustc, so the
    combined object carries the same minimum and not the host's."""
    output = subprocess.check_output(["rustc", "--print", "deployment-target",
                                      "--target", APPLE_TRIPLE[platform]], text=True)
    return parse_deployment_target(output)


def llvm_objcopy():
    """Find the active Rust toolchain's object editor (llvm-tools-preview)."""
    sysroot = subprocess.check_output(["rustc", "--print", "sysroot"], text=True).strip()
    version = subprocess.check_output(["rustc", "-vV"], text=True)
    host = next(line.removeprefix("host: ") for line in version.splitlines()
                if line.startswith("host: "))
    bundled = Path(sysroot) / "lib/rustlib" / host / "bin/llvm-objcopy"
    if bundled.is_file():
        return str(bundled)
    external = shutil.which("llvm-objcopy")
    if external:
        return external
    raise SystemExit("macOS static packaging requires llvm-objcopy; install "
                     "llvm-tools-preview for the active Rust toolchain or put llvm-objcopy on PATH")


def localize_commands(platform, archive, exports, combined, deployment_target=None,
                      objcopy="llvm-objcopy"):
    """The commands that merge a static library into one relocatable object
    with only vanedb_rs_* global, and the one that repacks it.

    Returns `(combine, repack)`: `combine` is the list of argvs that produce
    `combined` from `archive`, `repack` the argv that rebuilds `archive` from
    `combined` after the original is removed. Windows must instead use the
    staticlib-only Rust LTO path in capi_windows_static.

    Apple's `ld -r` cannot infer the slice from a static archive ("Missing
    -arch option") and, given an arch, wants the platform and versions too
    ("Missing -platform_version option"). The relocatable link therefore
    goes through the compiler driver, which supplies both from `-arch` and
    `-mmacosx-version-min`; the minimum is what rustc built the library for
    (`rustc --print deployment-target`), not the host. `libtool -static`
    takes neither.

    The `capi` profile's fat LTO keeps `-C embed-bitcode=yes` (cargo passes
    `no` only without LTO), so every staticlib object carries LLVM bitcode:
    `__LLVM,__bitcode` on Mach-O, `.llvmbc` and `.llvmcmd` on ELF. Apple's
    `nm` (LLVM 17 in Xcode 16) fails to parse bitcode written by rustc's
    LLVM 22, and on Linux it is dead weight in the shipped archive. It is
    stripped from the combined object with objcopy. Apple's bitcode_strip
    invokes an obsolete linker option that fails on current Xcode, so macOS
    uses llvm-objcopy's Mach-O section removal. Nothing links against it.
    """
    archive, combined = str(archive), str(combined)
    exports = Path(exports)
    family = platform.split("-", 1)[0]
    if family == "linux":
        return (
            [["ld", "-r", "--whole-archive", archive, "--no-whole-archive", "-o", combined],
             ["objcopy", f"--keep-global-symbols={exports / 'vanedb_capi.syms'}",
              "--remove-section", ".llvmbc", "--remove-section", ".llvmcmd", combined]],
            ["ar", "rcs", archive, combined],
        )
    if family == "macos":
        if deployment_target is None:
            raise ValueError("a macOS relocatable link needs the deployment target")
        return (
            [["xcrun", "clang", "-arch", APPLE_ARCH[platform],
              f"-mmacosx-version-min={deployment_target}", "-nostdlib", "-r",
              f"-Wl,-force_load,{archive}",
              f"-Wl,-exported_symbols_list,{exports / 'vanedb_capi.exp'}",
              "-o", combined],
             [objcopy, "--remove-section=__LLVM,__bitcode",
              "--remove-section=__LLVM,__cmdline", combined]],
            ["libtool", "-static", "-o", archive, combined],
        )
    raise ValueError(f"no safe relocatable-link localization for {platform}")


def localize_static(archive, work, platform):
    """Leave only vanedb_rs_* global in the static library.

    A Rust staticlib exports every Rust symbol as global (thousands), which
    collides with any other Rust-built static library in the same program. The
    members are first merged into one relocatable object -- localizing across
    separate members would break their references to each other -- and then
    everything not on the allowlist is made local. Undefined symbols (the
    system libraries from native-static-libs) are untouched by definition.

    Linux: `ld -r` + `objcopy --keep-global-symbols`. macOS: `clang -r` with
    `-exported_symbols_list`, then `libtool -static`. Windows is handled by
    capi_windows_static before this function is reached.
    """
    target = apple_deployment_target(platform) if platform.startswith("macos") else None
    commands = localize_commands(platform, archive, ROOT / "vanedb-capi/exports",
                                 work / "vanedb_capi_combined.o", target,
                                 llvm_objcopy() if platform.startswith("macos") else "objcopy")
    combine, repack = commands
    for command in combine:
        subprocess.run(command, check=True)
    archive.unlink()
    subprocess.run(repack, check=True)
    (work / "vanedb_capi_combined.o").unlink()
    return True


def static_globals(archive):
    if sys.platform == "linux":
        listing = subprocess.check_output(["nm", "--defined-only", "--extern-only", str(archive)], text=True)
        return sorted({line.split()[-1] for line in listing.splitlines()
                       if line.strip() and not line.endswith(":") and len(line.split()) >= 3})
    listing = subprocess.check_output(["nm", "-gU", str(archive)], text=True)
    names = {line.split()[-1] for line in listing.splitlines()
             if line.strip() and not line.endswith(":") and len(line.split()) >= 3}
    return sorted(n[1:] if n.startswith("_") else n for n in names)


def bitcode_sections(archive):
    """Names of embedded-bitcode sections or segments still in the archive."""
    if sys.platform == "linux":
        listing = subprocess.check_output(["readelf", "-S", "-W", str(archive)], text=True)
        return sorted({name for name in (".llvmbc", ".llvmcmd") if name in listing})
    listing = subprocess.check_output(["otool", "-l", str(archive)], text=True)
    return ["__LLVM"] if "segname __LLVM" in listing or "sectname __bitcode" in listing else []


def check_static_globals(archive):
    leftover = bitcode_sections(archive)
    if leftover:
        raise SystemExit(f"{archive.name} still carries embedded bitcode: {leftover}")
    globals_ = static_globals(archive)
    stray = [n for n in globals_ if not n.startswith("vanedb_rs_")]
    if stray:
        raise SystemExit(f"{archive.name} still exports {len(stray)} non-vanedb_rs_ globals, e.g. {stray[:10]}")
    expected = capi_exports.functions((ROOT / "vanedb-capi/include/vanedb_rs_capi.h").read_text(encoding="utf-8"))
    missing = sorted(set(expected) - set(globals_))
    if missing:
        raise SystemExit(f"{archive.name} lost exported functions: {missing}")
    print(f"{archive.name}: only the {len(globals_)} vanedb_rs_* functions are global")


# --- consumers ----------------------------------------------------------------

def run(command, cwd, env=None):
    print("+", " ".join(str(c) for c in command), flush=True)
    subprocess.run([str(c) for c in command], cwd=cwd, check=True, env=env)


def test_consumers(extracted, work):
    """Everything below sees only the extracted archive."""
    build = work / "examples-build"
    run(["cmake", "-S", extracted / "examples", "-B", build, "-DCMAKE_BUILD_TYPE=Release"], work)
    run(["cmake", "--build", build, "--config", "Release"], work)
    run(["ctest", "--test-dir", build, "--build-config", "Release", "--output-on-failure", "--no-tests=error"], work)
    if sys.platform == "darwin":
        links = subprocess.check_output(["otool", "-L", str(build / "acceptance")], text=True)
        if "@rpath/libvanedb_capi.dylib" not in links or str(ROOT) in links:
            raise SystemExit(f"C consumer links outside the extracted archive:\n{links}")

    build = work / "cmake-consumer-build"
    run(["cmake", "-S", extracted / "consumers/cmake", "-B", build, "-DCMAKE_BUILD_TYPE=Release",
         f"-DCMAKE_PREFIX_PATH={extracted}"], work)
    run(["cmake", "--build", build, "--config", "Release"], work)
    run(["ctest", "--test-dir", build, "--build-config", "Release", "--output-on-failure", "--no-tests=error"], work)

    if sys.platform != "win32":
        if not shutil.which("pkg-config"):
            raise SystemExit("pkg-config is required to test the pkg-config consumer")
        env = dict(os.environ, PKG_CONFIG_PATH=str(extracted / "lib/pkgconfig"))
        out = work / "pkgconfig-consumer-build"
        out.mkdir()
        modversion = subprocess.check_output(["pkg-config", "--modversion", "vanedb"], env=env, text=True).strip()
        print(f"pkg-config reports vanedb {modversion}")
        run(["make", "-C", extracted / "consumers/pkgconfig", f"OUT={out}", "test"], work, env=env)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--platform", required=True, choices=[
        "linux-x86_64", "linux-aarch64", "macos-x86_64", "macos-aarch64", "windows-x86_64"
    ])
    parser.add_argument("--library-dir", type=Path, default=ROOT / "target" / PROFILE,
                        help=f"where cargo put the libraries (default: target/{PROFILE})")
    parser.add_argument("--output", type=Path, default=ROOT / "target/c-artifacts")
    parser.add_argument("--native-static-libs", default=None,
                        help="the `native-static-libs` line to record instead of asking cargo")
    args = parser.parse_args()
    version = tomllib.loads((ROOT / "vanedb-capi/Cargo.toml").read_text())["package"]["version"]
    native = args.native_static_libs
    if native is None and sys.platform != "win32":
        native = native_static_libs()
    names = LIBRARIES[sys.platform]
    name = f"vanedb-capi-{version}-{args.platform}"
    args.output.mkdir(parents=True, exist_ok=True)
    archive = args.output.resolve() / f"{name}.zip"
    with tempfile.TemporaryDirectory(prefix="vanedb-c-package-") as temporary:
        directory = Path(temporary)
        package = directory / "stage" / name
        (package / "lib").mkdir(parents=True)
        for kind in ("shared", "static", "import"):
            if names[kind] and not (sys.platform == "win32" and kind == "static"):
                shutil.copy2(args.library_dir / names[kind], package / "lib" / names[kind])
        shared = package / "lib" / names["shared"]
        static = package / "lib" / names["static"]
        before = shared.stat().st_size
        strip_shared(shared)
        print(f"{names['shared']}: {before} bytes before strip, {shared.stat().st_size} after")
        if sys.platform == "win32":
            api = set(capi_exports.functions((ROOT / "vanedb-capi/include/vanedb_rs_capi.h").read_text()))
            built_native = capi_windows_static.build_and_package(ROOT, static, directory, api)
            if native is not None and link_tokens(native) != link_tokens(built_native):
                raise SystemExit("--native-static-libs differs from the isolated Windows build")
            native = built_native
        elif localize_static(static, directory, args.platform):
            check_static_globals(static)
        print(f"native-static-libs: {native}")
        capi_exports.check(shared)
        if sys.platform == "darwin":
            # Rust's default install name points into the build checkout.
            # A distributed consumer must load this archive's copy instead.
            subprocess.run(["install_name_tool", "-id", "@rpath/libvanedb_capi.dylib", str(shared)], check=True)
            subprocess.run(["codesign", "--force", "--sign", "-", str(shared)], check=True)
        for relative in PACKAGED_SOURCES:
            destination = package / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / "vanedb-capi" / relative, destination)
        shutil.copy2(ROOT / "LICENSE", package / "LICENSE")
        write_package_files(package, version, native)
        requirements = compatibility(shared, args.platform, native)
        (package / "compatibility.json").write_text(json.dumps(requirements, indent=2) + "\n")
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
            for source in sorted(package.rglob("*")):
                if source.is_file():
                    output.write(source, source.relative_to(package.parent))
        # Compile and execute using only the archive's files and libraries.
        shutil.rmtree(package.parent)
        with zipfile.ZipFile(archive) as packaged:
            packaged.extractall(directory / "extracted")
        test_consumers(directory / "extracted" / name, directory)
    with archive.open("rb") as packaged:
        digest = hashlib.file_digest(packaged, "sha256").hexdigest()
    archive.with_suffix(".zip.sha256").write_text(f"{digest}  {archive.name}\n")
    print(f"Verified {archive}")


if __name__ == "__main__":
    main()
