"""Build and isolate the Windows static implementation before native packaging.

Do not partially link Rust COFF members with GNU ld: it can leave COMDAT and
weak-alias indices invalid. Rust's staticlib-only fat LTO supplies one object
instead. Native import members stay byte-for-byte as rustc produced them.
"""

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess


TARGET = "x86_64-pc-windows-msvc"
IMPORTS = {
    "kernel32.dll": {"CloseHandle", "GetLastError", "GetModuleHandleA", "GetProcAddress", "Sleep"},
    "bcryptprimitives.dll": {"ProcessPrng"},
    "api-ms-win-core-synch-l1-2-0.dll": {"WaitOnAddress", "WakeByAddressAll", "WakeByAddressSingle"},
}


def import_symbols(dll):
    """Exact system-import exceptions, not a wildcard for runtime globals."""
    stem = dll.removesuffix(".dll")
    functions = IMPORTS[dll]
    return functions | {"__imp_" + name for name in functions} | {
        "__IMPORT_DESCRIPTOR_" + stem, "__NULL_IMPORT_DESCRIPTOR_" + stem,
        "\x7f" + stem + "_NULL_THUNK_DATA",
    }


def archive_members(data):
    """Read MSVC/GNU ar members without extracting attacker-controlled paths."""
    if not data.startswith(b"!<arch>\n"):
        raise ValueError("expected a regular COFF archive")
    offset, strings = 8, b""
    while offset < len(data):
        header = data[offset:offset + 60]
        if len(header) != 60 or header[58:] != b"`\n":
            raise ValueError("invalid archive member header")
        size = int(header[48:58])
        if size < 0 or offset + 60 + size > len(data):
            raise ValueError("truncated archive member")
        name = header[:16].decode("ascii").strip()
        body = data[offset + 60:offset + 60 + size]
        offset += 60 + size + size % 2
        if offset > len(data):
            raise ValueError("missing archive member padding")
        if name == "//":
            strings = body
            continue
        if name in ("/", "/SYM64/"):
            continue
        if name.startswith("/"):
            start = int(name[1:])
            if start < 0 or start >= len(strings):
                raise ValueError("invalid archive long-name offset")
            # MSVC uses NUL; GNU ar uses slash + newline.
            ends = [end for marker in (b"\0", b"/\n")
                    if (end := strings.find(marker, start)) >= 0]
            if not ends:
                raise ValueError("unterminated archive long name")
            name = strings[start:min(ends)].decode("utf-8")
        else:
            name = name.removesuffix("/")
        if not name or name.startswith("#1/"):
            raise ValueError("unsupported archive member name")
        yield name, body


def llvm_tool(name):
    sysroot = subprocess.check_output(["rustc", "--print", "sysroot"], text=True).strip()
    version = subprocess.check_output(["rustc", "-vV"], text=True)
    host = next(line.removeprefix("host: ") for line in version.splitlines()
                if line.startswith("host: "))
    suffix = ".exe" if os.name == "nt" else ""
    path = Path(sysroot) / "lib/rustlib" / host / "bin" / (name + suffix)
    if not path.is_file():
        raise SystemExit(f"{name} required: install llvm-tools-preview for the active Rust toolchain")
    return str(path)


def gnu_objcopy():
    candidates = [os.environ.get("VANEDB_COFF_OBJCOPY"), shutil.which("objcopy"),
                  "C:/mingw64/bin/objcopy.exe", "C:/msys64/mingw64/bin/objcopy.exe",
                  "C:/msys64/ucrt64/bin/objcopy.exe"]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            version = subprocess.check_output([candidate, "--version"], text=True)
            if "GNU objcopy" in version:
                print(version.splitlines()[0])
                return candidate
    raise SystemExit("Windows static packaging requires GNU COFF objcopy; set VANEDB_COFF_OBJCOPY")


def symbols(nm, path, defined=True):
    output = subprocess.check_output(
        [nm, "--format=posix", "--no-demangle", "--extern-only",
         "--defined-only" if defined else "--undefined-only", str(path)], text=True)
    result = set()
    for line in output.splitlines():
        if not line.strip() or line.endswith(":"):
            continue
        fields = line.split()
        if len(fields) < 2 or len(fields[1]) != 1:
            raise ValueError(f"unrecognized llvm-nm output: {line!r}")
        result.add(fields[0])
    return result


def require_symbols(actual, expected, description):
    if actual != expected:
        raise SystemExit(f"{description}: missing {sorted(expected - actual)}, "
                         f"unexpected {sorted(actual - expected)}")


def check_object(data):
    # Support ordinary AMD64 COFF only; a future bigobj build needs explicit
    # validation rather than silently bypassing architecture/bitcode checks.
    if len(data) < 20:
        raise ValueError("truncated COFF object")
    machine, count = struct.unpack_from("<HH", data)
    optional = struct.unpack_from("<H", data, 16)[0]
    if machine != 0x8664 or optional or not count or len(data) < 20 + 40 * count:
        raise ValueError("expected an ordinary AMD64 COFF object")
    names = {data[20 + 40 * i:28 + 40 * i].rstrip(b"\0") for i in range(count)}
    if names & {b".llvmbc", b".llvmcmd"}:
        raise ValueError("localized COFF object still carries embedded LLVM bitcode")


def verify_archive_members(archive, members):
    """COFF import grouping depends on member names as well as their bytes."""
    expected = [(path.name, hashlib.sha256(path.read_bytes()).digest()) for path in members]
    actual = [(name, hashlib.sha256(body).digest())
              for name, body in archive_members(archive.read_bytes())]
    if actual != expected:
        raise SystemExit("archive writer changed member names, order, or implementation/native import payloads")


def isolate(raw_archive, lto_object, output, work, api, nm, ar, objcopy):
    """Independently check the implementation and each retained import member."""
    work.mkdir(parents=True, exist_ok=True)
    before_globals = symbols(nm, lto_object)
    undefined = symbols(nm, lto_object, defined=False)
    if not api <= before_globals:
        raise SystemExit("LTO object is missing public C ABI definitions")
    imports, allowed, seen = [], set(), set()
    for name, body in archive_members(raw_archive.read_bytes()):
        if not name.lower().endswith(".dll"):
            continue
        if name not in IMPORTS:
            raise SystemExit(f"unreviewed native import member: {name}")
        digest = hashlib.sha256(body).digest()
        if digest in seen:
            continue
        seen.add(digest)
        # MSVC groups the import descriptor and thunk members by their original
        # DLL basename. Renaming them to import-NNN.obj leaves zero ILT/IAT
        # pointers in the linked PE even though every payload is unchanged.
        path = work / f"import-{len(imports):03d}" / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(body)
        exported = symbols(nm, path)
        if not exported or not exported <= import_symbols(name):
            raise SystemExit(f"{name}: unexpected native import definitions {sorted(exported)}")
        allowed |= exported
        imports.append(path)
    # Any implementation definition still needed from an omitted member makes
    # this toolchain/feature combination unsupported. Never silently drop it.
    omitted = symbols(nm, raw_archive) - before_globals - allowed
    needed = undefined & omitted
    if needed:
        raise SystemExit(f"LTO object still needs omitted implementation members: {sorted(needed)}")
    localized = work / "vanedb_capi.obj"
    allowlist = work / "api.txt"
    allowlist.write_text("\n".join(sorted(api)) + "\n", encoding="utf-8")
    subprocess.run([objcopy, f"--keep-global-symbols={allowlist}",
                    "--remove-section=.llvmbc", "--remove-section=.llvmcmd",
                    str(lto_object), str(localized)], check=True)
    check_object(localized.read_bytes())
    require_symbols(symbols(nm, localized), api, "localized Windows implementation")
    require_symbols(symbols(nm, localized, defined=False), undefined,
                    "Windows implementation undefined references")
    # Always create a new archive. Quick append preserves repeated DLL member
    # basenames; replacement mode must not collapse descriptor/thunk members.
    replacement = work / "vanedb_capi.lib"
    replacement.unlink(missing_ok=True)
    subprocess.run([ar, "qcs", str(replacement), str(localized), *map(str, imports)], check=True)
    verify_archive_members(replacement, [localized, *imports])
    require_symbols(symbols(nm, replacement), api | allowed, "Windows static archive")
    shutil.copy2(replacement, output)
    print(f"{output.name}: {len(api)} API globals plus {len(allowed)} checked native import symbols")
    return allowed


def build_and_package(root, output, work, api, diagnostics=False):
    nm, ar, objcopy = llvm_tool("llvm-nm"), llvm_tool("llvm-ar"), gnu_objcopy()
    target_dir = root / "target/capi-windows-static"
    lto_object = work / "windows-lto.obj"
    command = ["cargo", "rustc", "-p", "vanedb-capi", "--lib", "--crate-type", "staticlib",
               "--profile", "capi", "--target", TARGET, "--target-dir", str(target_dir),
               "--locked", "--color", "never", "--", f"--emit=obj={lto_object}",
               "-C", "lto=fat", "-C", "codegen-units=1", "--print", "native-static-libs"]
    if diagnostics:
        versions = {}
        for tool in (["git", "rev-parse", "HEAD"], ["rustc", "-vV"], ["cargo", "--version"],
                     [nm, "--version"], [ar, "--version"], [objcopy, "--version"],
                     ["cmake", "--version"], ["cl"]):
            try:
                result = subprocess.run(tool, cwd=root, check=False, text=True, capture_output=True)
                versions[" ".join(tool)] = {"exit_code": result.returncode,
                                            "output": result.stdout + result.stderr}
            except OSError as error:
                # CMake can discover MSVC even when cl is absent from PATH;
                # its retained configure logs identify the selected compiler.
                versions[" ".join(tool)] = {"unavailable": str(error)}
        (work / "build-diagnostics.json").write_text(
            json.dumps({"command": command, "tools": versions}, indent=2) + "\n")
    built = subprocess.run(command, cwd=root, check=False, text=True, capture_output=True,
                           env=dict(os.environ, CARGO_TERM_COLOR="never"))
    if diagnostics:
        (work / "rust-static-build.log").write_text(built.stdout + built.stderr)
    if built.returncode:
        raise SystemExit(f"staticlib-only Rust build failed:\n{built.stdout}{built.stderr}")
    match = re.search(r"native-static-libs:\s*(.*)", built.stderr)
    if not match or not lto_object.is_file():
        raise SystemExit("staticlib-only Rust build did not produce its object and native link requirements")
    raw = target_dir / TARGET / "capi/vanedb_capi.lib"
    if diagnostics:
        shutil.copy2(raw, work / "windows-raw-static.lib")
    isolate(raw, lto_object, output, work / "windows-static", api, nm, ar, objcopy)
    return match[1].strip()
