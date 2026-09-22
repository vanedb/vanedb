#!/usr/bin/env python3
"""Tests for the argv `package_capi.py` assembles.

The script itself runs only on the native packaging legs, after a full
release build, so a wrong flag surfaces as a failed macOS or Windows job
twenty minutes in. The first such failure was Apple's `ld -r` refusing a
static archive without `-arch`; these cases pin the commands without a Mac.
"""

import sys
import json
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_capi_header  # noqa: E402
import package_capi  # noqa: E402
import capi_windows_static as windows  # noqa: E402

ARCHIVE = Path("/pkg/lib/libvanedb_capi.a")
EXPORTS = Path("/src/vanedb-capi/exports")
COMBINED = Path("/work/vanedb_capi_combined.o")


def commands(platform, deployment_target=None):
    return package_capi.localize_commands(platform, ARCHIVE, EXPORTS, COMBINED, deployment_target)


class LocalizeCommands(unittest.TestCase):
    def test_apple_relocatable_link_goes_through_the_driver_with_arch_and_minimum(self):
        for platform, arch, minimum in [("macos-aarch64", "arm64", "11.0"), ("macos-x86_64", "x86_64", "10.12")]:
            with self.subTest(platform=platform):
                combine, repack = commands(platform, minimum)
                self.assertEqual(len(combine), 2)
                link = combine[0]
                self.assertEqual(link[:2], ["xcrun", "clang"])
                self.assertEqual(link[link.index("-arch") + 1], arch)
                self.assertIn(f"-mmacosx-version-min={minimum}", link)
                self.assertIn("-r", link)
                self.assertIn("-nostdlib", link)
                self.assertIn(f"-Wl,-force_load,{ARCHIVE}", link)
                self.assertIn(f"-Wl,-exported_symbols_list,{EXPORTS / 'vanedb_capi.exp'}", link)
                self.assertEqual(link[-2:], ["-o", str(COMBINED)])
                self.assertEqual(repack, ["libtool", "-static", "-o", str(ARCHIVE), str(COMBINED)])
                for argv in combine + [repack]:
                    self.assertNotEqual(argv[0], "ld", "a bare ld -r cannot supply -platform_version")
                    self.assertNotIn("-platform_version", argv)

    def test_apple_link_refuses_to_guess_the_deployment_target(self):
        with self.assertRaises(ValueError):
            commands("macos-aarch64")

    def test_deployment_target_is_parsed_from_rustc(self):
        self.assertEqual(package_capi.parse_deployment_target("MACOSX_DEPLOYMENT_TARGET=11.0\n"), "11.0")
        self.assertEqual(package_capi.parse_deployment_target("MACOSX_DEPLOYMENT_TARGET=10.12\n"), "10.12")
        self.assertEqual(package_capi.parse_deployment_target("deployment_target=11.0"), "11.0")
        with self.assertRaises(SystemExit):
            package_capi.parse_deployment_target("")

    def test_embedded_bitcode_is_stripped_from_the_combined_object(self):
        # Fat LTO keeps embed-bitcode=yes; Apple's nm cannot read rustc's
        # LLVM 22 bitcode and the Linux archive would ship it for nothing.
        combine, _ = commands("macos-aarch64", "11.0")
        self.assertEqual(combine[1], ["llvm-objcopy", "--remove-section=__LLVM,__bitcode",
                                    "--remove-section=__LLVM,__cmdline", str(COMBINED)])
        combine, _ = commands("linux-x86_64")
        objcopy = combine[1]
        self.assertEqual(objcopy[0], "objcopy")
        for section in [".llvmbc", ".llvmcmd"]:
            self.assertEqual(objcopy[objcopy.index(section) - 1], "--remove-section")
        self.assertEqual(objcopy[-1], str(COMBINED))

    def test_linux_uses_objcopy_and_takes_no_arch(self):
        for platform in ["linux-x86_64", "linux-aarch64"]:
            with self.subTest(platform=platform):
                combine, repack = commands(platform)
                self.assertEqual(combine[0][:2], ["ld", "-r"])
                self.assertEqual(combine[1][0], "objcopy")
                self.assertIn(f"--keep-global-symbols={EXPORTS / 'vanedb_capi.syms'}", combine[1])
                for argv in combine + [repack]:
                    self.assertNotIn("-arch", argv)
                self.assertEqual(repack, ["ar", "rcs", str(ARCHIVE), str(COMBINED)])

    def test_windows_must_not_use_a_coff_partial_link(self):
        with self.assertRaises(ValueError):
            commands("windows-x86_64")

    def test_every_packaged_platform_is_covered(self):
        for platform in ["linux-x86_64", "linux-aarch64", "macos-x86_64", "macos-aarch64"]:
            self.assertIsNotNone(commands(platform, "11.0"), platform)


class MsvcBatch(unittest.TestCase):
    def test_the_batch_quotes_the_vcvars_path_and_escapes_nothing(self):
        vcvars = r"C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
        text = check_capi_header.msvc_batch(vcvars, Path(r"D:\src\vanedb-capi\include"), Path(r"C:\tmp\hdr"))
        lines = text.split("\r\n")
        self.assertEqual(lines[0], "@echo off")
        self.assertEqual(lines[1], f'call "{vcvars}" >nul || exit /b 1')
        self.assertNotIn('\\"', text, "backslash-escaped quotes are what cmd.exe cannot read")
        legs = [l for l in lines if l.startswith("cl ")]
        self.assertEqual(len(legs), len(check_capi_header.MSVC_LEGS))
        self.assertIn("/std:c11", legs[1])
        self.assertIn("/std:c++17", legs[2])
        self.assertIn(r"/Tc C:\tmp\hdr\tu.c", legs[0])
        self.assertIn(r"/Tp C:\tmp\hdr\tu.cpp", legs[2])
        for leg in legs:
            self.assertTrue(leg.endswith("|| exit /b 1"))
            self.assertIn("/W4 /WX", leg)


class StaticLinkLine(unittest.TestCase):
    def test_native_static_libs_is_parsed_without_colour_codes(self):
        coloured = ("\x1b[1m\x1b[32m   Compiling\x1b[0m vanedb-capi v0.1.1\n"
                    "\x1b[1m\x1b[36mnote\x1b[0m\x1b[1m: native-static-libs: "
                    "-lgcc_s -lutil -lrt -lpthread -lm -ldl -lc\x1b[0m\n")
        self.assertEqual(package_capi.parse_native_static_libs(coloured),
                         "-lgcc_s -lutil -lrt -lpthread -lm -ldl -lc")
        plain = "note: native-static-libs: -lSystem -framework CoreFoundation\n"
        self.assertEqual(package_capi.parse_native_static_libs(plain),
                         "-lSystem -framework CoreFoundation")
        self.assertIsNone(package_capi.parse_native_static_libs("   Compiling vanedb-capi\n"))

    def test_frameworks_stay_paired_and_defaultlib_is_dropped_for_cmake(self):
        native = "-lSystem -framework CoreFoundation -lc -lSystem"
        self.assertEqual(package_capi.link_tokens(native),
                         ["-lSystem", "-framework CoreFoundation", "-lc"])
        self.assertEqual(package_capi.cmake_link_items(native),
                         "System;-framework CoreFoundation;c")
        msvc = "kernel32.lib advapi32.lib /defaultlib:msvcrt"
        self.assertEqual(package_capi.cmake_link_items(msvc), "kernel32.lib;advapi32.lib")
        self.assertEqual(package_capi.pkgconfig_libs_private(msvc), "kernel32.lib advapi32.lib")


def ar_member(name, body):
    header = f"{name:<16}{0:<12}{0:<6}{0:<6}{0:<8}{len(body):<10}`\n".encode()
    return header + body + (b"\n" if len(body) % 2 else b"")


class WindowsStaticIsolation(unittest.TestCase):
    def test_native_archive_names_and_duplicate_members_are_not_lost(self):
        for table in (b"bcryptprimitives.dll\0", b"bcryptprimitives.dll/\n"):
            data = b"!<arch>\n" + ar_member("/", b"index") + ar_member("//", table)
            data += ar_member("/0", b"first") + ar_member("/0", b"second")
            self.assertEqual(list(windows.archive_members(data)),
                             [("bcryptprimitives.dll", b"first"), ("bcryptprimitives.dll", b"second")])

    def test_bad_archive_offsets_and_truncation_fail_closed(self):
        bad = [b"!<thin>\n", b"!<arch>\nshort",
               b"!<arch>\n" + ar_member("/42", b"body"),
               (b"!<arch>\n" + ar_member("x.obj/", b"body"))[:-1],
               b"!<arch>\n" + ar_member("#1/5", b"abcdebody")]
        for data in bad:
            with self.subTest(data=data), self.assertRaises(ValueError):
                list(windows.archive_members(data))

    def test_import_exceptions_do_not_allow_rust_or_arbitrary_windows_symbols(self):
        allowed = windows.import_symbols("bcryptprimitives.dll")
        self.assertIn("__imp_ProcessPrng", allowed)
        self.assertIn("__IMPORT_DESCRIPTOR_bcryptprimitives", allowed)
        for name in ("rust_eh_personality", "__rust_alloc", "__imp_Arbitrary", "vanedb_rs_hidden"):
            self.assertNotIn(name, allowed)

    def test_coff_gate_rejects_bitcode_wrong_machine_and_bigobj(self):
        def coff(machine, name):
            return struct.pack("<HHIIIHH", machine, 1, 0, 0, 0, 0, 0) + name.ljust(8, b"\0") + bytes(32)
        windows.check_object(coff(0x8664, b".text"))
        for data in (coff(0x8664, b".llvmbc"), coff(0x8664, b".llvmcmd"),
                     coff(0xAA64, b".text"), bytes(60), b"bad"):
            with self.subTest(data=data), self.assertRaises(ValueError):
                windows.check_object(data)

    def test_omitted_implementation_reference_stops_packaging_before_objcopy(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            archive = work / "raw.lib"
            archive.write_bytes(b"!<arch>\n" + ar_member("code.obj/", b"opaque"))
            def symbol_set(_nm, path, defined=True):
                if path == archive:
                    return {"vanedb_rs_api", "required_builtin"}
                return {"vanedb_rs_api"} if defined else {"required_builtin"}
            with patch.object(windows, "symbols", side_effect=symbol_set), \
                    patch.object(windows.subprocess, "run") as run:
                with self.assertRaisesRegex(SystemExit, "required_builtin"):
                    windows.isolate(archive, work / "lto.obj", work / "out.lib", work / "stage",
                                    {"vanedb_rs_api"}, "nm", "ar", "objcopy")
                run.assert_not_called()

    def test_extra_and_missing_symbols_both_fail_the_exact_gate(self):
        for actual in ({"vanedb_rs_api", "__rust_alloc"}, set()):
            with self.assertRaises(SystemExit):
                windows.require_symbols(actual, {"vanedb_rs_api"}, "test archive")


class PackagingDiagnostics(unittest.TestCase):
    def test_default_work_directory_is_removed_on_failure(self):
        with self.assertRaisesRegex(RuntimeError, "failed"):
            with package_capi.packaging_work_directory() as directory:
                self.assertTrue(directory.is_dir())
                raise RuntimeError("failed")
        self.assertFalse(directory.exists())

    def test_requested_work_directory_survives_failure_and_is_unique(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "failed"):
                with package_capi.packaging_work_directory(parent) as first:
                    (first / "evidence").write_text("retained")
                    raise RuntimeError("failed")
            with package_capi.packaging_work_directory(parent) as second:
                self.assertNotEqual(first, second)
            self.assertEqual((first / "evidence").read_text(), "retained")
            self.assertTrue(second.is_dir())

    def check_consumer_results(self, packaged_fails, control_fails):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            extracted = work / "extracted"
            (extracted / "lib").mkdir(parents=True)
            (extracted / "lib/vanedb_capi.lib").write_bytes(b"localized")
            raw = work / "windows-raw-static.lib"
            raw.write_bytes(b"untouched rust archive")
            errors = {}
            commands = []

            def run(command, cwd, env=None):
                commands.append(command)
                if command[0] != "ctest":
                    return
                self.assertEqual(env["VANEDB_CAPI_TRACE"], "1")
                control = command[2].name == "raw-static-control-build"
                if control:
                    self.assertIn("^acceptance_static$", command)
                    self.assertEqual((work / "raw-static-control-package/lib/vanedb_capi.lib").read_bytes(),
                                     raw.read_bytes())
                if control_fails if control else packaged_fails:
                    error = subprocess.CalledProcessError(8, list(map(str, command)))
                    errors["control" if control else "packaged"] = error
                    raise error

            with patch.object(package_capi.sys, "platform", "win32"), \
                    patch.object(package_capi, "run", side_effect=run):
                if packaged_fails or control_fails:
                    with self.assertRaises(subprocess.CalledProcessError) as raised:
                        package_capi.test_cmake_consumer(extracted, work, raw)
                    self.assertIs(raised.exception, errors["packaged" if packaged_fails else "control"])
                else:
                    package_capi.test_cmake_consumer(extracted, work, raw)
            self.assertEqual(sum(command[0] == "ctest" for command in commands), 2)
            self.assertEqual((extracted / "lib/vanedb_capi.lib").read_bytes(), b"localized")
            results = json.loads((work / "consumer-results.json").read_text())
            self.assertEqual(set(results), {"packaged", "raw-static-control"})

    def test_control_cannot_mask_packaged_failure(self):
        self.check_consumer_results(packaged_fails=True, control_fails=False)

    def test_both_failures_preserve_packaged_error(self):
        self.check_consumer_results(packaged_fails=True, control_fails=True)

    def test_control_failure_also_gates(self):
        self.check_consumer_results(packaged_fails=False, control_fails=True)

    def test_both_consumers_pass(self):
        self.check_consumer_results(packaged_fails=False, control_fails=False)


if __name__ == "__main__":
    unittest.main()
