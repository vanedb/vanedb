#!/usr/bin/env python3
"""Tests for the argv `package_capi.py` assembles.

The script itself runs only on the native packaging legs, after a full
release build, so a wrong flag surfaces as a failed macOS or Windows job
twenty minutes in. The first such failure was Apple's `ld -r` refusing a
static archive without `-arch`; these cases pin the commands without a Mac.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_capi_header  # noqa: E402
import package_capi  # noqa: E402

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
        self.assertEqual(combine[1], ["xcrun", "bitcode_strip", "-r", str(COMBINED), "-o", str(COMBINED)])
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

    def test_windows_has_no_localization(self):
        self.assertIsNone(commands("windows-x86_64"))

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


if __name__ == "__main__":
    unittest.main()
