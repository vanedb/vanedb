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
import package_capi  # noqa: E402

ARCHIVE = Path("/pkg/lib/libvanedb_capi.a")
EXPORTS = Path("/src/vanedb-capi/exports")
COMBINED = Path("/work/vanedb_capi_combined.o")


def commands(platform):
    return package_capi.localize_commands(platform, ARCHIVE, EXPORTS, COMBINED)


class LocalizeCommands(unittest.TestCase):
    def test_apple_ld_r_names_the_arch_of_the_platform(self):
        for platform, arch in [("macos-aarch64", "arm64"), ("macos-x86_64", "x86_64")]:
            with self.subTest(platform=platform):
                combine, repack = commands(platform)
                self.assertEqual(len(combine), 1)
                ld = combine[0]
                self.assertEqual(ld[:2], ["ld", "-r"])
                self.assertIn("-arch", ld)
                self.assertEqual(ld[ld.index("-arch") + 1], arch)
                self.assertEqual(ld[ld.index("-exported_symbols_list") + 1],
                                 str(EXPORTS / "vanedb_capi.exp"))
                self.assertEqual(ld[-2:], ["-o", str(COMBINED)])
                self.assertEqual(repack, ["libtool", "-static", "-o", str(ARCHIVE), str(COMBINED)])
                self.assertNotIn("-arch", repack)

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
            self.assertIsNotNone(commands(platform), platform)


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
