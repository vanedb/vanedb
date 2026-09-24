#!/usr/bin/env python3
"""Exercise the wheel guard in fresh interpreters with isolated import paths."""

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class WheelInstallTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="wheel-guard-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.repo = self.root / "vanedb"
        self.checker = self.repo / "scripts/check_wheel_install.py"
        self.checker.parent.mkdir(parents=True)
        shutil.copyfile(Path(__file__).with_name("check_wheel_install.py"), self.checker)
        self.site = self.root / "vanedb-venv/lib/site-packages"
        self.site.mkdir(parents=True)
        self.platlib = self.root / "platform/site-packages"
        self.platlib.mkdir(parents=True)
        self.name = "wheel_guard_fixture"

    def package(self, parent, namespace=False):
        package = parent / self.name
        package.mkdir(parents=True)
        if not namespace:
            (package / "__init__.py").write_text("VALUE = 'fixture'\n")
        return package

    def check(self, paths, checker=None):
        # Override only the interpreter's configured install roots. Imports,
        # path filtering and module origins all exercise the real interpreter.
        config = json.dumps({
            "paths": list(map(str, paths)),
            "roots": {"purelib": str(self.site), "platlib": str(self.platlib)},
            "checker": str(checker or self.checker),
            "name": self.name,
        })
        return subprocess.run(
            [sys.executable, "-I", "-c", """
import json, runpy, sys, sysconfig
config = json.loads(sys.argv[1])
sysconfig.get_paths = lambda: config["roots"]
sys.path[:0] = config["paths"]
sys.argv = [config["checker"], config["name"]]
runpy.run_path(config["checker"], run_name="__main__")
""", config],
            cwd=self.repo,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_sibling_environment_sharing_checkout_prefix_passes(self):
        package = self.package(self.site)
        result = self.check([self.site])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(str(package / "__init__.py"), result.stdout)

    def test_platform_install_root_passes(self):
        self.package(self.platlib)
        result = self.check([self.platlib])
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_source_tree_import_paths_are_removed(self):
        for path in [self.repo, self.repo / "src"]:
            with self.subTest(path=path):
                self.package(path)
                result = self.check(["", path])
                self.assertEqual(result.returncode, 1)
                self.assertIn("is not installed", result.stderr)

    def test_source_shadow_does_not_replace_installed_module(self):
        source = self.package(self.repo)
        (source / "__init__.py").write_text("raise AssertionError('source imported')\n")
        installed = self.package(self.site)
        result = self.check(["", self.repo, self.site])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(str(installed / "__init__.py"), result.stdout)

    def test_site_packages_sibling_is_not_an_install_root(self):
        # Keep this path independent of the checkout prefix, so this catches
        # false acceptance by the install-root check on its own.
        sibling = self.platlib.with_name("site-packages-untrusted")
        self.package(sibling)
        result = self.check([sibling])
        self.assertEqual(result.returncode, 1)
        self.assertIn("outside site-packages", result.stderr)

    def test_namespace_package_is_not_a_wheel(self):
        self.package(self.site, namespace=True)
        result = self.check([self.site])
        self.assertEqual(result.returncode, 1)
        self.assertIn("namespace package", result.stderr)

    def symlink(self, link, target):
        try:
            link.symlink_to(target, target_is_directory=True)
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"directory symlinks unavailable: {error}")

    def test_symlink_into_checkout_is_removed(self):
        self.package(self.repo)
        alias = self.root / "source-alias"
        self.symlink(alias, self.repo)
        result = self.check([alias])
        self.assertEqual(result.returncode, 1)
        self.assertIn("is not installed", result.stderr)

    def test_symlinked_checkout_still_removes_source(self):
        self.package(self.repo)
        alias = self.root / "checkout-alias"
        self.symlink(alias, self.repo)
        result = self.check([self.repo], checker=alias / "scripts/check_wheel_install.py")
        self.assertEqual(result.returncode, 1)
        self.assertIn("is not installed", result.stderr)

    def test_package_symlink_outside_site_packages_is_rejected(self):
        outside = self.package(self.root / "outside")
        self.symlink(self.site / self.name, outside)
        result = self.check([self.site])
        self.assertEqual(result.returncode, 1)
        self.assertIn("outside site-packages", result.stderr)


if __name__ == "__main__":
    unittest.main()
