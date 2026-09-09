#!/usr/bin/env python3
"""Tests for `bump_version.py`, run by CI's workflow-lint job.

The script previously shipped a bug that made it unable to complete a single
bump: four sites live in `cpp/src/core/version.h`, and edits planned against
the text read at plan time meant the last write discarded the other three. It
was not caught, because the only check performed was a round trip — bump away
and back — and corrupting a value then corrupting it back leaves `git diff`
clean. These tests assert the resulting values instead.
"""

import pathlib
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import bump_version  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent
NEEDED = sorted({p for p, _, _ in bump_version.SITES} | {bump_version.IDENTITY})


def fixture(tmp):
    """A copy of just the files the script reads or writes."""
    for rel in NEEDED:
        dst = tmp / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / rel, dst)
    return tmp


def value(root, path, pattern):
    import re
    text = (root / path).read_text(encoding="utf-8")
    match = re.search(pattern, text)
    return match.group(1) if match else None


CHECKS = [
    # Every site, by the same pattern the script uses, asserted individually.
    ("vanedb/Cargo.toml",        r'(?m)^version = "([^"]+)"',          "full"),
    ("vanedb-py/Cargo.toml",     r'(?m)^version = "([^"]+)"',          "full"),
    ("vanedb-capi/Cargo.toml",   r'(?m)^version = "([^"]+)"',          "full"),
    ("vanedb-wasm/Cargo.toml",   r'(?m)^version = "([^"]+)"',          "full"),
    ("vanedb-py/pyproject.toml", r'(?m)^version = "([^"]+)"',          "full"),
    ("cpp/pyproject.toml",       r'(?m)^version = "([^"]+)"',          "full"),
    ("cpp/src/core/version.h",   r'VERSION_STRING = "([^"]+)"',        "full"),
    ("cpp/Doxyfile",             r'PROJECT_NUMBER\s+= "([^"]+)"',      "full"),
    ("cpp/CMakeLists.txt",       r'project\(vanedb VERSION ([0-9.]+)', "core"),
    ("cpp/src/core/version.h",   r"VERSION_MAJOR = (\d+)",             "major"),
    ("cpp/src/core/version.h",   r"VERSION_MINOR = (\d+)",             "minor"),
    ("cpp/src/core/version.h",   r"VERSION_PATCH = (\d+)",             "patch"),
]


def test_every_site_holds_the_new_value(failures):
    """The bug that shipped: four sites in one file, only the last surviving."""
    for version, core, parts in [("9.8.7-rc.7", "9.8.7", ("9", "8", "7")),
                                 ("1.2.3", "1.2.3", ("1", "2", "3"))]:
        with tempfile.TemporaryDirectory() as tmp:
            root = fixture(pathlib.Path(tmp))
            if bump_version.bump(root, version, quiet=True) != 0:
                failures.append(f"{version}: bump returned non-zero")
                continue
            expect = {"full": version, "core": core,
                      "major": parts[0], "minor": parts[1], "patch": parts[2]}
            for path, pattern, kind in CHECKS:
                got = value(root, path, pattern)
                if got != expect[kind]:
                    failures.append(
                        f"{version}: {path} ({kind}) is {got!r}, expected {expect[kind]!r}")


def test_malformed_versions_write_nothing(failures):
    bad = ["1.2", "abc", "v0.1.0", " 0.1.0", "0.1.0+build5", "0.1.0.1", "0.2.0\n", ""]
    for version in bad:
        with tempfile.TemporaryDirectory() as tmp:
            root = fixture(pathlib.Path(tmp))
            before = {p: (root / p).read_bytes() for p in NEEDED}
            if bump_version.bump(root, version, quiet=True) == 0:
                failures.append(f"{version!r} was accepted; it should be refused")
            after = {p: (root / p).read_bytes() for p in NEEDED}
            if before != after:
                failures.append(f"{version!r} was refused but modified files")


def test_a_failed_site_writes_nothing(failures):
    """A pattern that stops matching must abort before any write."""
    with tempfile.TemporaryDirectory() as tmp:
        root = fixture(pathlib.Path(tmp))
        doxyfile = root / "cpp/Doxyfile"
        doxyfile.write_text(
            doxyfile.read_text(encoding="utf-8").replace('PROJECT_NUMBER         = "',
                                                         "PROJECT_NUMBER = "),
            encoding="utf-8")
        before = {p: (root / p).read_bytes() for p in NEEDED if p != "cpp/Doxyfile"}
        if bump_version.bump(root, "4.5.6", quiet=True) == 0:
            failures.append("a non-matching site did not abort the bump")
        after = {p: (root / p).read_bytes() for p in NEEDED if p != "cpp/Doxyfile"}
        if before != after:
            failures.append("a failed site left other files modified")


def test_line_endings_are_preserved(failures):
    """A CRLF file must not come back LF, and vice versa."""
    with tempfile.TemporaryDirectory() as tmp:
        root = fixture(pathlib.Path(tmp))
        doxyfile = root / "cpp/Doxyfile"
        doxyfile.write_bytes(doxyfile.read_bytes().replace(b"\n", b"\r\n"))
        crlf_before = doxyfile.read_bytes().count(b"\r\n")
        bump_version.bump(root, "2.3.4", quiet=True)
        crlf_after = doxyfile.read_bytes().count(b"\r\n")
        if crlf_after != crlf_before:
            failures.append(
                f"CRLF endings changed: {crlf_before} before, {crlf_after} after")


def test_unchecked_sites_ignores_comments(failures):
    """A path named only in a comment is not coverage."""
    with tempfile.TemporaryDirectory() as tmp:
        root = fixture(pathlib.Path(tmp))
        identity = root / bump_version.IDENTITY
        identity.write_text(
            identity.read_text(encoding="utf-8") + "\n// mentions only/in/a/comment.toml\n",
            encoding="utf-8")
        if not bump_version.unchecked_sites(root, {"only/in/a/comment.toml"}):
            failures.append("a path named only in a comment counted as checked")


def main():
    failures = []
    for test in (test_every_site_holds_the_new_value, test_malformed_versions_write_nothing,
                 test_a_failed_site_writes_nothing, test_line_endings_are_preserved,
                 test_unchecked_sites_ignores_comments):
        test(failures)
        print(f"  ran {test.__name__}")
    for failure in failures:
        print(f"FAIL {failure}")
    print(f"{'FAILED' if failures else 'ok'} — {len(failures)} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
