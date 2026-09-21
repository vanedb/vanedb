#!/usr/bin/env python3
"""Compare the C ABI shared library against the previous tagged release with
abidiff (libabigail), RFC 0002 stage 1.

The baseline is the Linux x86-64 archive attached to the newest GitHub
Release whose `vanedb-v<version>` (or `vanedb-crate-v<version>`) tag is older
than the version in vanedb-capi/Cargo.toml. A removed or changed symbol
fails; added symbols are allowed (`--no-added-syms`), which is what the
header rule promises. When no such release carries the archive -- true until
the first release published under RFC 0002 stage 5 -- the check is skipped
with a notice rather than passed silently. Needs `gh` with a token that can
read releases, and `abidiff` on PATH.
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAG_PREFIXES = ("vanedb-crate-v", "vanedb-v")
ASSET = "linux-x86_64.zip"


def parse_version(text):
    """(major, minor, patch, is_release, prerelease) so releases sort above
    their prereleases and prereleases compare lexically among themselves."""
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z.-]+))?", text)
    if not match:
        return None
    major, minor, patch, pre = match.groups()
    return (int(major), int(minor), int(patch), pre is None, pre or "")


def tag_version(tag):
    for prefix in TAG_PREFIXES:
        if tag.startswith(prefix):
            return parse_version(tag[len(prefix):])
    return None


def notice(message):
    print(f"::notice title=abidiff::{message}")
    print(message)


def releases(repo):
    command = ["gh", "release", "list", "--repo", repo, "--limit", "200",
               "--json", "tagName,isDraft,isPrerelease"]
    return json.loads(subprocess.check_output(command, text=True))


def baseline_tag(current, published):
    candidates = []
    for release in published:
        if release.get("isDraft"):
            continue
        version = tag_version(release["tagName"])
        if version and version < current:
            candidates.append((version, release["tagName"]))
    if not candidates:
        return None
    return max(candidates)[1]


def download_baseline(repo, tag, directory):
    command = ["gh", "release", "download", tag, "--repo", repo, "--dir", str(directory),
               "--pattern", f"vanedb-capi-*-{ASSET}"]
    completed = subprocess.run(command, text=True, capture_output=True)
    archives = list(directory.glob(f"*-{ASSET}"))
    if completed.returncode != 0 or not archives:
        return None
    with zipfile.ZipFile(archives[0]) as archive:
        archive.extractall(directory / "baseline")
    found = list((directory / "baseline").rglob("libvanedb_capi.so"))
    return found[0] if found else None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("current", type=Path, help="the freshly built libvanedb_capi.so")
    parser.add_argument("--repo", default="vanedb/vanedb")
    parser.add_argument("--baseline", type=Path, default=None,
                        help="compare against this shared object instead of a release asset")
    args = parser.parse_args()
    if not args.current.exists():
        raise SystemExit(f"{args.current} does not exist; build with cargo build -p vanedb-capi --profile capi")
    version_text = tomllib.loads((ROOT / "vanedb-capi/Cargo.toml").read_text())["package"]["version"]
    current_version = parse_version(version_text)

    with tempfile.TemporaryDirectory(prefix="vanedb-abidiff-") as temporary:
        directory = Path(temporary)
        baseline = args.baseline
        if baseline is None:
            tag = baseline_tag(current_version, releases(args.repo))
            if tag is None:
                notice(f"no release older than {version_text} exists; nothing to compare against "
                       f"(the first baseline is the first release that ships C ABI archives)")
                return 0
            baseline = download_baseline(args.repo, tag, directory)
            if baseline is None:
                notice(f"release {tag} carries no vanedb-capi {ASSET} asset; skipped "
                       f"(releases before RFC 0002 stage 5 attached no C ABI archives)")
                return 0
            print(f"baseline: {tag}")
        command = ["abidiff", "--no-added-syms",
                   "--headers-dir2", str(ROOT / "vanedb-capi/include"),
                   str(baseline), str(args.current)]
        print("+", " ".join(command), flush=True)
        completed = subprocess.run(command)
        # abidiff's exit status is a bit set: 4 = ABI change, 8 = incompatible
        # change, 1/2 = error or usage. Any of them fails the gate.
        if completed.returncode != 0:
            print(f"abidiff exited {completed.returncode}: the C ABI changed against the baseline",
                  file=sys.stderr)
            return 1
        print("abidiff: no removed or changed symbols against the baseline")
        return 0


if __name__ == "__main__":
    sys.exit(main())
