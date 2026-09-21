#!/usr/bin/env python3
"""The C ABI compatibility gate against the previous release (RFC 0002 stage 1).

Two layers, because neither alone sees enough:

1. Prototypes. The baseline archive's `include/vanedb_rs_capi.h` and the
   current header are parsed by `scripts/capi_exports.py` into normalised
   prototypes (return type, name, parameter types; typedefs expanded and
   parameter names dropped).
   A removed or changed prototype fails; an addition passes, which is the
   header's rule. This is the only layer that sees a changed signature: the
   shipped library carries no DWARF (the `capi` profile inherits `release`,
   and the packaged copy is stripped), so abidiff cannot. Measured on this
   PR: against the 0.1.1 library, stripped, abidiff reports "0 Removed,
   0 Changed" although every handle parameter changed from a pointer to
   `uint64_t`; a debug build of both sides reports 47 changed.
2. abidiff (libabigail) on the ELF, `--no-added-syms`. Without debug info it
   compares the dynamic symbol tables, so it detects a removed symbol and
   nothing else; `--no-added-syms` also hides additions. It stays as the
   binary-level check that a symbol the header declares has not vanished
   from the library.

`VANEDB_RS_ABI_VERSION` keys the verdict. A baseline whose header carries a
lower version, or none (0.1.1 predates the macro and counts as 0), is an
intentional incompatible release: both layers still run and print what they
found, and the job passes with a notice. With the same version both layers
must pass. Tool errors always fail, and ABI-version downgrades are rejected.

The baseline is the newest non-draft GitHub Release tagged
`vanedb-crate-v<version>` -- the crate release is what carries the C ABI
archives; `vanedb-v<version>` is the Python tag and is tried second at the
same version -- whose version is at most the crate's, so after a release the
next builds compare against that release itself. Candidates are tried
newest-first and the first whose assets list the Linux x86-64 archive wins;
a listed asset that fails to download fails the job. Only when no release
lists the archive does the job pass with a notice. 0.1.1 is the first
baseline. Needs `gh` with a token that can read releases, and `abidiff`.
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import capi_exports  # noqa: E402

# Preference order at one version: the crate release carries the archives.
TAG_PREFIXES = ("vanedb-crate-v", "vanedb-v")
ASSET_SUFFIX = "linux-x86_64.zip"
CURRENT_HEADER = ROOT / "vanedb-capi/include/vanedb_rs_capi.h"


def parse_version(text):
    """(major, minor, patch, is_release, prerelease) so releases sort above
    their prereleases and prereleases compare lexically among themselves."""
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z.-]+))?", text)
    if not match:
        return None
    major, minor, patch, pre = match.groups()
    return (int(major), int(minor), int(patch), pre is None, pre or "")


def tag_version(tag):
    """(version, prefix rank) for a release tag this gate understands."""
    for rank, prefix in enumerate(TAG_PREFIXES):
        if tag.startswith(prefix):
            version = parse_version(tag[len(prefix):])
            return (version, rank) if version else None
    return None


def notice(message):
    print(f"::notice title=C ABI gate::{message}")
    print(message)


def gh(*args):
    try:
        return subprocess.run(["gh", *args], text=True, capture_output=True, check=True).stdout
    except FileNotFoundError:
        raise SystemExit("gh is not installed; the baseline is a GitHub Release asset "
                         "(pass --baseline <archive.zip> to compare locally)") from None
    except subprocess.CalledProcessError as error:
        raise SystemExit(f"gh {' '.join(args)} failed:\n{error.stderr}") from None


def releases(repo):
    return json.loads(gh("release", "list", "--repo", repo, "--limit", "200",
                         "--json", "tagName,isDraft"))


def assets_of(repo, tag):
    return [a["name"] for a in json.loads(gh("release", "view", tag, "--repo", repo,
                                             "--json", "assets"))["assets"]]


def baseline_candidates(current, published):
    """Release tags at most `current`, newest first; at one version the crate
    tag before the Python tag."""
    candidates = []
    for release in published:
        if release.get("isDraft"):
            continue
        parsed = tag_version(release["tagName"])
        if parsed and parsed[0] <= current:
            version, rank = parsed
            candidates.append((version, -rank, release["tagName"]))
    return [tag for _, _, tag in sorted(candidates, reverse=True)]


def select_baseline(current, published, assets):
    """(tag, asset name) of the newest candidate that lists the archive."""
    for tag in baseline_candidates(current, published):
        for name in assets(tag):
            if name.endswith(f"-{ASSET_SUFFIX}") and name.startswith("vanedb-capi-"):
                return tag, name
    return None


def download(repo, tag, asset, directory):
    gh("release", "download", tag, "--repo", repo, "--dir", str(directory), "--pattern", asset)
    archive = directory / asset
    if not archive.exists():
        raise SystemExit(f"release {tag} lists {asset} but it was not downloaded")
    return archive


def extract(archive, directory):
    """(header, shared library) from a vanedb-capi archive."""
    with zipfile.ZipFile(archive) as packaged:
        packaged.extractall(directory)
    headers = list(directory.rglob("vanedb_rs_capi.h"))
    libraries = list(directory.rglob("libvanedb_capi.so"))
    if not headers or not libraries:
        raise SystemExit(f"{archive} does not carry include/vanedb_rs_capi.h and lib/libvanedb_capi.so")
    return headers[0], libraries[0]


def compare_prototypes(baseline_text, current_text):
    """{"removed": [...], "changed": [(old, new)...], "added": [...]}."""
    old, new = capi_exports.prototypes(baseline_text), capi_exports.prototypes(current_text)
    return {
        "removed": sorted(old[n] for n in old if n not in new),
        "changed": sorted((old[n], new[n]) for n in old if n in new and old[n] != new[n]),
        "added": sorted(new[n] for n in new if n not in old),
    }


def verdict(baseline_abi, current_abi, diff, abidiff_status):
    """(exit status, message). Only a version increase permits ABI changes;
    it never permits a failed comparison tool or a version downgrade."""
    broken = diff["removed"] or diff["changed"] or (abidiff_status not in (0, None))
    summary = (f"{len(diff['removed'])} removed, {len(diff['changed'])} changed, "
               f"{len(diff['added'])} added prototypes; abidiff "
               f"{'not run' if abidiff_status is None else f'exit {abidiff_status}'}")
    # libabigail statuses are bit flags: 1/2 mean execution/usage errors,
    # 4/8 mean ABI changes. A signal or unknown bit is not evidence either.
    if abidiff_status is not None and abidiff_status not in (0, 4, 8, 12):
        return 1, f"abidiff failed; compatibility was not established ({summary})"
    if current_abi < baseline_abi:
        return 1, (f"VANEDB_RS_ABI_VERSION must not decrease: baseline {baseline_abi} "
                   f"vs current {current_abi} ({summary})")
    if current_abi > baseline_abi:
        return 0, (f"intentional ABI break: baseline VANEDB_RS_ABI_VERSION {baseline_abi} "
                   f"vs current {current_abi} ({summary})")
    if broken:
        return 1, f"the C ABI changed under the same VANEDB_RS_ABI_VERSION {current_abi} ({summary})"
    return 0, f"compatible with the baseline under VANEDB_RS_ABI_VERSION {current_abi} ({summary})"


def print_diff(diff):
    for prototype in diff["removed"]:
        print(f"  removed: {prototype}")
    for old, new in diff["changed"]:
        print(f"  changed: {old}\n       -> {new}")
    for prototype in diff["added"]:
        print(f"  added:   {prototype}")


def run_abidiff(baseline_so, current_so):
    if not shutil.which("abidiff"):
        raise SystemExit("abidiff is not installed (apt: abigail-tools)")
    command = ["abidiff", "--no-added-syms", "--headers-dir2", str(CURRENT_HEADER.parent),
               str(baseline_so), str(current_so)]
    print("+", " ".join(command), flush=True)
    # Exit status is a bit set: 4 = ABI change, 8 = incompatible change,
    # 1/2 = error or usage. Any nonzero fails under the same ABI version.
    return subprocess.run(command).returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("current", type=Path, help="the freshly built libvanedb_capi.so")
    parser.add_argument("--repo", default="vanedb/vanedb")
    parser.add_argument("--baseline", type=Path, default=None,
                        help="a vanedb-capi-<version>-linux-x86_64.zip to compare against "
                             "instead of a release asset (both layers), or a bare .so "
                             "(abidiff layer only)")
    parser.add_argument("--skip-abidiff", action="store_true",
                        help="prototype layer only, for a host without libabigail")
    args = parser.parse_args()
    if not args.current.exists():
        raise SystemExit(f"{args.current} does not exist; build with cargo build -p vanedb-capi --profile capi")
    version_text = tomllib.loads((ROOT / "vanedb-capi/Cargo.toml").read_text())["package"]["version"]
    current_text = CURRENT_HEADER.read_text(encoding="utf-8")
    current_abi = capi_exports.abi_version(current_text) or 0

    with tempfile.TemporaryDirectory(prefix="vanedb-abi-gate-") as temporary:
        directory = Path(temporary)
        baseline_header = baseline_so = None
        if args.baseline is not None and args.baseline.suffix != ".zip":
            baseline_so = args.baseline
        else:
            archive = args.baseline
            if archive is None:
                selected = select_baseline(parse_version(version_text), releases(args.repo),
                                           lambda tag: assets_of(args.repo, tag))
                if selected is None:
                    notice(f"no release at or below {version_text} lists a vanedb-capi {ASSET_SUFFIX} "
                           f"asset; nothing to compare against")
                    return 0
                tag, asset = selected
                print(f"baseline: {tag} / {asset}")
                archive = download(args.repo, tag, asset, directory)
            baseline_header, baseline_so = extract(archive, directory / "baseline")

        diff = {"removed": [], "changed": [], "added": []}
        baseline_abi = current_abi
        if baseline_header is not None:
            baseline_text = baseline_header.read_text(encoding="utf-8")
            baseline_abi = capi_exports.abi_version(baseline_text) or 0
            diff = compare_prototypes(baseline_text, current_text)
            print(f"prototypes: baseline ABI {baseline_abi}, current ABI {current_abi}")
            print_diff(diff)
        else:
            print("no baseline header: prototype layer skipped (bare .so given)")

        status = None if args.skip_abidiff else run_abidiff(baseline_so, args.current)
        code, message = verdict(baseline_abi, current_abi, diff, status)
        if code == 0:
            notice(message)
        else:
            print(message, file=sys.stderr)
        return code


if __name__ == "__main__":
    sys.exit(main())
