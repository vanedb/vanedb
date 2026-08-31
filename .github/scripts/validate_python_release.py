#!/usr/bin/env python3
"""Validate the identity and coverage of a Python release artifact set."""

from __future__ import annotations

import argparse
from collections import Counter
from email.parser import Parser
from pathlib import Path
import tarfile
import tomllib
import zipfile

from packaging.utils import canonicalize_name, parse_sdist_filename, parse_wheel_filename
from packaging.version import Version


def expected_count(value: str) -> tuple[str, int]:
    key, separator, count = value.partition("=")
    if not separator or not key:
        raise argparse.ArgumentTypeError("expected KEY=COUNT")
    try:
        return key, int(count)
    except ValueError as error:
        raise argparse.ArgumentTypeError("COUNT must be an integer") from error


def package_metadata(path: Path) -> tuple[str, Version]:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            names = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
            if len(names) != 1:
                raise ValueError(f"{path.name}: expected one wheel METADATA file, found {len(names)}")
            contents = archive.read(names[0]).decode("utf-8")
    else:
        with tarfile.open(path, "r:gz") as archive:
            members = [
                member
                for member in archive.getmembers()
                if member.isfile() and member.name.endswith("/PKG-INFO")
            ]
            if len(members) != 1:
                raise ValueError(f"{path.name}: expected one sdist PKG-INFO file, found {len(members)}")
            extracted = archive.extractfile(members[0])
            if extracted is None:
                raise ValueError(f"{path.name}: could not read PKG-INFO")
            contents = extracted.read().decode("utf-8")

    metadata = Parser().parsestr(contents)
    name = metadata.get("Name")
    version = metadata.get("Version")
    if not name or not version:
        raise ValueError(f"{path.name}: package metadata is missing Name or Version")
    return canonicalize_name(name), Version(version)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--distribution", required=True)
    parser.add_argument("--pyproject", type=Path, required=True)
    parser.add_argument("--wheel-count", type=int, required=True)
    parser.add_argument("--sdist-count", type=int, required=True)
    parser.add_argument("--python-tag", action="append", type=expected_count, default=[])
    parser.add_argument("--platform-prefix", action="append", type=expected_count, default=[])
    args = parser.parse_args()

    expected_name = canonicalize_name(args.distribution)
    expected_version = Version(
        tomllib.loads(args.pyproject.read_text(encoding="utf-8"))["project"]["version"]
    )
    wheels = sorted(args.directory.glob("*.whl"))
    sdists = sorted(args.directory.glob("*.tar.gz"))
    other_files = sorted(
        path.name
        for path in args.directory.iterdir()
        if path.is_file() and path not in {*wheels, *sdists}
    )

    if len(wheels) != args.wheel_count:
        raise SystemExit(f"expected {args.wheel_count} wheels, found {len(wheels)}")
    if len(sdists) != args.sdist_count:
        raise SystemExit(f"expected {args.sdist_count} sdists, found {len(sdists)}")
    if other_files:
        raise SystemExit(f"unexpected release files: {other_files}")

    python_counts: Counter[str] = Counter()
    platform_counts: Counter[str] = Counter()
    expected_python = dict(args.python_tag)
    expected_platform = dict(args.platform_prefix)

    for wheel in wheels:
        filename_name, filename_version, _build, tags = parse_wheel_filename(wheel.name)
        if canonicalize_name(filename_name) != expected_name or filename_version != expected_version:
            raise SystemExit(
                f"{wheel.name}: filename identity does not match "
                f"{expected_name} {expected_version}"
            )

        interpreters = {tag.interpreter for tag in tags} & expected_python.keys()
        platforms = {
            prefix
            for prefix in expected_platform
            if any(tag.platform.startswith(prefix) for tag in tags)
        }
        if len(interpreters) != 1:
            raise SystemExit(f"{wheel.name}: expected one known Python tag, found {sorted(interpreters)}")
        if len(platforms) != 1:
            raise SystemExit(f"{wheel.name}: expected one known platform family, found {sorted(platforms)}")
        python_counts.update(interpreters)
        platform_counts.update(platforms)

    for sdist in sdists:
        filename_name, filename_version = parse_sdist_filename(sdist.name)
        if canonicalize_name(filename_name) != expected_name or filename_version != expected_version:
            raise SystemExit(
                f"{sdist.name}: filename identity does not match "
                f"{expected_name} {expected_version}"
            )

    for artifact in [*wheels, *sdists]:
        metadata_name, metadata_version = package_metadata(artifact)
        if metadata_name != expected_name or metadata_version != expected_version:
            raise SystemExit(
                f"{artifact.name}: embedded metadata is {metadata_name} {metadata_version}, "
                f"expected {expected_name} {expected_version}"
            )

    if python_counts != Counter(expected_python):
        raise SystemExit(f"Python wheel coverage is {dict(python_counts)}, expected {expected_python}")
    if platform_counts != Counter(expected_platform):
        raise SystemExit(f"platform wheel coverage is {dict(platform_counts)}, expected {expected_platform}")

    print(
        f"validated {expected_name} {expected_version}: "
        f"{len(wheels)} wheels, {len(sdists)} sdist"
    )


if __name__ == "__main__":
    main()
