#!/usr/bin/env python3
"""Assert the release validator's wheel tripwires match the build matrices.

The tripwires exist to catch matrix drift and have themselves drifted twice --
once claiming 24 wheels while asserting 16, and once asserting 32 when the
matrices produce 28. Deriving the expected total from the workflow removes the
chance to get it wrong by hand.

A build job produces (matrix combinations) x (interpreters in its maturin args)
wheels: the Linux legs pass four `-i python3.N` flags each, while macOS and
Windows pass a single `-i python` and vary the interpreter through the matrix.
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import yaml

WORKFLOW = Path(__file__).resolve().parents[1] / "workflows" / "publish-rust.yml"
BUILD_JOBS = ("linux", "macos", "windows")


def combinations(matrix: dict) -> int:
    """Number of legs a matrix expands to."""
    axes = [v for k, v in matrix.items() if k not in ("include", "exclude") and isinstance(v, list)]
    if axes:
        # `include` entries that only add keys (a runner per arch) refine legs
        # rather than adding them, so they do not multiply the count.
        return math.prod(len(a) for a in axes)
    return len(matrix.get("include", []))


def interpreters(job: dict) -> int:
    """How many wheels one leg of this job builds."""
    for step in job.get("steps", []):
        args = step.get("with", {}).get("args", "")
        if "-i " in args:
            return len(re.findall(r"-i \S+", args))
    raise SystemExit(f"no maturin args found in job: {job.get('name')}")


def main() -> int:
    text = WORKFLOW.read_text(encoding="utf-8")
    jobs = yaml.safe_load(text)["jobs"]

    per_job = {}
    for name in BUILD_JOBS:
        job = jobs[name]
        per_job[name] = combinations(job["strategy"]["matrix"]) * interpreters(job)
    total = sum(per_job.values())

    declared = int(re.search(r"--wheel-count (\d+)", text).group(1))
    tags = {m[0]: int(m[1]) for m in re.findall(r"--python-tag (cp\d+)=(\d+)", text)}
    prefixes = {m[0]: int(m[1]) for m in re.findall(r"--platform-prefix (\w+)=(\d+)", text)}

    breakdown = ", ".join(f"{n} {c}" for n, c in per_job.items())
    failures = []
    if declared != total:
        failures.append(f"--wheel-count is {declared}; the matrices build {total} ({breakdown})")
    # Every wheel carries exactly one Python tag and one platform prefix, so
    # each set of tripwires must partition the same total.
    if sum(tags.values()) != total:
        failures.append(f"--python-tag counts sum to {sum(tags.values())}, expected {total}")
    if sum(prefixes.values()) != total:
        failures.append(f"--platform-prefix counts sum to {sum(prefixes.values())}, expected {total}")

    for line in failures:
        print(f"wheel matrix drift: {line}", file=sys.stderr)
    if failures:
        return 1
    print(f"wheel tripwires match the matrices: {total} wheels ({breakdown})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
