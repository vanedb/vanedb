#!/usr/bin/env python3
"""Check the RFC index: every RFC has a valid status line, the README index and
the roadmap link every RFC, and no two RFCs share a number.

Run by CI's workflow-lint job. Exits non-zero with one line per defect.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RFCS = ROOT / "docs/rfcs"
STATUSES = ("draft", "accepted", "implemented", "parked", "rejected")
STATUS_LINE = re.compile(r"^- Status: (?P<status>.+)$", re.MULTILINE)
SUPERSEDED = re.compile(r"^superseded by \d{4}$")


def status_is_valid(value):
    """`accepted; the amendment below is draft` counts: the first word decides."""
    head = value.split(";")[0].split("(")[0].strip().rstrip(",")
    first = head.split(",")[0].strip()
    return first in STATUSES or SUPERSEDED.match(first) is not None


def check(root=ROOT):
    rfcs = root / "docs/rfcs"
    problems = []
    numbers = {}
    files = sorted(p for p in rfcs.glob("[0-9][0-9][0-9][0-9]-*.md") if not p.name.startswith("0000-"))
    if not files:
        return ["no RFC files found under docs/rfcs"]
    index = (rfcs / "README.md").read_text()
    roadmap = (root / "docs/ROADMAP.md").read_text()
    for path in files:
        number = path.name[:4]
        if number in numbers:
            problems.append(f"{path.name}: duplicates RFC number of {numbers[number]}")
        numbers[number] = path.name
        text = path.read_text()
        if not text.startswith(f"# RFC {number}: "):
            problems.append(f"{path.name}: title must start with '# RFC {number}: '")
        match = STATUS_LINE.search(text)
        if match is None:
            problems.append(f"{path.name}: missing '- Status:' line")
        elif not status_is_valid(match.group("status")):
            problems.append(f"{path.name}: status {match.group('status')!r} is not one of {STATUSES} or 'superseded by NNNN'")
        for section in ("## Problem", "## Decision", "## Acceptance criteria"):
            if section not in text:
                problems.append(f"{path.name}: missing section {section!r}")
        if f"]({path.name})" not in index:
            problems.append(f"docs/rfcs/README.md: index does not link {path.name}")
        if f"rfcs/{path.name}" not in roadmap:
            problems.append(f"docs/ROADMAP.md: does not link rfcs/{path.name}")
    return problems


def main():
    problems = check()
    for line in problems:
        print(line)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
