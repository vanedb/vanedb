#!/usr/bin/env python3
"""Prove a test run exercises the installed wheel, not the source tree.

The classic false pass: pytest runs from the repository root, `import vanedb`
resolves to the checked-out directory, and the wheel under test is never
loaded. This repository is unusually exposed to it — the Rust crate directory
is literally named `vanedb`, so it is importable as a namespace package.

Usage: check_wheel_install.py <import name>

Exits non-zero unless the name resolves to a real module inside this
interpreter's site-packages.
"""

import importlib
import sys
import sysconfig
from importlib import metadata
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    name = sys.argv[1]

    # Nothing in the checkout may satisfy the import, including the implicit
    # "" entry for the working directory.
    sys.path[:] = [
        p for p in sys.path if p and not Path(p).resolve().is_relative_to(REPO)
    ]

    try:
        module = importlib.import_module(name)
    except ImportError as err:
        print(f"{name} is not installed: {err}", file=sys.stderr)
        return 1

    location = getattr(module, "__file__", None)
    if location is None:
        print(
            f"{name} resolved to a namespace package, not an installed "
            f"distribution (path: {list(getattr(module, '__path__', []))})",
            file=sys.stderr,
        )
        return 1

    roots = {
        Path(sysconfig.get_paths()[key]).resolve() for key in ("purelib", "platlib")
    }
    resolved = Path(location).resolve()
    if not any(resolved.is_relative_to(root) for root in roots):
        print(
            f"{name} resolved to {resolved}, outside site-packages "
            f"({', '.join(sorted(map(str, roots)))})",
            file=sys.stderr,
        )
        return 1

    try:
        version = metadata.version(name)
    except metadata.PackageNotFoundError:
        version = "unknown version"
    print(f"{name} {version} imported from {resolved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
