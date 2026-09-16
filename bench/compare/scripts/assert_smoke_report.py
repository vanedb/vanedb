#!/usr/bin/env python3
"""Assert competitor-harness smoke JSON looks sane (CI only)."""

from __future__ import annotations

import json
import sys


def main() -> int:
    path = sys.argv[1]
    with open(path, encoding="utf-8") as fh:
        report = json.load(fh)
    results = report["results"]
    assert len(results) == 6, results
    for row in results:
        assert row["recall_at_k_by_ef"], row
        assert all(x["mean_recall"] >= 0.5 for x in row["recall_at_k_by_ef"]), row
        if row["engine"] in ("vanedb", "usearch", "hnswlib", "sqlite-vec"):
            assert row.get("delete_ok") is True, row
    print(f"ok: {len(results)} engines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
