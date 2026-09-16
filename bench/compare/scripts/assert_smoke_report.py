#!/usr/bin/env python3
"""Assert competitor-harness smoke JSON looks sane (CI only)."""

from __future__ import annotations

import json
import sys

PUBLISH_EF = {16, 32, 50, 100}
EXPECTED = {
    "vanedb",
    "usearch",
    "hnswlib",
    "instant-distance",
    "hnsw_rs",
    "sqlite-vec",
}


def main() -> int:
    path = sys.argv[1]
    with open(path, encoding="utf-8") as fh:
        report = json.load(fh)
    assert report.get("fixture_role") == "smoke", report.get("fixture_role")
    assert report.get("metric") == "cosine", report.get("metric")
    assert report.get("recorded_at_utc"), report
    assert report.get("report_kind") == "vanedb-compare-v1", report.get("report_kind")
    results = report["results"]
    engines = {row["engine"] for row in results}
    assert engines == EXPECTED, engines
    for row in results:
        assert row["recall_at_k_by_ef"], row
        # Smoke queries are near-copies of docs; healthy HNSW should be near-perfect.
        assert all(x["recall_median"] >= 0.85 for x in row["recall_at_k_by_ef"]), row
        efs = {x["ef"] for x in row["latency_ns_by_ef"]}
        if row["engine"] in ("instant-distance", "sqlite-vec"):
            # Brute/construction-ef only — must NOT echo the full publish sweep.
            assert len(row["latency_ns_by_ef"]) == 1, row["latency_ns_by_ef"]
        else:
            assert efs == PUBLISH_EF, (row["engine"], efs)
        if row["engine"] in ("vanedb", "usearch", "hnswlib", "sqlite-vec"):
            assert row.get("delete_ok") is True, row
        if row["engine"] in ("usearch", "hnsw_rs"):
            assert any("seed" in n.lower() for n in row.get("notes", [])), row.get("notes")
        if row["engine"] == "vanedb":
            assert any("add_batch" in n for n in row.get("notes", [])), row.get("notes")
    print(f"ok: {len(results)} engines role=smoke metric=cosine")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
