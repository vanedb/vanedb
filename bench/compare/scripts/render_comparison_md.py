#!/usr/bin/env python3
"""Render a compare JSON report to markdown for pasting into COMPARISON.md.

Prefer producing markdown on the dedicated machine via `compare run --markdown`.
This helper re-renders an existing publish JSON without re-running engines.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def fmt_bytes(v: int | None) -> str:
    if v is None:
        return "n/a"
    if v >= 1 << 30:
        return f"{v / (1 << 30):.2f} GiB"
    if v >= 1 << 20:
        return f"{v / (1 << 20):.2f} MiB"
    if v >= 1 << 10:
        return f"{v / (1 << 10):.2f} KiB"
    return f"{v} B"


def fmt_ns(ns: float) -> str:
    if ns >= 1_000_000:
        return f"{ns / 1_000_000:.2f} ms"
    if ns >= 1_000:
        return f"{ns / 1_000:.2f} µs"
    return f"{ns:.0f} ns"


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} path/to/report.json", file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    report = json.loads(path.read_text())
    role = report.get("fixture_role", "?")
    if role != "publish":
        print(
            f"refusing to render fixture_role={role!r}; "
            "only publish JSON belongs in COMPARISON.md",
            file=sys.stderr,
        )
        return 1
    hw = report.get("hardware_label", "unlabelled")
    if not hw or hw == "unlabelled":
        print("refusing to render unlabelled hardware_label", file=sys.stderr)
        return 1
    metric = report.get("metric")
    if metric not in ("cosine", "l2"):
        print(f"refusing to render missing/unknown metric={metric!r}", file=sys.stderr)
        return 1
    if report.get("fixture_n_docs", 0) < 100_000:
        print("refusing to render fixture_n_docs < 100000", file=sys.stderr)
        return 1
    if report.get("fixture_n_queries", 0) < 1000:
        print("refusing to render fixture_n_queries < 1000", file=sys.stderr)
        return 1
    if not report.get("recorded_at_utc"):
        print("refusing to render without recorded_at_utc", file=sys.stderr)
        return 1
    rounds = max((r.get("rounds") or 0) for r in report.get("results", []))
    if rounds < 2:
        print("refusing to render rounds < 2", file=sys.stderr)
        return 1
    # Publish JSON must include delete/save evidence for engines that support them.
    for r in report["results"]:
        if r["engine"] in ("vanedb", "usearch", "hnswlib", "sqlite-vec"):
            if r.get("delete_ok") is None:
                print(
                    f"refusing to render {r['engine']} without delete_ok",
                    file=sys.stderr,
                )
                return 1
        if r["engine"] in ("vanedb", "usearch", "hnswlib", "sqlite-vec", "instant-distance"):
            # instant-distance has no save; others that support save must report size
            if r["engine"] != "instant-distance" and r["engine"] != "hnsw_rs":
                if r.get("file_size_bytes") is None and r["engine"] in (
                    "vanedb",
                    "usearch",
                    "sqlite-vec",
                ):
                    print(
                        f"refusing to render {r['engine']} without file_size_bytes",
                        file=sys.stderr,
                    )
                    return 1
    if any(r.get("engine") == "sqlite-vec" and metric == "cosine" for r in report["results"]):
        print(
            "refusing to render cosine report that includes sqlite-vec",
            file=sys.stderr,
        )
        return 1

    when = report.get("recorded_at_utc", "unknown")
    print(f"### {hw} ({report.get('hostname', '?')}) — metric `{metric}`\n")
    print(f"- Date (UTC) / commit: `{when}` / `{report.get('commit', '?')}`")
    print(
        f"- Fixture: role={role}, dim={report['fixture_dim']}, "
        f"n_docs={report['fixture_n_docs']}, n_queries={report['fixture_n_queries']}, "
        f"sha256=`{report['fixture_sha256']}`"
    )
    p = report["params"]
    print(
        f"- Params: M={p['m']}, ef_construction={p['ef_construction']}, "
        f"k={report['k']}, seed={p['seed']}\n"
    )
    print(
        "| Engine | Version | Build (s, median) | Spread | Peak RSS | File size | Delete OK |"
    )
    print("|---|---|---:|---:|---:|---:|:---:|")
    for r in report["results"]:
        dok = r.get("delete_ok")
        dok_s = "yes" if dok is True else "FAIL" if dok is False else "n/a"
        print(
            f"| {r['engine']} | {r['version']} | {r['build_secs_median']:.4f} | "
            f"{r['build_secs_spread'] * 100:.1f}% | {fmt_bytes(r.get('peak_rss_bytes'))} | "
            f"{fmt_bytes(r.get('file_size_bytes'))} | {dok_s} |"
        )
    print("\nRecall@10 and latency by ef (median across rounds):\n")
    for r in report["results"]:
        print(f"**{r['engine']}**\n")
        print("| ef | latency/query | spread | recall@10 |")
        print("|---:|---:|---:|---:|")
        for lat, rec in zip(r["latency_ns_by_ef"], r["recall_at_k_by_ef"]):
            print(
                f"| {lat['ef']} | {fmt_ns(lat['median_ns'])} | "
                f"{lat['spread'] * 100:.1f}% | {rec['mean_recall']:.3f} |"
            )
        print()
        if r.get("notes"):
            print("Notes:")
            for n in r["notes"]:
                print(f"- {n}")
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
