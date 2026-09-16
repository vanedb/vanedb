#!/usr/bin/env python3
"""Render a compare JSON report to markdown for pasting into COMPARISON.md.

Prefer producing markdown on the dedicated machine via `compare run --markdown`.
This helper re-renders an existing publish JSON without re-running engines.

It applies the same *policy* gates as the Rust harness (role, sha pin, params,
full engine set, dedicated attestation, shared-runner refuse). It does **not**
cryptographically prove timings were not hand-edited — paste only JSON you
recorded yourself on dedicated hardware (or attached as evidence on the PR).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PUBLISH_M = 16
PUBLISH_EF_CONSTRUCTION = 200
PUBLISH_K = 10
PUBLISH_SEED = 42
PUBLISH_DIM = 768
PUBLISH_EF = (16, 32, 50, 100)
PUBLISH_ENGINES_COSINE = (
    "vanedb",
    "usearch",
    "hnswlib",
    "instant-distance",
    "hnsw_rs",
)
PUBLISH_ENGINES_L2 = PUBLISH_ENGINES_COSINE + ("sqlite-vec",)
SUMS_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "SHA256SUMS"


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


def load_sums() -> dict[str, str]:
    if not SUMS_PATH.is_file():
        raise SystemExit(f"missing {SUMS_PATH}")
    out: dict[str, str] = {}
    for line in SUMS_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        digest, name = parts[0], parts[-1]
        out[name] = digest
    return out


def recall_value(rec: dict) -> float:
    if "recall_median" in rec:
        return float(rec["recall_median"])
    # legacy field name from earlier harness builds
    return float(rec["mean_recall"])


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} path/to/report.json", file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    report = json.loads(path.read_text())
    if report.get("report_kind") != "vanedb-compare-v1":
        print(
            "refusing to render: report_kind must be vanedb-compare-v1 "
            "(harness-produced JSON only)",
            file=sys.stderr,
        )
        return 1
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
    host_os = report.get("host_os")
    host_arch = report.get("host_arch")
    host_has_avx2 = report.get("host_has_avx2")
    host_android = report.get("host_android")
    if None in (host_os, host_arch, host_has_avx2, host_android):
        print(
            "refusing to render without host_os/host_arch/host_has_avx2/host_android "
            "(harness ≥ vanedb-compare-v1 with host facts)",
            file=sys.stderr,
        )
        return 1
    if hw.startswith("android-arm64-"):
        if host_android is not True:
            print("refusing to render android-* JSON with host_android!=true", file=sys.stderr)
            return 1
    elif hw.startswith("apple-"):
        if host_os != "macos" or host_arch != "aarch64":
            print(
                f"refusing to render apple-* JSON from host_os={host_os!r} arch={host_arch!r}",
                file=sys.stderr,
            )
            return 1
    elif hw.startswith("linux-avx2"):
        if host_os != "linux" or host_has_avx2 is not True:
            print(
                f"refusing to render linux-avx2* JSON from host_os={host_os!r} "
                f"host_has_avx2={host_has_avx2!r}",
                file=sys.stderr,
            )
            return 1
    else:
        print(
            f"refusing to render hardware_label={hw!r}; "
            "need apple-*, linux-avx2*, or android-arm64-*",
            file=sys.stderr,
        )
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
    if "queries_measured" not in report:
        print(
            "refusing to render without queries_measured "
            "(need harness that records measured vs fixture query counts)",
            file=sys.stderr,
        )
        return 1
    if report.get("queries_measured") != report.get("fixture_n_queries"):
        print(
            f"refusing to render: queries_measured={report.get('queries_measured')!r} != "
            f"fixture_n_queries={report.get('fixture_n_queries')!r} "
            "(publish must use the full query set; no --max-queries shrink)",
            file=sys.stderr,
        )
        return 1
    if report.get("fixture_dim") != PUBLISH_DIM:
        print(
            f"refusing to render fixture_dim={report.get('fixture_dim')!r}; "
            f"need dim={PUBLISH_DIM}",
            file=sys.stderr,
        )
        return 1
    if not report.get("recorded_at_utc"):
        print("refusing to render without recorded_at_utc", file=sys.stderr)
        return 1
    rounds = max((r.get("rounds") or 0) for r in report.get("results", []))
    if rounds < 2:
        print("refusing to render rounds < 2", file=sys.stderr)
        return 1

    if "shared_runner" not in report or report.get("shared_runner") is not False:
        print(
            "refusing to render: shared_runner must be present and false "
            "(omit/true means untrusted or shared-runner JSON)",
            file=sys.stderr,
        )
        return 1
    if "dedicated_attested" not in report or report.get("dedicated_attested") is not True:
        print(
            "refusing to render JSON without dedicated_attested=true "
            "(set VANEDB_COMPARE_DEDICATED=1 on the dedicated machine before run)",
            file=sys.stderr,
        )
        return 1

    p = report.get("params") or {}
    if (
        p.get("m") != PUBLISH_M
        or p.get("ef_construction") != PUBLISH_EF_CONSTRUCTION
        or report.get("k") != PUBLISH_K
        or p.get("seed") != PUBLISH_SEED
    ):
        print(
            "refusing to render non-canonical params "
            f"(need M={PUBLISH_M}, ef_construction={PUBLISH_EF_CONSTRUCTION}, "
            f"k={PUBLISH_K}, seed={PUBLISH_SEED})",
            file=sys.stderr,
        )
        return 1

    sha = report.get("fixture_sha256")
    if not sha:
        print("refusing to render without fixture_sha256", file=sys.stderr)
        return 1
    sums = load_sums()
    expected = sums.get("embeddings.vnef")
    if expected is None:
        print(
            "refusing to render: embeddings.vnef not yet listed in "
            f"{SUMS_PATH} (publish fixture not pinned)",
            file=sys.stderr,
        )
        return 1
    if sha != expected:
        print(
            f"refusing to render: fixture_sha256 {sha} != SHA256SUMS embeddings.vnef {expected}",
            file=sys.stderr,
        )
        return 1

    required = PUBLISH_ENGINES_COSINE if metric == "cosine" else PUBLISH_ENGINES_L2
    got = [r["engine"] for r in report["results"]]
    if set(got) != set(required) or len(got) != len(required):
        print(
            f"refusing to render incomplete/extra engine set for {metric}: "
            f"got {got}, need {list(required)}",
            file=sys.stderr,
        )
        return 1

    # Publish JSON must include delete/save evidence for engines that support them.
    for r in report["results"]:
        if r["engine"] in ("vanedb", "usearch", "hnswlib", "sqlite-vec"):
            if r.get("delete_ok") is not True:
                print(
                    f"refusing to render {r['engine']} without delete_ok=true "
                    f"(got {r.get('delete_ok')!r})",
                    file=sys.stderr,
                )
                return 1
            if r.get("file_size_bytes") is None:
                print(
                    f"refusing to render {r['engine']} without file_size_bytes",
                    file=sys.stderr,
                )
                return 1
        efs = tuple(lat["ef"] for lat in r.get("latency_ns_by_ef") or [])
        if r["engine"] == "instant-distance":
            if len(efs) != 1:
                print(
                    "refusing to render instant-distance without exactly one ef row",
                    file=sys.stderr,
                )
                return 1
        elif r["engine"] == "sqlite-vec":
            if len(efs) != 1:
                print(
                    "refusing to render sqlite-vec without exactly one ef row "
                    "(brute force; ef unused)",
                    file=sys.stderr,
                )
                return 1
        elif efs != PUBLISH_EF:
            print(
                f"refusing to render {r['engine']} ef sweep {efs} != {PUBLISH_EF}",
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
        f"n_docs={report['fixture_n_docs']}, n_queries={report['fixture_n_queries']} "
        f"(measured {report['queries_measured']}), "
        f"sha256=`{report['fixture_sha256']}`"
    )
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
        print("| ef | latency/query | spread | recall@10 (median) |")
        print("|---:|---:|---:|---:|")
        for lat, rec in zip(r["latency_ns_by_ef"], r["recall_at_k_by_ef"]):
            print(
                f"| {lat['ef']} | {fmt_ns(lat['median_ns'])} | "
                f"{lat['spread'] * 100:.1f}% | {recall_value(rec):.3f} |"
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
