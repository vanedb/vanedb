#!/usr/bin/env python3
"""Measure filtered recall on the pinned RFC 0003 embedding fixture.

Requires an installed vanedb wheel and numpy. Run outside the checkout so
imports resolve to the wheel. This reports recall, not performance timings.
"""

import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
from vanedb import ApproxIndex, FlatIndex, Metric


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--ef-search", type=int, default=50)
    parser.add_argument("--caps", type=int, nargs="+", default=[200, 1600])
    args = parser.parse_args()
    if args.queries <= 0 or args.ef_search <= 0 or any(cap <= 0 for cap in args.caps):
        parser.error("queries, ef-search, and caps must be positive")

    digest = hashlib.sha256(args.fixture.read_bytes()).hexdigest()
    pins = dict(line.split()[::-1] for line in
                args.fixture.with_name("SHA256SUMS").read_text().splitlines())
    if pins.get(args.fixture.name) != digest:
        parser.error("fixture does not match SHA256SUMS")
    with args.fixture.open("rb") as source:
        magic, version, dim, n, nq, metric, reserved = struct.unpack("<4s6I", source.read(28))
    if (magic, version, metric, reserved) != (b"VNEF", 1, 1, 0) or not dim:
        parser.error("expected a VNEF v1 cosine embedding fixture")
    if args.queries > nq or n < 1000:
        parser.error("insufficient documents or queries")
    if args.fixture.stat().st_size != 28 + (n + nq) * dim * 4 + n * 8:
        parser.error("unexpected fixture length")

    vectors = np.memmap(args.fixture, dtype="<f4", mode="r", offset=28, shape=(n, dim))
    queries = np.memmap(args.fixture, dtype="<f4", mode="r", offset=28 + n * dim * 4,
                        shape=(nq, dim))
    ids = np.memmap(args.fixture, dtype="<u8", mode="r", offset=28 + (n + nq) * dim * 4,
                    shape=(n,))
    exact = FlatIndex(dim, Metric.COSINE)
    graph = ApproxIndex(dim, Metric.COSINE, capacity=n, m=16, ef_construction=200, seed=42)
    exact.add_batch(ids, vectors)
    graph.add_batch(ids, vectors)

    rows = []
    for stride in (2, 10, 100):
        allowed = np.sort(ids[::stride])
        allowed_set = set(map(int, allowed))
        reference = [set(i for i, _ in exact.search(q, 10, allow_ids=allowed))
                     for q in queries[:args.queries]]
        for cap in args.caps:
            hits = [graph.search(q, 10, ef_search=args.ef_search,
                                 max_ef_search=cap, allow_ids=allowed)
                    for q in queries[:args.queries]]
            assert all(i in allowed_set for result in hits for i, _ in result)
            rows.append({
                "selectivity": len(allowed) / n,
                "ef_search": args.ef_search,
                "max_ef_search": cap,
                "recall_at_10": sum(len({i for i, _ in h} & r) / len(r)
                                    for h, r in zip(hits, reference)) / args.queries,
                "fraction_with_10_results": sum(len(h) == 10 for h in hits) / args.queries,
                "mean_result_count": sum(map(len, hits)) / args.queries,
            })
    print(json.dumps({"fixture_sha256": digest, "documents": n, "dimensions": dim,
                      "queries": args.queries, "metric": "cosine", "k": 10,
                      "m": 16, "ef_construction": 200, "seed": 42, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
