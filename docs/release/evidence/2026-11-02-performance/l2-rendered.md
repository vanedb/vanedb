### apple-m4-pro (unknown) — metric `l2`

- Date (UTC) / commit: `2026-11-02T08:32:36Z` / `e1761dc`
- Fixture: role=publish, dim=768, n_docs=100000, n_queries=1000 (measured 1000), sha256=`4676911521f13dc6592d102cf3b4fcace85c6b62b3011e9bd57557dfc9c01f2d`
- Params: M=16, ef_construction=200, k=10, seed=42

| Engine | Version | Build (s, median) | Spread | Peak RSS | File size | Delete OK |
|---|---|---:|---:|---:|---:|:---:|
| vanedb | 0.2.0-rc.2 | 32.5740 | 5.1% | 329.12 MiB | 316.12 MiB | yes |
| usearch | 2.21.0 | 49.1395 | 5.5% | 299.19 MiB | 307.16 MiB | yes |
| hnswlib | 0.8.0 | 99.8550 | 5.1% | 2.20 MiB | 307.13 MiB | yes |
| instant-distance | 0.6.1 | 107.4400 | 3.5% | 371.80 MiB | n/a | n/a |
| hnsw_rs | 0.3.4 | 117.9553 | 5.1% | 123.27 MiB | n/a | n/a |
| sqlite-vec | 0.1.6 | 44.4412 | 17.2% | 3.42 MiB | 296.54 MiB | yes |

Recall@10 and latency by ef (median across rounds):

**vanedb**

| ef | latency/query | spread | recall@10 (median) |
|---:|---:|---:|---:|
| 16 | 80.54 µs | 8.5% | 0.928 |
| 32 | 129.66 µs | 9.0% | 0.970 |
| 50 | 183.50 µs | 8.0% | 0.983 |
| 100 | 320.59 µs | 8.9% | 0.993 |

Notes:
- vanedb crate 0.2.0-rc.2; ApproxIndex M=16 efC=200 seed=42; build uses add_batch (same graph topology as serial add; wall time is not sequential-insert peer)

**usearch**

| ef | latency/query | spread | recall@10 (median) |
|---:|---:|---:|---:|
| 16 | 96.41 µs | 7.9% | 0.940 |
| 32 | 151.02 µs | 8.1% | 0.976 |
| 50 | 210.59 µs | 8.0% | 0.986 |
| 100 | 373.49 µs | 6.9% | 0.995 |

Notes:
- connectivity(M)=16 expansion_add(efC)=200; seed not exposed by usearch IndexOptions (requested 42)

**hnswlib**

| ef | latency/query | spread | recall@10 (median) |
|---:|---:|---:|---:|
| 16 | 160.22 µs | 3.2% | 0.932 |
| 32 | 249.34 µs | 0.6% | 0.975 |
| 50 | 345.19 µs | 1.0% | 0.985 |
| 100 | 586.65 µs | 4.4% | 0.994 |

Notes:
- vendored hnswlib v0.8.0; M=16 efC=200 seed=42

**instant-distance**

| ef | latency/query | spread | recall@10 (median) |
|---:|---:|---:|---:|
| 50 | 746.76 µs | 2.1% | 0.995 |

Notes:
- M is a crate constant (32); cannot set harness M
- ef_construction=200 ef_search=50 seed=42
- requested M=16 ignored; instant-distance uses M=32
- per-query ef is ignored; harness emits a single latency/recall row at construction ef_search

**hnsw_rs**

| ef | latency/query | spread | recall@10 (median) |
|---:|---:|---:|---:|
| 16 | 212.74 µs | 3.5% | 0.912 |
| 32 | 338.25 µs | 4.1% | 0.950 |
| 50 | 464.23 µs | 3.8% | 0.962 |
| 100 | 783.90 µs | 2.6% | 0.973 |

Notes:
- max_nb_connection(M)=16 efC=200 max_layer=11; seed not plumbed by hnsw_rs Hnsw::new (requested 42)

**sqlite-vec**

| ef | latency/query | spread | recall@10 (median) |
|---:|---:|---:|---:|
| 50 | 66.34 ms | 3.4% | 1.000 |

Notes:
- brute-force KNN via vec0 for L2 (no ANN; sqlite-vec #25)
- ef unused — single construction-ef latency row only
- metric l2: native vec0 MATCH
