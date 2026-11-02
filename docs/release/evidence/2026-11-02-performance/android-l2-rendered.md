### android-arm64-emulator (emu64a) — metric `l2`

- Date (UTC) / commit: `2026-11-02T11:32:50Z` / `unknown`
- Fixture: role=publish, dim=768, n_docs=100000, n_queries=1000 (measured 1000), sha256=`4676911521f13dc6592d102cf3b4fcace85c6b62b3011e9bd57557dfc9c01f2d`
- Params: M=16, ef_construction=200, k=10, seed=42

| Engine | Version | Build (s, median) | Spread | Peak RSS | File size | Delete OK |
|---|---|---:|---:|---:|---:|:---:|
| vanedb | 0.2.0-rc.2 | 37.6520 | 7.3% | n/a | 316.12 MiB | yes |
| usearch | 2.21.0 | 55.4377 | 3.1% | n/a | 307.14 MiB | yes |
| hnswlib | 0.8.0 | 108.4466 | 0.8% | n/a | 307.13 MiB | yes |
| instant-distance | 0.6.1 | 318.2067 | 0.2% | n/a | n/a | n/a |
| hnsw_rs | 0.3.4 | 135.0904 | 4.0% | n/a | n/a | n/a |
| sqlite-vec | 0.1.6 | 136.5145 | 0.7% | n/a | 296.54 MiB | yes |

Recall@10 and latency by ef (median across rounds):

**vanedb**

| ef | latency/query | spread | recall@10 (median) |
|---:|---:|---:|---:|
| 16 | 92.82 µs | 8.3% | 0.928 |
| 32 | 145.03 µs | 6.2% | 0.970 |
| 50 | 203.88 µs | 9.1% | 0.983 |
| 100 | 365.52 µs | 13.0% | 0.993 |

Notes:
- vanedb crate 0.2.0-rc.2; ApproxIndex M=16 efC=200 seed=42; build uses add_batch (same graph topology as serial add; wall time is not sequential-insert peer)

**usearch**

| ef | latency/query | spread | recall@10 (median) |
|---:|---:|---:|---:|
| 16 | 110.42 µs | 3.1% | 0.938 |
| 32 | 177.52 µs | 4.6% | 0.977 |
| 50 | 245.76 µs | 5.7% | 0.985 |
| 100 | 425.31 µs | 1.8% | 0.994 |

Notes:
- connectivity(M)=16 expansion_add(efC)=200; seed not exposed by usearch IndexOptions (requested 42)

**hnswlib**

| ef | latency/query | spread | recall@10 (median) |
|---:|---:|---:|---:|
| 16 | 174.93 µs | 1.5% | 0.932 |
| 32 | 278.75 µs | 5.5% | 0.975 |
| 50 | 379.80 µs | 3.4% | 0.985 |
| 100 | 646.66 µs | 1.5% | 0.994 |

Notes:
- vendored hnswlib v0.8.0; M=16 efC=200 seed=42

**instant-distance**

| ef | latency/query | spread | recall@10 (median) |
|---:|---:|---:|---:|
| 50 | 769.55 µs | 3.2% | 0.995 |

Notes:
- M is a crate constant (32); cannot set harness M
- ef_construction=200 ef_search=50 seed=42
- requested M=16 ignored; instant-distance uses M=32
- per-query ef is ignored; harness emits a single latency/recall row at construction ef_search

**hnsw_rs**

| ef | latency/query | spread | recall@10 (median) |
|---:|---:|---:|---:|
| 16 | 271.79 µs | 6.3% | 0.915 |
| 32 | 441.61 µs | 11.6% | 0.953 |
| 50 | 609.60 µs | 17.6% | 0.963 |
| 100 | 1.01 ms | 6.9% | 0.973 |

Notes:
- max_nb_connection(M)=16 efC=200 max_layer=11; seed not plumbed by hnsw_rs Hnsw::new (requested 42)

**sqlite-vec**

| ef | latency/query | spread | recall@10 (median) |
|---:|---:|---:|---:|
| 50 | 66.95 ms | 12.7% | 1.000 |

Notes:
- brute-force KNN via vec0 for L2 (no ANN; sqlite-vec #25)
- ef unused — single construction-ef latency row only
- metric l2: native vec0 MATCH
