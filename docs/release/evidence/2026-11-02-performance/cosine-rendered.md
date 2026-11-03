### apple-m4-pro (unknown) — metric `cosine`

- Date (UTC) / commit: `2026-11-02T07:32:29Z` / `e1761dc`
- Fixture: role=publish, dim=768, n_docs=100000, n_queries=1000 (measured 1000), sha256=`4676911521f13dc6592d102cf3b4fcace85c6b62b3011e9bd57557dfc9c01f2d`
- Params: M=16, ef_construction=200, k=10, seed=42

| Engine | Version | Build (s, median) | Spread | Peak RSS | File size | Delete OK |
|---|---|---:|---:|---:|---:|:---:|
| vanedb | 0.2.0-rc.2 | 42.1981 | 4.4% | 329.30 MiB | 316.26 MiB | yes |
| usearch | 2.21.0 | 55.0176 | 2.8% | 287.98 MiB | 307.16 MiB | yes |
| hnswlib | 0.8.0 | 99.8138 | 1.9% | 2.03 MiB | 307.13 MiB | yes |
| instant-distance | 0.6.1 | 127.5928 | 0.7% | 371.39 MiB | n/a | n/a |
| hnsw_rs | 0.3.4 | 159.3735 | 2.5% | 129.20 MiB | n/a | n/a |

Recall@10 and latency by ef (median across rounds):

**vanedb**

| ef | latency/query | spread | recall@10 (median) |
|---:|---:|---:|---:|
| 16 | 93.46 µs | 6.4% | 0.935 |
| 32 | 150.51 µs | 3.2% | 0.970 |
| 50 | 212.29 µs | 1.9% | 0.983 |
| 100 | 365.39 µs | 4.2% | 0.993 |

Notes:
- vanedb crate 0.2.0-rc.2; ApproxIndex M=16 efC=200 seed=42; build uses add_batch (same graph topology as serial add; wall time is not sequential-insert peer)

**usearch**

| ef | latency/query | spread | recall@10 (median) |
|---:|---:|---:|---:|
| 16 | 104.53 µs | 6.6% | 0.943 |
| 32 | 165.42 µs | 5.9% | 0.975 |
| 50 | 234.34 µs | 3.9% | 0.986 |
| 100 | 398.38 µs | 3.5% | 0.995 |

Notes:
- connectivity(M)=16 expansion_add(efC)=200; seed not exposed by usearch IndexOptions (requested 42)

**hnswlib**

| ef | latency/query | spread | recall@10 (median) |
|---:|---:|---:|---:|
| 16 | 161.11 µs | 2.0% | 0.944 |
| 32 | 251.53 µs | 1.8% | 0.977 |
| 50 | 348.50 µs | 1.9% | 0.986 |
| 100 | 595.23 µs | 2.5% | 0.995 |

Notes:
- vendored hnswlib v0.8.0; M=16 efC=200 seed=42
- cosine via InnerProductSpace on L2-normalized vectors (hnswlib convention)

**instant-distance**

| ef | latency/query | spread | recall@10 (median) |
|---:|---:|---:|---:|
| 50 | 877.78 µs | 2.2% | 0.993 |

Notes:
- M is a crate constant (32); cannot set harness M
- ef_construction=200 ef_search=50 seed=42
- requested M=16 ignored; instant-distance uses M=32
- per-query ef is ignored; harness emits a single latency/recall row at construction ef_search

**hnsw_rs**

| ef | latency/query | spread | recall@10 (median) |
|---:|---:|---:|---:|
| 16 | 290.14 µs | 3.4% | 0.920 |
| 32 | 472.99 µs | 5.8% | 0.955 |
| 50 | 637.82 µs | 2.5% | 0.963 |
| 100 | 1.06 ms | 1.6% | 0.975 |

Notes:
- max_nb_connection(M)=16 efC=200 max_layer=11; seed not plumbed by hnsw_rs Hnsw::new (requested 42)
