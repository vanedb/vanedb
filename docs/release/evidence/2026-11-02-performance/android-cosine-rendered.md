### android-arm64-emulator (emu64a) — metric `cosine`

- Date (UTC) / commit: `2026-11-02T10:17:50Z` / `unknown`
- Fixture: role=publish, dim=768, n_docs=100000, n_queries=1000 (measured 1000), sha256=`4676911521f13dc6592d102cf3b4fcace85c6b62b3011e9bd57557dfc9c01f2d`
- Params: M=16, ef_construction=200, k=10, seed=42

| Engine | Version | Build (s, median) | Spread | Peak RSS | File size | Delete OK |
|---|---|---:|---:|---:|---:|:---:|
| vanedb | 0.2.0-rc.2 | 45.8733 | 7.8% | n/a | 316.26 MiB | yes |
| usearch | 2.21.0 | 59.0057 | 3.0% | n/a | 307.14 MiB | yes |
| hnswlib | 0.8.0 | 169.6426 | 0.2% | n/a | 307.13 MiB | yes |
| instant-distance | 0.6.1 | 405.7684 | 0.3% | n/a | n/a | n/a |
| hnsw_rs | 0.3.4 | 174.9291 | 4.0% | n/a | n/a | n/a |

Recall@10 and latency by ef (median across rounds):

**vanedb**

| ef | latency/query | spread | recall@10 (median) |
|---:|---:|---:|---:|
| 16 | 104.72 µs | 3.3% | 0.935 |
| 32 | 170.20 µs | 5.3% | 0.970 |
| 50 | 239.14 µs | 6.2% | 0.983 |
| 100 | 418.73 µs | 5.6% | 0.993 |

Notes:
- vanedb crate 0.2.0-rc.2; ApproxIndex M=16 efC=200 seed=42; build uses add_batch (same graph topology as serial add; wall time is not sequential-insert peer)

**usearch**

| ef | latency/query | spread | recall@10 (median) |
|---:|---:|---:|---:|
| 16 | 119.41 µs | 2.1% | 0.944 |
| 32 | 187.01 µs | 5.2% | 0.974 |
| 50 | 262.10 µs | 1.4% | 0.986 |
| 100 | 457.05 µs | 4.5% | 0.995 |

Notes:
- connectivity(M)=16 expansion_add(efC)=200; seed not exposed by usearch IndexOptions (requested 42)

**hnswlib**

| ef | latency/query | spread | recall@10 (median) |
|---:|---:|---:|---:|
| 16 | 274.52 µs | 5.4% | 0.944 |
| 32 | 423.66 µs | 2.0% | 0.977 |
| 50 | 584.34 µs | 1.2% | 0.986 |
| 100 | 1.00 ms | 1.8% | 0.995 |

Notes:
- vendored hnswlib v0.8.0; M=16 efC=200 seed=42
- cosine via InnerProductSpace on L2-normalized vectors (hnswlib convention)

**instant-distance**

| ef | latency/query | spread | recall@10 (median) |
|---:|---:|---:|---:|
| 50 | 957.64 µs | 5.0% | 0.995 |

Notes:
- M is a crate constant (32); cannot set harness M
- ef_construction=200 ef_search=50 seed=42
- requested M=16 ignored; instant-distance uses M=32
- per-query ef is ignored; harness emits a single latency/recall row at construction ef_search

**hnsw_rs**

| ef | latency/query | spread | recall@10 (median) |
|---:|---:|---:|---:|
| 16 | 350.79 µs | 12.3% | 0.919 |
| 32 | 566.14 µs | 13.6% | 0.956 |
| 50 | 778.47 µs | 5.7% | 0.965 |
| 100 | 1.30 ms | 7.1% | 0.975 |

Notes:
- max_nb_connection(M)=16 efC=200 max_layer=11; seed not plumbed by hnsw_rs Hnsw::new (requested 42)
