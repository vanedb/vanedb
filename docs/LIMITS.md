# Limits and capacity

What each index can hold, what it costs in memory and on disk, and what it
does not do, as of 0.1.1. Formulas are derived from the source and the format
specifications and are marked **computed**; a figure marked **measured** names
the run it came from. Nothing here is a benchmark. The roadmap RFCs that change
a limit are named beside it.

## Hard limits

| Limit | Value | Where | Changes with |
|---|---|---|---|
| Stored slots per `ApproxIndex`, tombstones included | 100 million | `MAX_ELEMENTS` in the core; also the VNDB v2 capacity cap | none planned |
| Graph levels | 0 to 32 | `MAX_LEVEL` | none planned |
| Dimension | any nonzero `usize`; the loaders bound it against the file length | validation | none planned |
| Ids | `u64`, unique among live entries | validation | none planned |
| Vector components | finite `f32` | validation on add, search and disk build | RFC 0005 adds int8 and binary storage |
| `k` | at least 1; the graph's effective beam is at least `k` | validation | none planned |
| Payload | none stored | | RFC 0009 |
| Metadata filtering | none in 0.1.1; ID lists and predicates implemented for 0.2.0 | external metadata only | RFC 0004 |
| WebAssembly linear memory | 4 GiB (wasm32) | platform | none planned |

## Memory, computed

`n` vectors of dimension `d`.

| Index | Resident bytes | Notes |
|---|---|---|
| `FlatIndex` | `n × (4d + 8)` plus id map | vectors plus `u64` ids; the id map is a hash map over `u64` |
| `ApproxIndex` (M = 16 default) | `n × (4d + 8 + 4 + 1)` for vectors, ids, level, tombstone; plus links: `n × (2M × 8 + ~M × 8 × 0.06)` ≈ `n × 280` bytes; plus per-node `Vec` headers | links are `Vec<Vec<Vec<usize>>>` today: 24 bytes of header per layer per node on top of the `usize` slot numbers; RFC 0013 stores `u32` slots on disk and RFC 0008 moves the resident copy to a flat `u32` layout |
| `DiskIndexBuilder` (build) | `n × (4d + 8)` | buffers every vector until `save`; RFC 0008 streams it |
| `DiskIndex` (open) | page cache only; the id map (`n × ~16` bytes) is resident | read-only memory mapping of the file |

Worked examples at `d = 768` (nomic-embed-text, EmbeddingGemma):

| n | `FlatIndex` | `ApproxIndex` vectors + links | `DiskIndex` resident |
|---|---|---|---|
| 100k | ~310 MB | ~310 MB + ~28 MB | ~2 MB |
| 1M | ~3.1 GB | ~3.1 GB + ~280 MB | ~16 MB |
| 10M | ~31 GB | ~31 GB + ~2.8 GB | ~160 MB |

These are the reason RFC 0005 (quantized storage: int8 is 4× smaller, binary
32×) and RFC 0008 (vectors paged from disk, links resident) exist.

## File size, computed

| Format | Bytes |
|---|---|
| VNDB v1 (`DiskIndex`) | `32 + n × (8 + 4d)` |
| VNDB v2 (`ApproxIndex`) | `96 + Σ per slot (16 + 4d + Σ per layer (8 + 8 × degree)) + continuation` |

`save` writes tombstoned slots too; `compact()` first if size matters.

## Time, what is and is not known

- Search cost: `FlatIndex` and `DiskIndex` scan every vector (linear in `n`);
  `ApproxIndex` visits roughly `ef_search × (levels + 1)` nodes. Absolute
  figures live only in `bench/README.md`, from dedicated hardware, and are
  not repeated here.
- Build cost: `ApproxIndex` construction is single-threaded and superlinear in
  `n`; a 10k × 128 build is timed by `examples/profile_index_build.rs`.
- `compact()` is a full rebuild under the write lock.
- Cold-cache `DiskIndex` search pays one page fault per touched 4 KiB page;
  no on-device figure has been recorded.

## Concurrency

- One index, one `RwLock`: many readers or one writer. Searches hold a read
  lock; `add`, `remove`, `upsert` hold the write lock; `compact()` holds it for
  the whole rebuild.
- No cross-process coordination. A mapped `DiskIndex` file must not change
  while any mapping is open; nothing enforces this (`DiskIndex::open` is
  `unsafe` for this reason).
- No sync, no replication, no encryption at rest.

## What is not supported

- Payload or metadata storage (RFC 0009). Filtered search is absent in 0.1.1
  and implemented for 0.2.0 through external ID lists or predicates (RFC 0004).
  Exact indexes scan once; approximate filtering can return fewer than `k`
  matches at its beam cap. Predicates must not access the index being searched.
- Quantized or compressed vectors (RFC 0005).
- Approximate search over a corpus larger than RAM (RFC 0008).
- Mobile SDKs beyond the C ABI (RFC 0007).
- GPU acceleration of any index (RFC 0001; the Metal feature exposes distance
  scans only).

## Open questions this page cannot answer yet

Recorded in #210, the [capacity study](research/2026-09-capacity-study.md):
what corpus sizes and memory budgets the target users actually have per
segment (mobile, browser, desktop RAG, gateway), what capacity each competitor
supports at what memory cost, and the measured latency and recall trade-off
for the mapped and quantized designs in RFCs 0005 and 0008.
