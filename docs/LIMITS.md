# Limits and capacity

What each index can hold, what it costs in memory and on disk, and what it
does not do, as of 0.1.1 plus the unreleased 0.2.0 filtered-search API.
Formulas are derived from the source and the format specifications and are
marked **computed**; a figure marked **measured** names
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
| Metadata filtering | external ID allow/deny lists and predicates (0.2.0, unreleased); no metadata is stored | `SearchParams::filter` on every index (RFC 0004) | RFC 0009 adds stored payload |
| Filtered graph beam cap (`max_ef_search`) | default 4 × the effective initial beam; raised to at least that beam and capped at the stored slot count, tombstones included | `ApproxIndex::search_with`; bounds the beam, not the distance evaluations | none planned |
| Allow/deny list length | no limit beyond address space (the C ABI rejects a length above `isize::MAX / 8`); entries must be strictly ascending with no duplicates | validation on every search | none planned |
| Filter predicates | synchronous, called on the searching thread, possibly more than once per id on the graph; run under the index's read lock (`FlatIndex`, `ApproxIndex`) so they must not touch the searched index, though they may consult other indexes; a C callback must not unwind a foreign exception or `longjmp` | `Filter::Predicate`; C `vanedb_rs_filter_fn` | none planned |
| `ef_search` / `max_ef_search` on `FlatIndex` and `DiskIndex` | ignored: exact scans have no beam, only the filter is read | `SearchParams` | none planned |
| WebAssembly linear memory | 4 GiB (wasm32) | platform | none planned |

## Memory, computed

`n` vectors of dimension `d`.

| Index | Resident bytes | Notes |
|---|---|---|
| `FlatIndex` | `n × (4d + 8)` plus id map | vectors plus `u64` ids; the id map is a hash map over `u64` costing `17 × B(n) + 16` bytes, 19 to 39 per entry, with `B(n) = next_power_of_two(⌈8n / 7⌉)` (computed; matched to the byte by the allocator check in the [capacity study](research/capacity.md) §4.4). Exact for `add_batch`; `add` grows the vectors by `Vec` doubling |
| `ApproxIndex` (M = 16 default) | `n × (4d + 8 + 4 + 1)` for vectors, ids, level, tombstone; plus links: `n × (2M × 8 + ~M × 8 × 0.06)` ≈ `n × 280` bytes; plus per-node `Vec` headers (measured ~297 bytes per node on random vectors at n = 8192, capacity study §4.4) | links are `Vec<Vec<Vec<usize>>>` today: 24 bytes of header per layer per node on top of the `usize` slot numbers; RFC 0013 stores `u32` slots on disk and RFC 0008 moves the resident copy to a flat `u32` layout |
| `DiskIndexBuilder` (build) | `n × (4d + 8)` | buffers every vector until `save`; RFC 0008 streams it |
| `DiskIndex` (open) | page cache only; the id map (`17 × B(n) + 16` bytes, 19 to 39 per entry; computed, checked against the allocator in the capacity study §4.4) is resident | read-only memory mapping of the file |

Worked examples at `d = 768` (nomic-embed-text, EmbeddingGemma):

| n | `FlatIndex` | `ApproxIndex` vectors + links | `DiskIndex` resident |
|---|---|---|---|
| 100k | ~310 MB | ~310 MB + ~28 MB | ~2.2 MB |
| 1M | ~3.1 GB | ~3.1 GB + ~280 MB | ~36 MB |
| 10M | ~31 GB | ~31 GB + ~2.8 GB | ~285 MB |

`DiskIndex` resident figures are the id map alone (computed from the hash
table's bucket rule; the same formula was measured at 34.0 bytes per entry
for n = 8192 and 27.9 for n = 10,000 in the capacity study §4.4).

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
- Filter predicates run inside that read lock, so a predicate that calls
  into the same index, or waits for another thread to modify it, can
  deadlock. `DiskIndex` is immutable and has no lock.
- No cross-process coordination. A mapped `DiskIndex` file must not change
  while any mapping is open; nothing enforces this (`DiskIndex::open` is
  `unsafe` for this reason).
- No sync, no replication, no encryption at rest.

## What is not supported

- Payload or metadata storage (RFC 0009). Filtered search (0.2.0, unreleased)
  works through external ID lists or predicates (RFC 0004): exact indexes
  scan once; approximate filtering can return fewer than `k` matches at its
  beam cap. Predicates must not access the index being searched.
- Quantized or compressed vectors (RFC 0005).
- Approximate search over a corpus larger than RAM (RFC 0008).
- Mobile SDKs beyond the C ABI (RFC 0007).
- GPU acceleration of any index (RFC 0001, #208). The experimental `gpu-metal`
  Cargo feature builds only on macOS and exposes `vanedb::gpu::MetalCompute`:
  a standalone API that uploads a caller-supplied vector matrix to a Metal
  buffer and runs L2, cosine and dot-product distance scans against it.
  `FlatIndex`, `ApproxIndex` and `DiskIndex` never call it; enabling the
  feature changes nothing about how any index builds or searches, and no
  binding exposes it. It has no benchmark showing a gain over the CPU kernels.
  Whether it is finished into index acceleration or removed is decided after
  0.3.0 (#257).

## Open questions this page cannot answer yet

Answered in part by the [capacity study](research/capacity.md) (#210):
corpus sizes and budgets per segment, competitor capacity, and computed
per-vector costs. Still open: the measured latency and recall trade-off for
the mapped and quantized designs in RFCs 0005 and 0008 (study §8.1).
