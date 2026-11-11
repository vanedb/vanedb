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
| Filtered graph beam cap (`max_ef_search`) | default 4 × the effective initial beam; raised to at least that beam and capped at the stored slot count, tombstones included; one 20k-vector L2 experiment at 0.2% selectivity and `k = 10` under-filled 180/200 queries; tune on the actual workload, since no universal selectivity floor is established (#304) | `ApproxIndex::search_with`; bounds the beam width alone, not the visited nodes or distance evaluations, and widening runs pass after pass up to it until `k` matches are found or every stored slot has been scored (PR #263) | none planned |
| Allow/deny list length | no limit beyond address space (the C ABI rejects a length above `isize::MAX / 8`); entries must be strictly ascending with no duplicates | validation on every search | none planned |
| Filter predicates | synchronous, called on the searching thread, possibly more than once per id on the graph; run under the index's read lock (`FlatIndex`, `ApproxIndex`) so they must not touch the searched index, though they may consult other indexes; a C callback must not unwind a foreign exception or `longjmp` | `Filter::Predicate`; C `vanedb_rs_filter_fn` | none planned |
| `ef_search` / `max_ef_search` on `FlatIndex` and `DiskIndex` | ignored: exact scans have no beam, only the filter is read | `SearchParams` | none planned |
| WebAssembly linear memory | 4 GiB (wasm32) | platform | none planned |

## Graph reachability and exhaustive queries

A graph built by this release keeps every stored vector reachable: layer 0
carries a spanning tree whose edges pruning never removes (#299). Releases up
to 0.2.0-rc.2, in both engines, could prune every inbound link to a DOT
vector or cut off a tightly clustered cosine island. Files they wrote load with the topology they were saved with, stranded vectors included;
`compact()` rebuilds the graph and reconnects them.

Reachable is not the same as found. A bounded approximate search can still
miss a stored vector its beam never gets near, so it is not a completeness
guarantee. In particular, a vector no neighbour kept a link to is chained to
the vector stored just before it; it is on a path, but only a wide beam or
the exhaustive scan below will return it.

How often that happens depends on the data and on `M`. Measured on 20,000
64-dimension vectors: Gaussian data under cosine chains about 4% of nodes at
`M = 4` and almost none at `M = 8`; under L2 it chains 39% at `M = 4`, 9% at
`M = 8` and 2% at `M = 12`. Under DOT with norms spread over two orders of
magnitude the few high-norm vectors fill up and most nodes are chained, at
every `M` including 16.

Keeping every vector connected costs narrow-beam recall where 0.2.0-rc.2
stranded vectors instead. Against rc.2, recall@10 on that synthetic data is
lower by up to 3 to 4.5 points for Gaussian L2 at `M = 4`, 1 to 2 points at
`M = 8`, about 1 point at `M = 12`; by 1 to 1.5 points for Gaussian cosine and
DOT at `M = 4`; and by about 2 points for uneven-norm DOT at `M = 8`. On the
768-dimension embedding fixture, recall at `M = 4`, 8 and 12 is within 0.2
points of rc.2 or above it, except L2 at `M = 4` with a beam of 10, which is
1.7 points lower. At `M = 16` the differences are within 0.6 points on all of
it.

0.2.0-rc.3 allowed one tree child per node when `M < 16`. That made the tree
the insertion order for every metric: each node spent two links on its
predecessor and successor, and narrow-beam recall fell against rc.2 by about
one point at `M = 8` and five to eight at `M = 4` on the embedding fixture,
and by 1 to 2.5 and 2 to 8.5 points on Gaussian data. From rc.4 the limit is
at least two. That recovers the embedding fixture and Gaussian cosine, and
Gaussian L2 partly. It does not help uneven-norm DOT, which did best with the
chain: there rc.4 is below rc.3 by about 1 point at `M = 8` and 3 to 5 points
at `M = 4`. Graphs built with `M >= 16` are byte-identical between rc.3 and
rc.4. An rc.3 graph built with `M < 16` keeps its topology on load;
`compact()` rebuilds it.

Inserting into a graph loaded from an earlier release protects only the nodes
that still have a mutual link to an earlier node. The first insert after any
load derives the tree over the whole graph under the write lock: about 0.03 s
at 300,000 × 128 on an M4 Pro, growing linearly. A read-only index never pays
it.

The derived tree equals the writer's when the same distance kernel reads the
file. A different kernel (another SIMD path, or the other engine) can order a
near-tie differently and derive a different, equally valid tree; an index
resumed that way stays connected but is no longer byte-identical to one that
was never saved.

When the effective beam reaches the **stored slot count**, search performs an
exact scan with bounded top-k selection. This also works for disconnected
legacy/C++ files and excludes tombstones and rejected filter IDs. Request it
with `ef_search >= len() + tombstones()` (JavaScript: `efSearch`), or request
that many results. Filtered widening can reach it through `max_ef_search`;
a cap alone does not force exhaustive search if an earlier pass already fills
`k`. The scan costs O(n × dimension + n log k), so use it deliberately.

Loading, searching and saving still preserve the original graph topology.
This query fallback does not repair connectivity or claim improved recall
for smaller beams. `compact()` reclaims tombstones, not a guaranteed graph
connectivity repair. Use exact indexes when every query needs completeness.

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
- Filter predicates run inside that read lock, so a Rust predicate that
  calls into the same index, or waits for another thread to modify it, can
  deadlock. The Python and WebAssembly bindings detect a same-index call
  from a predicate and raise (`RuntimeError`; an `Error` with
  `code === "ERR_REENTRANT_SEARCH"`) instead; waiting on another thread is
  still a deadlock there. `DiskIndex` is immutable and has no lock.
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
