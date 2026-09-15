# RFC 0008: Streaming disk build and mapped graph

- Status: draft; direction accepted 2026-09-13, acceptance gated on the
  capacity study (#210) and the cold-cache spike below
- Milestone: 0.4.0
- Tracking issue: #203
- Supersedes / superseded by: none; relies on the RFC 0013 container layout

## Problem

`DiskIndex` exists so that "a corpus larger than RAM stays searchable", and its
builder buffers every vector in memory until `save`. The index's stated purpose
is contradicted by its own construction path. Approximate search has no disk
option at all: `ApproxIndex` is entirely resident. Turso's DiskANN, Meilisearch's
`arroy` and LanceDB's IVF-PQ all answer the same demand: search over more
vectors than fit in memory, on one machine.

## Decision

Two changes. First, `DiskIndexBuilder` streams rows to the destination file as
they arrive and finalises the header at `save`, holding only the id set in
memory. Second, `ApproxIndex` gains a read-only memory-mapped open in which
f32 vectors are paged from the file, graph links are resident as flat `u32`
arrays, and, when the file carries a quantized encoding (RFC 0005), the
quantized vectors are resident too and drive the graph walk, with the mapped
f32 copy used only to rescore the final candidates. This is DiskANN's
navigate-on-compressed, rescore-on-exact trick applied to the existing HNSW
graph, without a second index structure.

Before implementation, a spike measures cold-cache search on one Android
device and one NVMe laptop at 1M × 768-d; if p99 exceeds 20 ms the fallback is
IVF over the mapped `DiskIndex` (below), and this RFC is amended.

## Design

### Streaming `DiskIndexBuilder`

- `DiskIndexBuilder::create(path, dim, metric)` opens a temporary file beside
  `path`, writes a placeholder 32-byte VNDB v1 header, and appends ids and
  vectors as `add` / `add_batch` arrive through a `BufWriter`.
- The v1 layout is ids first, then vectors. Streaming therefore writes two
  temporary files (ids, vectors) and concatenates them at `save`, or writes
  vectors to the temporary and ids to memory (8 bytes per row, 8 MB per
  million rows) and emits ids then vectors at `save`. The second is chosen:
  ids are needed in memory anyway for the duplicate check.
- `save` writes the final header with the count, copies the vector stream
  after the ids, fsyncs, and renames atomically through the existing
  `atomic_write` path. The file length equality rule is unchanged.
- The existing all-in-memory `DiskIndexBuilder::new` remains for small
  corpora; both produce identical bytes for identical input, checked by a
  test.
- Python `DiskIndexBuilder(path=...)` and the C ABI `_disk_builder_create`
  expose the streaming variant; wasm has no disk index.

### Mapped `ApproxIndex`

- `ApproxIndex::open_mapped(path) -> Result<ApproxIndex>` (feature `disk`),
  `unsafe` for the same reason `DiskIndex::open` is: the mapping cannot defend
  itself against a concurrent writer. Read-only: `add`, `remove`, `upsert`,
  `compact` return `VaneError::ReadOnly` (new variant; `VaneError` is
  `#[non_exhaustive]`).
- Operates on VNDB v3 files (RFC 0013): the `vectors` section is contiguous
  and aligned by construction, and the `graph` section stores `u32` slot
  numbers. `open_mapped` on a v1 or v2 file returns `VaneError::Validation`
  telling the caller to load and save once. v2 interleaves vectors with links,
  which is why v3 exists.
- Links, levels, tombstone flags and the id map are read into flat resident
  arrays at open (`u32` slots, about 140 bytes per node at M = 16, half of
  today's `usize`). f32 vectors are read through the mapping during search;
  the distance kernels take slices from the mapping directly.
- With a quantized `vectors` section (RFC 0005), the int8 or binary vectors
  are read into memory (770 B or 96 B per 768-d vector) and the walk runs on
  them; `SearchParams::rescore(n)` re-ranks the top `n × k` candidates from
  the mapped `rescore_f32` section. Resident memory for 1M × 768-d is then
  about 140 MB of links plus 96 MB (binary) or 770 MB (int8); for 10M, 1.4 GB
  plus 0.96 GB or 7.7 GB.
- Search semantics identical to the resident index for the same file; a test
  asserts identical results on the conformance fixtures.

### Memory accounting

`ApproxIndex::resident_bytes()` (both modes) reports what the process holds,
so a mobile caller can budget. Documented as an estimate.

### Fallback: IVF over the mapped `DiskIndex`

If the spike fails the 20 ms p99 gate, the alternative with sequential rather
than random disk access: k-means centroids (resident, a few MB) partition the
rows of a `DiskIndex` file into clusters stored contiguously; a search scans
the `nprobe` nearest clusters from the mapping. Simpler than a mapped graph,
friendlier to flash, lower recall at equal latency. It would be a mode of
`DiskIndex` ("exact by default, approximate with clusters"), not a fourth
index type, and a `clusters` section in the v3 container.

### Capacity study (#210)

The choice between the designs above, and the numbers this RFC promises, rest
on facts not yet gathered: the corpus sizes and memory budgets each target
segment actually has (mobile, browser, desktop RAG, gateway); the capacity
each competitor supports and at what resident memory (ObjectBox, USearch,
sqlite-vec, libSQL DiskANN, LanceDB, EdgeVec); and the measured latency,
recall and memory of the mapped and quantized designs on real embeddings.
`docs/LIMITS.md` records what 0.1.1 can hold today; the study produces
`docs/research/capacity.md` and amends this RFC before it is accepted.

## Alternatives rejected

- **DiskANN-style on-disk graph with vectors and links both paged.** Rejected
  for now: a different index structure and file format; the mapped HNSW with
  quantized navigation covers the 100k to 10M range the audience has, with
  much less new code. Revisit only if the capacity study finds a segment past
  10M vectors on one device.
- **Vectors in a sidecar file for the mapped graph.** Rejected: one file per
  index (RFC 0013).

## Compatibility and migration

- `DiskIndex` file bytes unchanged; both builders produce the same file.
- No new identifiers: the layout is RFC 0013's. v1 and v2 files remain
  loadable; `open_mapped` needs a v3 file. Both `HnswData` mirrors are
  unaffected.
- `VaneError::ReadOnly` is a new variant; every binding maps it: Python a
  new `ReadOnlyError` under the package's base exception (not
  `PermissionError`, which would conflate it with OS permissions), C ABI
  `VANEDB_RS_READ_ONLY`, WebAssembly an error with the same name.

## Acceptance criteria

- [ ] Spike recorded: cold-cache p50/p99 at 1M × 768-d on one Android device
      and one NVMe laptop, f32 and binary navigation; the 20 ms gate decided.
- [ ] Capacity study (#210) published and this RFC amended with its findings.
- [ ] Streaming builder produces byte-identical files to the in-memory builder
      on the conformance fixtures and on a 1M-row synthetic corpus, with peak
      resident memory recorded for both.
- [ ] Duplicate-id and non-finite checks preserved on the streaming path;
      failure leaves no temporary file behind.
- [ ] `open_mapped` returns identical results to `load` on the fixtures for all
      three metrics; corruption suite applied to the mapped path.
- [ ] Mutation on a mapped index returns `ReadOnly` in every binding.
- [ ] `resident_bytes()` in every binding.
- [ ] Bench row: `index_search` resident vs mapped, warm and cold cache, on a
      dedicated machine, interleaved.
- [ ] README: "Building one still buffers its vectors in memory" removed;
      `CHANGELOG.md` entry.

## Evidence required before the claim

Peak RSS measurements from a dedicated machine, and a cold-cache search
latency figure so the docs can say what paging costs.

## Out of scope

Writable mapped graphs, DiskANN.
