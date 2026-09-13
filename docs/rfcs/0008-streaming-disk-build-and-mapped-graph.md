# RFC 0008: Streaming disk build and mapped graph

- Status: draft
- Milestone: 0.4.0
- Tracking issue: #203
- Supersedes / superseded by: none

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
vectors are paged from the file and graph links are resident.

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
- Requires the VNDB v2 vector region to be contiguous and 4-byte aligned in
  the file. The v2 specification in `conformance/graph/README.md` is checked
  for this before implementation; if the current layout interleaves vectors
  with links, a new kind (`4`, HNSW with a separated vector region) is
  specified with golden fixtures, and `open_mapped` accepts only that kind
  while `load` accepts both. Existing identifiers are not reinterpreted.
- Links, levels, tombstone flags and the id map are read into memory at open;
  vectors are read through the mapping during search. Distance kernels take
  slices from the mapping directly.
- Search semantics identical to the resident index for the same file; a test
  asserts identical results on the conformance fixtures.

### Memory accounting

`ApproxIndex::resident_bytes()` (both modes) reports what the process holds,
so a mobile caller can budget. Documented as an estimate.

## Alternatives rejected

- **DiskANN-style on-disk graph with vectors and links both paged.** Deferred:
  a different index structure and file format; the mapped HNSW covers the
  100k to few-million range that the audience has, with much less new code.
- **Vectors in a sidecar file for the mapped graph.** Rejected: one file per
  index is a property users rely on and the conformance suite checks.

## Compatibility and migration

- `DiskIndex` file bytes unchanged; both builders produce the same file.
- Possible new VNDB v2 kind; old kind unchanged and readable; C++ engine
  rejects the new kind cleanly (tested). Both `HnswData` mirrors updated in
  lockstep if the graph layout changes.
- `VaneError::ReadOnly` is a new variant; every binding maps it: Python a
  new `ReadOnlyError` under the package's base exception (not
  `PermissionError`, which would conflate it with OS permissions), C ABI
  `VANEDB_RS_READ_ONLY`, WebAssembly an error with the same name.

## Acceptance criteria

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

Writable mapped graphs, DiskANN, quantized mapped vectors (compose with RFC
0005 later).
