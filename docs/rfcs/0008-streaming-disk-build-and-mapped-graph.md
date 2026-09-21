# RFC 0008: Streaming disk build and mapped graph

- Status: accepted (for the streaming builder); capacity-study amendment
  2026-09-20; mapped graph draft, gated on RFC 0005 and the cold-cache
  spike. Direction accepted 2026-09-13.
- Milestone: 0.4.0
- Tracking issue: #203; capacity study #210
- Supersedes / superseded by: none; relies on the RFC 0013 container layout

## Problem

`DiskIndex` exists so that "a corpus larger than RAM stays searchable", and its
builder buffers every vector in memory until `save`. The index's stated purpose
is contradicted by its own construction path. Approximate search has no disk
option at all: `ApproxIndex` is entirely resident. Turso's DiskANN, Meilisearch's
`arroy` and LanceDB's IVF-PQ all answer the same demand: search over more
vectors than fit in memory, on one machine.

## Decision

Two changes, decided separately since the amendment of 2026-09-20 (below).
First, `DiskIndexBuilder` streams rows to the destination file as they arrive
and finalises the header at `save`, holding only the id set in memory. This
part is accepted with no gate. Second, `ApproxIndex` gains a read-only
memory-mapped open in which f32 vectors are paged from the file, graph links
are resident as flat `u32` arrays, and, when the file carries a quantized
encoding (RFC 0005), the quantized vectors are resident too and drive the
graph walk, with the mapped f32 copy used only to rescore the final
candidates. This is DiskANN's navigate-on-compressed, rescore-on-exact trick
applied to the existing HNSW graph, without a second index structure. This
part stays gated on RFC 0005 shipping and on the spike.

Before implementation of the mapped graph, a spike measures cold-cache search
on one Android device and one NVMe laptop at 1M × 768-d, f32 and binary
navigation; if p99 exceeds 20 ms the fallback is IVF over the mapped
`DiskIndex` (below), and this RFC is amended. The capacity study confirms
20 ms p99 as the right bar (study §2).

## Amendment: capacity study (2026-09-20)

[`docs/research/capacity.md`](../research/capacity.md) (#210, landed in #254
on 2026-09-20) answered the questions this RFC was gated on. Its decision
(study §7):

1. **RFC 0005 first, then RFC 0008.** Every segment's typical corpus (1k to
   100k vectors) fits the resident index today except under the 256 MB
   browser and low-RAM mobile budgets at 768-d, which int8 fixes; the mapped
   graph earns its place only at the segments' upper bounds (desktop RAG and
   embedded capture at 1M to 10M, mobile archives at 500k to 1M on 1 GB).
   The roadmap order (0005 at 0.3.0, 0008 at 0.4.0) is confirmed by the
   numbers.
2. **Split this RFC.** The streaming `DiskIndexBuilder` has no gate and is
   accepted as of this amendment; it fixes the build-time contradiction in
   the shipped product, touches no graph, and is adjacent to RFC 0010's
   write-path work (study §6 observation 3 and §7). The mapped graph stays gated on
   (a) RFC 0005 shipped, so the spike can measure binary navigation rather
   than only f32, and (b) the cold-cache spike above. The IVF fallback remains
   the fallback; nothing in the corpus data asks for it ahead of the spike.
3. **Thresholds at d = 768, binary budgets** (study §7, table in §6).
   Resident f32 wins below ~314k vectors on 1 GiB; int8 wins from there to
   ~958k on 1 GiB or ~1.9M on 2 GiB; above ~1M on 1 GiB or ~2M on 2 GiB only
   the mapped designs hold the corpus. Mapped f32 alone holds 12.7M at 768-d
   in 2 GiB and binary navigation with mapped rescoring holds 8.1M; the
   latter is the design worth shipping because it bounds cold-cache page
   faults by the rescoring set rather than by the beam (study §6,
   observation 2). Resident f32 thresholds scale with `1/d`; mapped f32 is
   dimension-independent at ~169 B per vector; binary with mapped rescoring
   moves only with `d/8`.
4. **No segment justifies DiskANN.** The largest upper bound is 10M on a
   workstation or gateway; the revisit condition under "Alternatives
   rejected" is not met.

Figures the study fixed or left open (each is also updated in place below):

- The id map is `HashMap<u64, usize>` in every index, including the
  `DiskIndex` this builder produces, and costs **19 to 39 bytes per entry**
  at open depending on where `n` falls between powers of two (22.3 B at
  100k, 35.7 B at 1M, 28.5 B at 10M): 190 to 390 MB resident at 10M on an
  index whose vectors are otherwise paged. Reducing it (sorted `u64` array
  or `u32` slot table) is adjacent to RFC 0010's hasher change and is
  decided there, not here (study §4.1, §8.3). The builder's own
  duplicate-check set (`HashSet<u64>`) is not costed by the study.
- Links cost **~297 B per node today** in the `Vec<Vec<Vec<usize>>>` layout
  (measured on random vectors, mean layer-0 degree 23.7 of the 32 cap) and
  would be **~116 B per node as flat `u32` arrays**; at a full layer 0 the
  figures are ~300 to 360 B and ~120 to 150 B. This RFC's "about 140 bytes
  per node" for the mapped layout was within the study's bracket; its "half
  of today's `usize`" (~280 B) undershot the measured ~297 B and the ~300 to
  360 B full-degree planning figure (study §4.4).
- Measurements the study could not make (#210 question 3): resident memory,
  p50/p99 warm and cold, and recall@10 for resident f32, mapped f32 and
  binary-plus-rescoring at 100k, 1M and 10M on one Android device and one
  NVMe laptop. They need the RFC 0003 fixture and dedicated hardware and
  belong to this RFC's spike (study §8.1); the spike criterion below records
  them.

## Design

### Streaming `DiskIndexBuilder`

- `DiskIndexBuilder::create(path, dim, metric)` opens a temporary file beside
  `path`, writes a placeholder 32-byte VNDB v1 header, and appends ids and
  vectors as `add` / `add_batch` arrive through a `BufWriter`.
- The v1 layout is ids first, then vectors. Streaming therefore writes two
  temporary files (ids, vectors) and concatenates them at `save`, or writes
  vectors to the temporary and ids to memory (8 bytes per row in the id
  list, 8 MB per million rows, plus the existing `HashSet<u64>` duplicate
  check) and emits ids then vectors at `save`. The second is chosen: ids are
  needed in memory anyway for the duplicate check. Build peak falls from
  `n × (4d + 8)` to the id term alone (study §6); the opened `DiskIndex`
  then holds its id map at 19 to 39 bytes per entry (study §4.1).
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
  arrays at open (`u32` slots: ~116 B per node of links at M = 16 at the
  degree random vectors reach, ~120 to 150 B at a full layer 0, against
  ~297 B today; plus 13 B of ids, level and tombstone and 19 to 39 B of id
  map per node; ~169 B per vector in all (at the study's ~120 B links and
  36 B id map at 1M), at every `d`; study §4.4, §6).
  f32 vectors are read through the mapping during search; the distance
  kernels take slices from the mapping directly.
- With a quantized `vectors` section (RFC 0005), the int8 or binary vectors
  are read into memory (770 B or 96 B per 768-d vector) and the walk runs on
  them; `SearchParams::rescore(n)` re-ranks the top `n × k` candidates from
  the mapped `rescore_f32` section. Resident memory for 1M × 768-d is then
  about 265 MB with binary navigation or 940 MB with int8 (links, ids, id
  map and the quantized copy; page cache excluded); for 10M, 2.7 GB or
  9.4 GB (study §6).
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

The choice between the designs above, and the numbers this RFC promises, rested
on facts gathered by [`docs/research/capacity.md`](../research/capacity.md)
(2026-09-20): the corpus sizes and memory budgets each target segment has
(study §2, §3), the capacity each competitor supports and at what resident
memory (study §5), and the computed resident cost of the mapped and quantized
designs (study §4, §6). `docs/LIMITS.md` records what 0.1.1 can hold today.
The study's decision and the corrections it made to this RFC are in the
amendment above; the on-device latency and recall it could not measure are
part of the spike.

## Alternatives rejected

- **DiskANN-style on-disk graph with vectors and links both paged.** Rejected
  for now: a different index structure and file format; the mapped HNSW with
  quantized navigation covers the 100k to 10M range the audience has, with
  much less new code. Revisit only if the capacity study finds a segment past
  10M vectors on one device. The study found none: the largest upper bound
  is 10M on a workstation or gateway (study §2, §7).
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
      The same run records resident memory, warm and cold p50/p99, and
      recall@10 for resident f32, mapped f32 and binary-plus-rescoring at
      100k, 1M and 10M (#210 question 3; study §8.1), and amends the study's
      table.
- [x] Capacity study (#210) published and this RFC amended with its findings
      (2026-09-20).
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

The streaming-builder criteria (third and fourth, and the README line) may be
met before the mapped-graph ones; the mapped-graph criteria wait on RFC 0005
and the spike.

## Evidence required before the claim

Peak RSS measurements from a dedicated machine, and a cold-cache search
latency figure so the docs can say what paging costs.

## Out of scope

Writable mapped graphs, DiskANN.
