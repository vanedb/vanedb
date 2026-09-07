# VNDB graph format

For the 1.0.0 release, VNDB v1 disk files and VNDB v2 graph files are the
persistence contract. The Rust engine in VaneDB 1.x will continue to read valid
files written by VaneDB 1.0.0, subject to documented platform and resource limits.
Existing version,
kind and continuation identifiers must never be reinterpreted; incompatible
encodings require new identifiers while retaining readers for existing files.
This does not promise that older readers accept future formats.

The Rust engine reproduces these fixtures, in both directions: it reads each
one and re-writing what it read reproduces the bytes
(`vanedb/tests/vndb_graph_format.rs`). The C++ engine does **not** read this
format yet — it reads its own legacy graph files — so the cross-engine half of
this contract is unimplemented. `VNDB` v1 disk files are cross-engine today;
graph files are not.
VNDB v1 remains the disk format unchanged. VNDB v2 identifies graph files, with
kind 1 identifying HNSW. Readers reject other versions and kinds.

Every numeric field is explicitly sized and little-endian. The header is 96 bytes:

| Offset | Type | Meaning |
|---|---|---|
| 0 | 4 bytes | Literal `VNDB` |
| 4 | u32 | Version: 2 |
| 8 | u32 | Kind: 1 (HNSW graph) |
| 12 | u32 | Metric: 0 squared L2, 1 cosine, 2 negative dot |
| 16 | u64 | Dimension |
| 24 | u64 | Stored slot count, including tombstones |
| 32 | u64 | Capacity hint, at least the stored count |
| 40 | u64 | M, at least 2; layer 0 degree cap is 2M |
| 48 | u64 | Construction beam width, nonzero |
| 56 | u64 | Default search beam width |
| 64 | u64 | Original construction seed |
| 72 | u64 | Entry slot; UINT64_MAX when stored count is zero |
| 80 | i32 | Maximum level; -1 when stored count is zero |
| 84 | u32 | Continuation encoding (below) |
| 88 | u64 | Continuation section byte length, at most 65536 |

Nodes follow in slot order. Each contains an external ID (`u64`), level (`u32`,
0–32), flags (`u32`: 0 live, 1 deleted), and exactly `dimension` finite `f32`
components. Then each layer from 0 through the node's level contains a degree
(`u64`) and that many neighbor slot numbers (`u64`). The continuation section
follows the last node. No padding or trailing bytes are allowed.

Live IDs are unique. Deleted slots may retain an ID now owned by a live slot.
Deleted nodes retain their vectors and links and may be traversed, but never
appear in search results or live-ID lookups. Zero stored slots means no entry
and maximum level -1. A graph whose slots are all deleted still retains its
entry and topology. Otherwise the entry has the observed maximum level.
Edges cannot be self-links or duplicates, must name existing slots, and cannot
point above a neighbor's level. Upper-layer degrees are at most M.

Readers check sizes against the available bytes and platform limits before
allocation, then validate the complete graph before exposing an index. Dimension
and capacity must be nonzero; capacity is capped at 100 million slots. Sizes
must fit the reader's address space and available memory. Readers allocate
storage for saved slots. The Rust engine treats capacity as a hint; a C++
reader, if one is written, would treat it as a hard insertion limit, and Rust
can grow beyond the original hint up to the same 100-million stored-slot limit.
Rust rejects additions and whole batches that exceed this limit before changing
the graph. An upsert consumes one new slot, even when replacing an existing ID;
a rejected upsert preserves that entry. Compact tombstones to reclaim room.
Writers preserve slot order, adjacency order, IDs, tombstones, and vector bits. They do
not rebuild or compact the graph as a side effect of saving.

## Continuation encodings

The graph and search contract is portable. Subsequent mutations may produce
different topology when moving between engines or RNG implementations, just as
independently building the same input does today.

- 1: Rust `rand` 0.10 `StdRng` seeded from the header, advanced by one level draw
  per stored slot. The section is empty.
The Rust engine reads, validates and preserves all three encodings — the
fixtures cover 1, 2 and 3, and each round-trips byte for byte. Encodings 2 and
3 exist so a C++ writer's RNG state survives a round trip through the Rust
engine unchanged; no such writer exists in this repository yet, so nothing
here emits them outside the fixtures.

- 2: C++ libstdc++ MT19937 stream: 624 decimal u32 state words and a position
  in 0–624, separated by ASCII whitespace.
- 3: C++ libc++/MSVC MT19937 stream: 624 decimal u32 state words separated by
  ASCII whitespace.

Rust `StdRng` does not promise the same output across dependency releases or
platforms. Encoding 1 preserves the graph and supports seed-based continuation;
it is not a promise that future dependency versions reproduce subsequent
insertions byte for byte. Review any RNG dependency or level-generation change
against the fixed fixtures and save/load continuation tests. If a future version
requires a distinct, reproducible generator contract, assign a new continuation
encoding and retain the reader for existing files; do not silently redefine it.

Words contain decimal digits only. A C++ writer would seed its fallback
MT19937 from the low 32 bits of the header seed, while preserving the full seed in the file.

A reader restores its native continuation when supported. Otherwise it seeds
its own level generator from the stored seed and advances it for the stored
slots. It preserves the original continuation bytes until an insertion or
rebuild replaces the generator state. Removal and search-setting changes do
not consume randomness. A load/save without mutation therefore preserves the
entire file even when the reader uses another generator internally.

## Fixtures and migration

`generate.py` writes independent fixtures into
`vanedb/tests/fixtures/vndb_graph/`, which is included with the Rust crate.
The fixtures cover all metrics and continuation encodings, an empty graph,
a tombstone with ID reuse, a deleted entry point, and an all-deleted graph. `SHA256SUMS` pins the contract bytes. Both
engine-written output and cross-engine roundtrips must match them.

Legacy readers remain available. Load an old file and save it to a new path to
migrate, retaining the original until the migrated file has been verified.
Old readers cannot open VNDB v2. Keep source vectors when moving between
pre-release builds; do not assume an unverified graph is a system of record.
