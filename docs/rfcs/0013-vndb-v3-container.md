# RFC 0013: VNDB v3 container format

- Status: accepted (2026-09-13)
- Milestone: 0.3.0, ahead of RFC 0005
- Tracking issue: #209
- Supersedes / superseded by: none; RFCs 0005, 0008 and 0009 target it

## Problem

Three accepted or pending RFCs each need to add something to the on-disk
format: quantized vector encodings (0005), a layout that a memory mapping can
search in place (0008), and a per-id payload (0009). Written independently
they produce three ad-hoc additions: two new graph kinds and a new disk
version, a fourth graph kind for a separated vector region, and a sidecar
file. Each needs its own fixtures, its own loader branch, and its own sentence
in the README. The sidecar in particular breaks the property users rely on:
one index is one file, and `save`/`load` take one name.

VNDB v1 (flat) and v2 (graph) are fixed-layout formats with no room for
optional content. Decision 12 of the 2026-09-13 review asked for a coherent
product; one container format designed once is the coherent answer.

## Decision

Specify VNDB version 3 as a sectioned container: one header, one section
table, and typed sections for ids, vectors, graph links, tombstones, payloads
and parameters. Every index type writes v3 from 0.3.0. Every existing reader
for v1 and v2 is retained forever. Section identifiers are never reused;
unknown optional sections are skipped, unknown required sections reject the
file. The frozen C++ engine does not learn v3; cross-engine conformance for
v3 is "C++ rejects it cleanly, tested", while v1 and v2 cross-load stays
green.

## Design

### Header (64 bytes, little-endian)

| Offset | Type | Meaning |
|---|---|---|
| 0 | 4 bytes | Literal `VNDB` |
| 4 | u32 | Version: 3 |
| 8 | u32 | Index kind: 1 flat (exact scan), 2 HNSW graph |
| 12 | u32 | Metric: 0 squared L2, 1 cosine, 2 negative dot |
| 16 | u64 | Dimension, nonzero |
| 24 | u64 | Stored slot count, including tombstones, at most 100 million |
| 32 | u64 | Section count, at most 64 |
| 40 | u64 | Total file length; must equal the actual length |
| 48 | u64 | Header checksum: CRC-32C of bytes 0..48, zero-extended |
| 56 | u64 | Reserved, zero |

### Section table

Immediately after the header: `section count` entries of 32 bytes each.

| Offset | Type | Meaning |
|---|---|---|
| 0 | u32 | Section identifier (below) |
| 4 | u32 | Flags: bit 0 `required`; other bits zero |
| 8 | u64 | Byte offset from file start; a multiple of 64 |
| 16 | u64 | Byte length |
| 24 | u64 | CRC-32C of the section bytes, zero-extended |

Rules checked before any allocation: identifiers unique; offsets ascending
and non-overlapping; every section inside the declared file length; the last
section ends at the declared length minus padding to 64 bytes; each section's
length matches what its identifier implies for the header's dimension and
count. All of the corruption-fixture discipline of v1 and v2 applies.

### Section identifiers

| Id | Name | Required | Contents |
|---|---|---|---|
| 1 | `ids` | yes | `count` × u64 external ids, slot order |
| 2 | `vectors` | yes | 16-byte encoding prefix (u32 encoding, u32 flags, u64 reserved) then the vector payload: encoding 0 f32 (`count × dim × 4`), 1 int8 (`count × dim`), 2 binary (`count × ceil(dim/8)`) |
| 3 | `quant_params` | if encoding ≠ 0 | int8: f32 offset, f32 scale, then `count` × f32 norms when metric is cosine; binary: `dim` × f32 mean, u32 padding bits |
| 4 | `tombstones` | no | `ceil(count/8)` bytes bitmap; absent means none |
| 5 | `graph` | kind 2 | u64 M, u64 construction beam, u64 default search beam, u64 seed, u64 entry slot, i32 max level, u32 continuation encoding, u64 continuation length, continuation bytes; then per slot: u32 level, and per layer 0..level a u32 degree and that many u32 slot numbers |
| 6 | `payloads` | no | u64 cap, then `count` × (u64 offset, u32 length, u32 reserved) then the bytes; offsets relative to the bytes region; `length` 0 with offset 0 means no payload |
| 7 | `rescore_f32` | no | `count × dim × 4` f32 vectors kept beside a quantized encoding for rescoring |

Identifiers 8 and above are unassigned. Slot numbers are u32 because the
stored-slot limit is 100 million (RFC 0008 asked for this: it halves link
memory versus v2's u64).

### Alignment and mapping

Every section starts on a 64-byte boundary and the `vectors` payload starts
16 bytes into its section, so f32 rows are 4-byte aligned and int8/binary rows
byte-aligned, which is what `open_mapped` (RFC 0008) needs to hand slices from
the mapping to the distance kernels without copying.

### Writers

`FlatIndex`, `ApproxIndex` and `DiskIndexBuilder` all write v3 from 0.3.0.
`DiskIndex` is a v3 file with kind 1 and no `graph` section. Writers emit
sections in identifier order, pad with zeros, and compute checksums; the
existing atomic-write path is unchanged. `save` never rebuilds or compacts as
a side effect, as today.

### Readers

`load` / `open` accept versions 1, 2 and 3 and dispatch on the version field.
v3 loaders: verify the header checksum, then the table, then each required
section's checksum before use; unknown identifiers with `required` set reject
the file with `VaneError::Corrupt` naming the identifier; unknown identifiers
without it are skipped and preserved verbatim on the next save (so a newer
writer's optional section survives a round trip through an older reader).

### What the checksum is and is not

CRC-32C detects accidental corruption, which v1 and v2 cannot; it is not an
integrity guarantee against an adversary. The documentation says so, in the
sentence that today says the format carries no checksum.

## Alternatives rejected

- **Three independent format additions (kinds 2, 3, 4; disk version 3; a
  sidecar).** Rejected: three loader branches, three fixture sets, two files
  per index.
- **A general-purpose container (zip, tar, Parquet, Arrow IPC).** Rejected: a
  dependency in the loader, no mmap alignment guarantees, and the hardened
  fixed-width discipline would have to be re-imposed on top.
- **Extend v2 with trailing sections.** Rejected: v2's "no trailing bytes"
  rule is a loader rule the corruption suite enforces; relaxing it for old
  readers is exactly the reinterpretation the invariants forbid.

## Compatibility and migration

- v1 and v2 files load unchanged forever. Saving re-emits v3. The README
  migration paragraph changes from "load in the original engine" to "load and
  save".
- The frozen C++ engine reads v1 and v2 only. `bench/tests/cross_engine_format.rs`
  and `cross_engine_graph.rs` keep their v1/v2 assertions and gain a "C++
  rejects v3 with a clean error" assertion. This is the same concession RFC
  0005 already made for quantized kinds; it is now made once.
- Both `HnswData` mirrors are unaffected (they mirror the legacy bincode
  layout, not VNDB).
- The `AGENTS.md` persistence invariant is preserved: no identifier is
  reinterpreted, old readers are retained, corruption checks are kept and
  extended.

## Acceptance criteria

- [ ] `conformance/v3/README.md` specifying every table above, with golden
      fixtures for: flat f32; graph f32; graph with tombstones; graph with
      payloads; int8 and binary once RFC 0005 lands; an unknown optional
      section that must survive a round trip; an unknown required section
      that must be rejected.
- [ ] Corruption fixtures: bad header checksum, bad section checksum,
      overlapping sections, misaligned offset, declared length mismatch,
      duplicate identifier, section length inconsistent with dimension and
      count, slot number out of range.
- [ ] Rust writer and reader for kinds 1 and 2; `load` dispatches v1/v2/v3;
      every existing v1 and v2 test still passes.
- [ ] C++ rejection test in the cross-engine suites.
- [ ] Round trip: v2 fixture loaded, saved as v3, reloaded, identical search
      results and identical graph adjacency.
- [ ] Interleaved bench: save and load times for v3 versus v2 recorded; search
      unchanged within noise.
- [ ] `README.md` persistence section and `CHANGELOG.md` updated; RFCs 0005,
      0008 and 0009 reference the section identifiers here rather than their
      own kinds.

## Evidence required before the claim

The fixture and corruption suites on every CI platform; no hardware claim.

## Out of scope

Encryption (would be a section-level flag and a separate RFC), compression,
multi-index files, streaming writes of the graph section.
