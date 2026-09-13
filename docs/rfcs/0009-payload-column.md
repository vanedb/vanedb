# RFC 0009: Optional payload column

- Status: draft
- Milestone: 0.4.0
- Tracking issue: #204
- Supersedes / superseded by: none

## Problem

The README's second paragraph says: "It holds only `(u64 id, vector)` pairs,
no metadata or payload storage and no filtered search, so keep your own
id-to-document mapping alongside it." That sentence loses the Python audience
to Chroma and LanceDB before the quick start, and every persona in issue #91
(note-taking plugin, RAG corpus, mobile app) has to build the mapping. RFC 0004
gives them filtering by id; this RFC gives them somewhere to keep the thing the
id points at.

## Decision

Add an optional opaque byte payload per id, stored beside the index in a
sidecar file with its own hardened format, returned with search results on
request. VaneDB does not interpret the bytes. Filtering on payload contents is
not part of this RFC.

## Design

### API

```rust
index.add_with_payload(id, &vector, payload: &[u8])?;
index.payload(id) -> Result<Option<Cow<'_, [u8]>>>;
index.set_payload(id, &[u8])?;            // replace; error if id absent
index.search_with(&q, k, &SearchParams::new().with_payloads(true))
    -> Vec<SearchResult>  // SearchResult gains `payload: Option<Vec<u8>>`
```

`SearchResult` is `#[non_exhaustive]` for exactly this. `PartialEq`/`Ord`
stay defined over `(id, distance)`. Payload size is bounded (default 64 KiB,
configurable at build); larger is `VaneError::Validation`.

Python: `add_batch(ids, vectors, payloads=None)`, `payload(id) -> bytes | None`,
`search(..., with_payloads=True)` returning `(id, distance, payload)` triples
only when asked, so existing callers see no change. WebAssembly: `Uint8Array`
payloads. C ABI: new functions with caller-provided buffers and a length
query, existing functions unchanged.

### Storage

- Sidecar file `<index path>.payload`, format `VNDP` v1: magic, version, count,
  then a table of `(u64 id, u64 offset, u32 length)` followed by the bytes.
  Fixed-width little-endian, exact length equality, no overlapping ranges, no
  duplicate ids, every id present in the main file; all checked at load with
  the same corruption-fixture discipline as VNDB.
- The main VNDB file is unchanged, so the frozen C++ engine, the conformance
  fixtures and the cross-engine tests are untouched. A C++ load simply has no
  payloads.
- `save` writes both files through the atomic path; a missing sidecar on load
  means "no payloads", a present but corrupt one is an error.
- In memory: a `Vec<u8>` arena plus per-slot `(offset, len)`, freed by
  `compact()` like tombstoned vectors. `tombstones()` semantics unchanged.

## Alternatives rejected

- **A new section inside the VNDB file.** Rejected: it would need a new kind,
  and the C++ engine could no longer open the file. The sidecar keeps
  cross-engine parity for free.
- **Typed metadata (JSON, key-value) with a query language.** Rejected: that
  is a database, and the market has several. Opaque bytes serve every persona
  and stay honest about what the engine is.
- **Payloads in `DiskIndex`.** Deferred: a mapped sidecar composes naturally
  with RFC 0008 later.

## Compatibility and migration

- Additive on every surface; results without `with_payloads` are unchanged.
- No VNDB change. Both `HnswData` mirrors unaffected.
- The sidecar's name and format are documented in `conformance/` with golden
  fixtures.

## Acceptance criteria

- [ ] `VNDP` v1 specified in `conformance/payload/` with golden and corruption
      fixtures; loader rejects every malformed case in the fixture set.
- [ ] Rust API above with tests; `compact()` reclaims payload space; `upsert`
      preserves or replaces payload as documented.
- [ ] Python, WebAssembly and C ABI surfaces with tests; the ctypes example
      stores and reads one payload.
- [ ] A file saved with payloads loads in the C++ engine (vectors only) and the
      cross-engine graph test passes unchanged.
- [ ] Interleaved bench: search without `with_payloads` unchanged within the
      noise floor.
- [ ] README's "keep your own id-to-document mapping" paragraph rewritten;
      `GETTING_STARTED.md` stores the sentence as the payload; `CHANGELOG.md`.

## Evidence required before the claim

Corruption fixtures and the cross-engine run; no performance claim is made.

## Out of scope

Filtering on payload contents, typed metadata, full-text search, payloads for
`DiskIndex`.
