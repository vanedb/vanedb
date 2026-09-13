# RFC 0009: Optional payload column

- Status: accepted (2026-09-13)
- Milestone: 0.4.0
- Tracking issue: #204
- Supersedes / superseded by: none; stores its data in the RFC 0013 container

## Problem

The README's second paragraph says: "It holds only `(u64 id, vector)` pairs,
no metadata or payload storage and no filtered search, so keep your own
id-to-document mapping alongside it." That sentence loses the Python audience
to Chroma and LanceDB before the quick start, and every persona in issue #91
(note-taking plugin, RAG corpus, mobile app) has to build the mapping. RFC 0004
gives them filtering by id; this RFC gives them somewhere to keep the thing the
id points at.

## Decision

Add an optional opaque byte payload per id, stored in the `payloads` section
of the VNDB v3 container (RFC 0013), returned with search results on request.
The core does not interpret the bytes. Python and JavaScript accept and return
strings and JSON-serialisable objects as a convenience, encoded as UTF-8 JSON
by the binding. Filtering on payload contents is not part of this RFC.

Decision 12 of the 2026-09-13 review: one file per index, opaque in the core,
convenient in the bindings.

## Design

### API

```rust
index.add_with_payload(id, &vector, payload: &[u8])?;
index.add_batch_with_payloads(ids, vectors, payloads: &[&[u8]])?;
index.payload(id) -> Result<Option<Cow<'_, [u8]>>>;   // None: no payload or unknown id (RFC 0011)
index.set_payload(id, &[u8])?;                         // replace; error if id absent
index.search_with(&q, k, &SearchParams::new().with_payloads(true))
    -> Vec<SearchResult>  // SearchResult gains `payload: Option<Vec<u8>>`
```

`SearchResult` is `#[non_exhaustive]` for exactly this. `PartialEq`/`Ord`
stay defined over `(id, distance)`. Payload size is bounded (default 64 KiB,
configurable at build, stored in the section's cap field); larger is
`VaneError::Validation`.

`upsert` replaces the payload when one is given and clears it when none is,
so an upsert is a whole-row replacement; `set_payload` is the way to change
only the payload.

### Binding conveniences

| Binding | Accepts | Returns | Encoding |
|---|---|---|---|
| Rust | `&[u8]` | `Cow<[u8]>` | none |
| C ABI | pointer + length | caller buffer + length query | none |
| Python | `bytes`, `str`, or any `json.dumps`-able object | `bytes` for bytes input; for `str`/object input the binding stores a one-byte tag then UTF-8 JSON, and returns the decoded `str`/object | tag byte `0x00` raw, `0x01` JSON |
| WebAssembly | `Uint8Array`, `string`, or a JSON-serialisable value | same rule as Python | same tag |

The tag byte is a binding convention documented in `conformance/`, so a
payload written from Python reads back as the same type in JavaScript, and
Rust and C see the raw tagged bytes. Raw `bytes` input is stored untagged
only when the caller asks (`raw=True`); the default is tagged so the type
round-trips.

Python: `add_batch(ids, vectors, payloads=None)`, `payload(id)`,
`search(..., with_payloads=True)` returning `(id, distance, payload)` triples
only when asked, so existing callers see no change. WebAssembly:
`search(query, k, { withPayloads: true })` returns a `payloads` array beside
`ids` and `distances`. C ABI: new functions with caller-provided buffers and a
length query, existing functions unchanged.

### Storage

- The RFC 0013 `payloads` section: cap, per-slot `(offset, length)` table,
  then the bytes. Fixed-width, exact length, no overlapping ranges, every
  table entry inside the bytes region; all checked at load with the
  corruption-fixture discipline.
- In memory: a `Vec<u8>` arena plus per-slot `(offset, len)`, reclaimed by
  `compact()` like tombstoned vectors. `tombstones()` semantics unchanged.
- An index with no payloads writes no `payloads` section; a v3 file without
  one loads with every payload `None`.

## Alternatives rejected

- **A sidecar file.** Rejected on 2026-09-13: two files per index, a
  missing-sidecar failure mode, and a paragraph of documentation about it.
- **Typed metadata (key-value schema) with a query language.** Rejected: that
  is a database, and the market has several. Opaque bytes with a JSON
  convenience serve every persona and keep the engine honest about what it is.
- **Untagged JSON by default.** Rejected: a payload written as `str` from
  Python would read back as bytes in JavaScript; the tag makes the round trip
  typed.
- **Payloads in `DiskIndex`.** Deferred: the mapped `payloads` section
  composes with RFC 0008 later without a format change.

## Compatibility and migration

- Additive on every surface; results without `with_payloads` are unchanged.
- No new format identifier: the section is part of RFC 0013.
- The frozen C++ engine reads v1 and v2 only; a v3 file with payloads is
  rejected by it like any v3 file (RFC 0013).

## Acceptance criteria

- [ ] `payloads` section golden and corruption fixtures in `conformance/v3/`;
      the loader rejects every malformed case.
- [ ] Rust API above with tests; `compact()` reclaims payload space; `upsert`
      semantics as specified.
- [ ] Python, WebAssembly and C ABI surfaces with tests; the tag convention
      round-trips a `str`, a `dict` and raw `bytes` between Python and
      JavaScript; the ctypes example stores and reads one payload.
- [ ] Interleaved bench: search without `with_payloads` unchanged within the
      noise floor.
- [ ] README's "keep your own id-to-document mapping" paragraph rewritten;
      `GETTING_STARTED.md` stores the sentence as the payload; `CHANGELOG.md`.

## Evidence required before the claim

Corruption fixtures and the binding round-trip tests; no performance claim is
made.

## Out of scope

Filtering on payload contents, typed metadata, full-text search, payloads for
`DiskIndex`.
