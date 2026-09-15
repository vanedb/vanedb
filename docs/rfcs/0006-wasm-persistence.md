# RFC 0006: WebAssembly persistence

- Status: accepted (2026-09-13)
- Milestone: 0.2.0
- Tracking issue: #201
- Supersedes / superseded by: none

## Problem

`@vanedb/wasm` exposes add, batch add, search, lookup, remove, upsert,
tombstones and compact on `ApproxIndex`, and nothing for persistence. A browser
application rebuilds its index on every page load, which is the one thing a
browser user cannot afford. Every browser competitor persists: EdgeVec and
`client-vector-search` to IndexedDB, voy through serialisation, `usearch-wasm`
through a manual byte view. Without this the package loses every browser
evaluation on the first criterion.

Every other binding says `save(path)` and `load(path)`. A browser has no paths.
The design has to keep one mental model across five bindings: save it under a
name, load it by that name.

The Rust core already writes VNDB v2 through `graph_format::write(impl Write, ..)`
and reads it in `ApproxIndex::load`; only the path-based entry points are
public.

## Decision

Three layers. The Rust core gains byte-based save and load. The wasm module
exposes them as `toBytes()` / `fromBytes()`. The JavaScript package, not the
wasm module, adds asynchronous `save(name)` / `load(name)` over a small storage
adapter interface with an IndexedDB adapter as the browser default and a
filesystem adapter as the Node default, so the same application code runs in
both. The bytes are exactly a VNDB file, so an index saved in a browser loads
in Rust, Python and C, and the reverse.

Decision 6 of the 2026-09-13 review chose this (option C) over bytes-only and
over a separate helper module, for coherence with the other bindings.

## Design

### Rust core

```rust
impl ApproxIndex {
    pub fn save_to(&self, writer: impl std::io::Write) -> Result<()>;
    pub fn load_from(reader: impl std::io::Read) -> Result<Self>;
    pub fn to_bytes(&self) -> Result<Vec<u8>>;      // save_to into a Vec
    pub fn from_bytes(bytes: &[u8]) -> Result<Self>; // load_from over a slice
}
```

`save` and `load` become thin wrappers over these plus the existing atomic
write. `load_from` applies every existing header, length and overflow check;
the corruption test suite runs against `from_bytes` as well as `load`.

### WebAssembly module (`wasm-bindgen`)

```ts
class ApproxIndex {
  toBytes(): Uint8Array;                        // a VNDB file
  static fromBytes(bytes: Uint8Array): ApproxIndex;
}
```

The returned `Uint8Array` is a copy out of wasm memory; the docs state the
size equals the file size and that a tombstoned index should be compacted
first. `FlatIndex` stays in-memory only, as in every binding.

### JavaScript package layer

```ts
interface Storage {
  put(name: string, bytes: Uint8Array): Promise<void>;
  get(name: string): Promise<Uint8Array | null>;
  delete(name: string): Promise<void>;
}

class ApproxIndex {
  save(name: string, storage?: Storage): Promise<void>;
  static load(name: string, storage?: Storage): Promise<ApproxIndex | null>;
}

export const indexedDbStorage: (dbName?: string) => Storage;   // browser default
export const fileStorage: (directory?: string) => Storage;      // Node default
```

- The default adapter is chosen at import time by runtime detection, the same
  mechanism the package already uses to pick the Node or browser wasm loader.
- `indexedDbStorage`: one database, one object store, bytes stored as a
  `Blob`. No dependencies.
- `fileStorage`: `name` is a file inside `directory` (default: the current
  working directory), written through a temporary file and rename, mirroring
  the core's atomic write.
- `load` resolves to `null` when nothing is stored under `name`, matching RFC
  0011's "a miss is a value, not an error".
- An Origin Private File System adapter is a later file that implements the
  same three functions; it changes nothing above.

### Size

The README gains the gzipped and unpacked size of the wasm bundle and of the
JavaScript layer, measured in CI by `check_npm_package.py`, so a regression is
visible.

## Decisions recorded

- 2026-09-13: option C (adapter behind `save`/`load`) accepted over bytes-only
  and over a separate helper module (decision 6).

## Alternatives rejected

- **Bytes only, storage left to the user.** Rejected: every user writes the
  same fifty lines, and the binding loses the `save`/`load` vocabulary.
- **A separate helper module (`saveIndex(db, key, index)`).** Rejected: two
  vocabularies in one package.
- **Persistence inside the wasm module.** Rejected: IndexedDB and the
  filesystem are asynchronous JavaScript APIs; the wasm module stays
  synchronous and dependency-free.
- **A wasm-specific compact serialisation.** Rejected: one format across all
  bindings is the point of VNDB.

## Compatibility and migration

- Additive on every surface. `save`/`load` by path unchanged in Rust and
  Python.
- No format change. VNDB golden fixtures and cross-engine tests apply to the
  byte path unchanged. When RFC 0013's container lands, `toBytes` emits it and
  `fromBytes` reads v2 and v3, like every other loader.
- The C ABI gains `vanedb_rs_index_save_to_buffer` / `_load_from_buffer` in
  the same release for parity (RFC 0002 rule: new functions, old unchanged).
- Python `to_bytes` / `from_bytes` for parity.

## Acceptance criteria

- [ ] `save_to` / `load_from` / `to_bytes` / `from_bytes` in the core, with
      the corruption suite running against the byte path.
- [ ] `toBytes` / `fromBytes` in the wasm module, tested in Node and in
      headless Chrome, Firefox and WebKit through the existing packaged-browser
      job.
- [ ] `save(name)` / `load(name)` with the `Storage` interface, the IndexedDB
      adapter and the file adapter; browser test that survives a page reload;
      Node test that survives a process restart; `load` of an unknown name
      resolves to `null`.
- [ ] Round trip: bytes saved in the browser load in Rust and reproduce the
      same search results on the conformance fixture, and a Rust-written
      fixture loads in the browser.
- [ ] Python `to_bytes` / `from_bytes`; C ABI buffer functions with a ctypes
      example.
- [ ] Bundle sizes (gzipped, unpacked) measured in CI and shown in the README.
- [ ] README's "persistence is not exposed" sentence removed; the JavaScript
      guide shows `save`/`load` in a browser and in Node; `CHANGELOG.md` entry.

## Evidence required before the claim

The three-browser packaged acceptance job, green, plus the reload and restart
tests.

## Out of scope

`DiskIndex` in WebAssembly (no mmap), the OPFS adapter, streaming load for
indexes larger than the tab's memory.
