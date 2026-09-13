# RFC 0006: WebAssembly persistence

- Status: draft
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

The Rust core already writes VNDB v2 through `graph_format::write(impl Write, ..)`
and reads it in `ApproxIndex::load`; only the path-based entry points are
public.

## Decision

Expose byte-based save and load in the Rust core, bind them in WebAssembly as
`Uint8Array` in and out, and ship a small IndexedDB helper module in the npm
package. The bytes are exactly a VNDB v2 file, so an index saved in a browser
loads in Rust, Python and C, and the reverse.

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

### WebAssembly

```ts
class ApproxIndex {
  save(): Uint8Array;                       // VNDB v2 bytes
  static load(bytes: Uint8Array): ApproxIndex;
}
```

`FlatIndex` stays in-memory only, as in every binding. The returned
`Uint8Array` is a copy out of wasm memory; the docs state the size equals the
file size and that a tombstoned index should be compacted first.

### IndexedDB helper

`@vanedb/wasm/idb` (a separate entry point so the core stays dependency-free):

```ts
export async function saveIndex(db: string, key: string, index: ApproxIndex): Promise<void>;
export async function loadIndex(db: string, key: string): Promise<ApproxIndex | null>;
```

Roughly fifty lines, no dependencies, one object store, the bytes stored as a
`Blob`. Node users get `fs` in the README example instead.

### Size

The README gains the gzipped size of the wasm bundle and the unpacked size,
measured in CI by `check_npm_package.py`, so a regression is visible.

## Alternatives rejected

- **Expose the Origin Private File System instead of IndexedDB.** Deferred:
  OPFS is faster for large files but its synchronous access handle only works
  in a worker; the helper can add an OPFS backend later without changing the
  core.
- **A wasm-specific compact serialisation.** Rejected: one format across all
  bindings is the point of VNDB.

## Compatibility and migration

- Additive on every surface. `save`/`load` by path unchanged in Rust and
  Python.
- No format change. VNDB v2 golden fixtures and cross-engine tests apply to
  the byte path unchanged.
- The C ABI gains `vanedb_rs_index_save_to_buffer` / `_load_from_buffer` in
  the same release for parity (RFC 0002 rule: new functions, old unchanged).

## Acceptance criteria

- [ ] `save_to` / `load_from` / `to_bytes` / `from_bytes` in the core, with
      the corruption suite running against the byte path.
- [ ] `save` / `load` in the wasm package with tests in Node and in headless
      Chrome, Firefox and WebKit through the existing packaged-browser job.
- [ ] Round trip test: bytes saved in the browser load in Rust and reproduce
      the same search results on the conformance fixture; and a Rust-written
      fixture loads in the browser.
- [ ] `@vanedb/wasm/idb` helper with a browser test that survives a page
      reload.
- [ ] Python `save`/`load` accept `bytes`-like objects (`to_bytes`,
      `from_bytes`) for parity.
- [ ] C ABI buffer functions with a ctypes example.
- [ ] Bundle sizes (gzipped, unpacked) measured in CI and shown in the README.
- [ ] README's "persistence is not exposed" sentence removed; `CHANGELOG.md`
      entry.

## Evidence required before the claim

The three-browser packaged acceptance job, green, plus the reload test.

## Out of scope

`DiskIndex` in WebAssembly (no mmap), OPFS backend, streaming load for
indexes larger than the tab's memory.
