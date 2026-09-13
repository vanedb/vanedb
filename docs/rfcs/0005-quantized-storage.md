# RFC 0005: Quantized storage

- Status: draft
- Milestone: 0.3.0
- Tracking issue: #200
- Supersedes / superseded by: none

## Problem

Every index stores `f32`. At 768 dimensions, 100k vectors is 307 MB before
graph links, which is the whole memory budget of a background process on a
mid-range phone and past the comfortable limit of a browser tab. The market
treats compression as table stakes for on-device search: EdgeVec leads with
"32x memory reduction", Turso ships 1-bit vectors, USearch stores f16, i8 and
binary. "Edge AI" is not credible without it.

## Decision

Add two storage encodings behind a builder option: 8-bit scalar quantization
with per-index affine parameters, and 1-bit binary quantization with optional
`f32` rescoring. The encoding is a property of the index, chosen at build time,
written to the file under a new VNDB kind, and invisible to the search API
apart from distance values being approximate.

## Design

### Encodings

| Encoding | Bytes per dimension | Distance kernel | Distance error |
|---|---|---|---|
| `F32` (default, unchanged) | 4 | existing | none |
| `Int8` | 1 | i8 dot / L2 via NEON `sdot`, AVX2 `vpmaddubsw`+`vpmaddwd`, scalar reference | small, documented per metric |
| `Binary` | 1/8 | Hamming via popcount (NEON `cnt`, x86 `popcnt`), scalar reference | large; use with rescoring |

- `Int8`: `q = round((x - offset) / scale)` with `offset`, `scale` per index,
  fitted from the first batch or supplied explicitly. Values outside the fitted
  range saturate; `add` records how many components saturated and exposes it
  as a counter. Cosine stores the original norm per vector as `f32` so the
  cosine distance stays exact up to quantization of the direction.
- `Binary`: sign of each component after centring on the per-index mean.
  Requires `dim` to be a multiple of 8; other dimensions are padded and the
  padding is recorded.
- **Rescoring.** `SearchParams::rescore(n)` for `Binary`: the walk runs on
  binary distances with a beam of `n × k`, then the top `n × k` candidates are
  re-ranked with `f32` vectors that the index keeps only when built with
  `rescore_store(true)`. Without the `f32` copy, rescoring is a
  `VaneError::Validation`.

### API

```rust
pub enum Storage { F32, Int8 { offset: Option<f32>, scale: Option<f32> }, Binary { keep_f32: bool } }

ApproxIndex::builder(dim, metric).storage(Storage::Int8 { .. })
FlatIndex::builder(dim, metric).storage(..)
DiskIndexBuilder::new(dim, metric).storage(..)
index.storage() -> Storage
```

`get` / `get_vector` return the dequantized `f32` vector (or the stored `f32`
copy when present) and document that it is lossy.

Python: `storage="int8"` / `"binary"` constructor keyword and a `storage`
property. WebAssembly: an options object. C ABI: new `_new_with_storage`
constructors; existing constructors unchanged.

### File format

- VNDB v2 (graph) gains kinds `2` (HNSW, int8 vectors) and `3` (HNSW, binary
  vectors, optional f32 section). The v1 disk header has no kind field and
  version `2` already names the graph format, so quantized disk files use a
  new version `3` header that adds a kind field (`1` int8, `2` binary); f32
  disk files stay version `1`. Exact layouts are specified in `conformance/`
  before implementation, with golden fixtures.
- Quantization parameters (`offset`, `scale`, mean vector, padding) live in
  the header or a fixed-length parameter section, never inferred from data.
- Existing readers reject the new kinds with `VaneError::Corrupt { .. }`
  naming the kind, as they do today for unknown kinds. Old files load
  unchanged.
- The frozen C++ engine does not learn the new kinds. Cross-engine conformance
  for quantized files is: the C++ reader rejects them cleanly, verified by a
  fixture test. Loading a Rust f32 file in C++ is unchanged.

### Kernels

Scalar reference first, in `distance/scalar.rs`, held to an `f64`
computation in `tests/search_correctness.rs`. NEON and AVX2 paths keep the
multi-accumulator unrolling rule from `AGENTS.md`; the kernel-bound-by-both-
lengths rule from the 0.1.1 security fix applies from the first commit.
Property tests compare all three paths on random inputs including the tail
lengths.

## Alternatives rejected

- **Product quantization.** Rejected for now: needs training, codebooks in the
  file, and a much larger kernel surface; the two encodings here cover the
  memory budget that the audience actually has.
- **f16.** Deferred: halves memory but needs `f16` arithmetic support that is
  uneven across NEON, AVX2 and wasm; int8 gives 4x with better-supported
  instructions. Can be a later kind.
- **Quantize on the fly at search time from f32 storage.** Rejected: does not
  save memory, which is the point.

## Compatibility and migration

- Additive API. Default storage remains `F32`; every existing test passes
  unchanged.
- New file kinds; existing identifiers untouched; readers for existing files
  retained. Both `HnswData` mirrors in `tests/corruption_tests.rs` and
  `tests/approx_id_map_conformance.rs` are updated in lockstep with any layout
  change, per `AGENTS.md`.
- A quantized index cannot be converted back to `F32` losslessly; the docs say
  to keep the source vectors.

## Acceptance criteria

- [ ] Format layouts for the new kinds specified in `conformance/` with golden
      fixtures and corruption fixtures; the C++ engine's rejection tested.
- [ ] Scalar, NEON and AVX2 kernels for i8 and binary with property tests and
      the `f64` reference suite.
- [ ] `Storage` API in Rust, Python, WebAssembly and C ABI with tests.
- [ ] Memory measurement: resident bytes for 100k × 768-d under each encoding,
      recorded in the README.
- [ ] Recall table: recall@10 for `Int8` and for `Binary` with rescoring
      factors 1, 4, 16 against `F32`, on the RFC 0003 embedding fixture.
- [ ] Interleaved bench: `F32` paths unchanged within the noise floor.
- [ ] Saturation counter and dequantized `get` documented.
- [ ] `CHANGELOG.md` entry; README "no quantization" disclaimer removed.

## Evidence required before the claim

Recall and memory on real embeddings, on a dedicated machine; on-device
memory on at least the Android emulator with the figure labelled as such.

## Out of scope

Product quantization, f16, GPU kernels, changing the default storage.
