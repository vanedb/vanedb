# VaneDB conformance

This directory owns the contract shared by the Rust and C++ engines. A
component-specific test may prove an implementation detail; the shared fixtures
prove behaviors both engines must honor. Their canonical bytes live in
[`vanedb/tests/fixtures/conformance/`](../vanedb/tests/fixtures/conformance/), so
the Rust crate includes every file its tests need. Both engines read that same
copy; this directory retains the specifications and independent generators.

## Initial regression set

The current conformance cases cover paired findings from both engines:

- cosine distance for small, identical vectors;
- non-finite vectors and queries are rejected at every public store/index
  boundary, persisted stores containing non-finite vectors fail to load, and
  defensive result ordering always places finite distances first. The shared
  input cases live in `non_finite_vectors.tsv`;
- HNSW construction in both engines rejects overflow in
  capacity-times-dimension and doubled-`M` derived sizes before allocation;
  the C++ legacy-format loader applies the same cases as corruption checks.
  The architecture-neutral symbolic cases live in `index_derived_sizes.tsv`;
- HNSW persistence with inconsistent external-id maps.

## Distance semantics

**Cosine.** The distance depends only on direction, so it must not change when
both inputs are rescaled. Normalisation divides by `sqrt(norm_a) * sqrt(norm_b)`
rather than `sqrt(norm_a * norm_b)`: the product grows with the fourth power of
magnitude, which would classify ordinary small vectors as zero and
overflowed to infinity for large ones.

A vector with no usable direction is defined to be `1.0` away from everything,
including itself. Finite inputs therefore never produce a non-finite cosine
distance. The rule is decided by the *computed* squared norm, so there are
three ways in:

- a zero vector;
- a squared norm that overflows `f32` (a magnitude from roughly 1.8e19 up).
  The norm decides, not any one component: 128 components of 1e19 are each
  under the bound and still sum past it;
- a squared norm that underflows `f32` to zero, which needs *every* component
  below roughly 2.6e-23 (`2^-75`) — one ordinary component keeps the norm
  usable. Such a vector is neither zero nor overflowing, and it is the case
  a caller is most likely to hit without noticing: the input looks ordinary
  and finite, and the answer is silently `1.0` — including against itself.

`cosine_scale_invariance.tsv` pins these cases for both engines and is consumed
by `vanedb/tests/cosine_conformance.rs` and
`cpp/tests/test_cosine_conformance.cpp`. Its 1e-18 … 1e18 rows pin the
invariance, and its 1e-25 rows pin the underflow end so a change in one engine
cannot quietly diverge from the other.

## Persisted identity

A loaded HNSW index must map each live external ID to the slot carrying that
ID. Both engines enforce this direction, where `stored_count` includes every
occupied slot:

```
id_map[id] == slot  implies  slot < stored_count  and  ext_ids[slot] == id
```

This rejects key/value mismatches, duplicated internal IDs and out-of-range
slots. Checking only map length and value range accepted files in which an ID
resolved to another slot's vector.

Rust supports deletion: a slot is live only when `id_map[ext_ids[slot]] == slot`.
A slot missing that mapping is a tombstone, and its external ID may have been
reused by another live slot. The public size counts live entries, while the
legacy file's `count` includes tombstones. Legacy C++ files instead require a
mapping for every stored slot. VNDB v2 encodes liveness with node flags, and both
readers rebuild the map from live slots while preserving tombstones. The frozen
C++ engine still has no public deletion API.

`index_id_map_consistency.tsv` pins the legacy map cases for both engines and is consumed
by `vanedb/tests/approx_id_map_conformance.rs` and
`cpp/tests/test_approx_id_map_conformance.cpp`.

## Universal persistence

The shared disk persistence format is **VNDB v1**, implemented by both engines.
The shared graph release candidate is **VNDB v2**, specified in
[graph/README.md](graph/README.md). Disk v1 is unchanged.

A shared persistence format must:

- begin with the literal four-byte magic `VNDB`;
- use explicitly sized, little-endian fields;
- reject truncated, overflowing, inconsistent, or unsupported data before
  exposing a partially loaded index;
- provide golden files written by each engine and loaded by both engines;
- preserve the serialized graph faithfully across a cross-engine load/save.

Independently building the same HNSW input in Rust and C++ does **not** require
identical adjacency. The conformance contract is structural validity,
compatible distance semantics, and an agreed recall floor. Cross-loading does
require the reader to preserve the graph represented in the file.

### VNDB v1 — DiskIndex

The `DiskIndex` payload already meets this contract and is specified here.
`vanedb/tests/fixtures/conformance/vndb/*.vndb` are the canonical fixtures, and both engines must
read them and reproduce them byte for byte.

All fields are little-endian. The header is exactly 32 bytes:

| Offset | Size | Field | Value |
|---|---|---|---|
| 0 | 4 | magic | the literal bytes `VNDB` |
| 4 | 4 | version | `1` |
| 8 | 8 | dim | components per vector, `u64` |
| 16 | 8 | count | number of vectors, `u64` |
| 24 | 4 | metric | `0` = L2, `1` = cosine, `2` = dot |
| 28 | 4 | reserved | `0` |

The header is followed by `count` ids (`u64` each), then `count * dim` vector
components (`f32` each), in the same order as the ids. No padding or trailing
bytes are allowed: the file length is exactly `32 + count * 8 + count * dim * 4`,
and both loaders reject any other length in either direction.

**Length equality is a loader rule, not just a writer rule.** A one-sided
"shorter than declared" check leaves a header that *understates* the geometry
accepted, because the expected length is derived from the header being checked:
one flipped bit in `dim` reads the payload at the wrong stride, and `get`
returns a vector that straddles two stored records. Equality does not make the
geometry tamper-proof — `32 + count * (8 + 4 * dim)` is the same for every
`(dim, count)` on that curve, and a header-only file (`count = 0`) fixes no
dimension at all — so a loader must also reject a `dim` too large to address,
which is what keeps the two engines agreeing on an empty store. Full integrity
needs a checksum, which v1 does not carry.

**Ids must be unique.** Both loaders reject a file where they are not, on the
same rule the HNSW payload uses above: the id map must end up the same size as
`count`. Neither builder writes duplicates, but `VNDB` is the format the two
engines exchange, so this is the loader's job rather than the writer's.
Accepting them is silent: `size` counts rows while lookups count ids, `get`
returns a row the id does not name, and one id appears twice in a single
result set.

The fixtures are generated by `conformance/vndb/generate.py` **from this
table**, never from either engine. That independence is the point: the two
engines are otherwise only compared to each other, so a change applied to both
passes every test. Encoding cosine as `2` in one engine keeps
`bench/tests/cross_engine_format.rs` green — both engines then agree on the
wrong meaning — while the fixture fails at offset 24.

Regenerate the fixtures only when this table changes, and treat any change to
them as a format version change.

### VNDB v2 — ApproxIndex

The [graph field table and fixtures](graph/README.md) define the release
candidate. Both engines reproduce the fixtures exactly, including all metrics,
continuation encodings, deleted entries and ID reuse. The benchmark harness's
`cross_engine_graph.rs` also crosses engine-written files both ways, verifies
byte-for-byte graph preservation, compares query results within the distance
tolerance, and inserts into imported graphs before crossing them again.

[Fixed legacy graph files](legacy_graph/README.md) protect the old Rust
`HNSW`/bincode and C++ `QVRD` readers. Migrate an old file in its original engine
by loading it and saving to a new path. They are compatibility fixtures for
those readers; the new format is defined by the VNDB v2 field table.
