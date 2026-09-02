# VaneDB conformance

This directory owns the contract shared by the Rust and C++ engines. A
component-specific test may prove an implementation detail; a fixture here
proves a product behavior that both implementations must honor.

## Initial regression set

The first conformance cases will cover the paired findings already present in
both issue trackers:

- cosine distance for small, identical vectors;
- non-finite vectors and queries are rejected at every public store/index
  boundary, persisted stores containing non-finite vectors fail to load, and
  defensive result ordering always places finite distances first. The shared
  input cases live in `non_finite_vectors.tsv`;
- HNSW construction in both engines rejects overflow in
  capacity-times-dimension and doubled-`M` derived sizes before allocation;
  the C++ legacy-format loader applies the same cases as corruption checks.
  The architecture-neutral symbolic cases live in `hnsw_derived_sizes.tsv`;
- HNSW persistence with inconsistent external-id maps.

## Distance semantics

**Cosine.** The distance depends only on direction, so it must not change when
both inputs are rescaled. Normalisation divides by `sqrt(norm_a) * sqrt(norm_b)`
rather than `sqrt(norm_a * norm_b)`: the product grows with the fourth power of
magnitude, which previously classified ordinary small vectors as zero and
overflowed to infinity for large ones.

A vector with no usable direction — a zero vector, or one whose squared norm
overflows `f32` — is defined to be `1.0` away from everything, including
itself. Finite inputs therefore never produce a non-finite cosine distance.

`cosine_scale_invariance.tsv` pins these cases for both engines and is consumed
by `vanedb/tests/cosine_conformance.rs` and
`cpp/tests/test_cosine_conformance.cpp`.

## Universal persistence

The first public persistence format is **VNDB v1**. Existing Rust and C++
format version numbers are pre-release implementation details and do not
determine the public version.

The format design must:

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

The concrete field table, bounds, and canonical fixtures will land with the
format implementation rather than being guessed during the repository
migration.
