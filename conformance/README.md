# VaneDB conformance

This directory owns the contract shared by the Rust and C++ engines. A
component-specific test may prove an implementation detail; a fixture here
proves a product behavior that both implementations must honor.

## Initial regression set

The first conformance cases will cover the paired findings already present in
both issue trackers:

- cosine distance for small, identical vectors;
- non-finite distances in top-k ordering;
- HNSW persistence with inconsistent external-id maps.

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
