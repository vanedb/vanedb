# Real-embedding requested-heap study — 23 September 2026

**PASS for the bounded allocator/link-density study.** All nine fresh-process cases completed, their retained files passed independent format/fixture checks, and candidate source remained clean at `35758c6ccac0b9e657be8d80e84d1744378df120`. Independent reviews are recorded in [QA-REVIEW.md](QA-REVIEW.md) and [ANALYST-REVIEW.md](ANALYST-REVIEW.md). This supplies the missing real-fixture sanity check from capacity-study open question 2; it does not complete the issue #210 hardware/mode table or alter the release scope.

## Measurements

All values below are decimal bytes. Approx heap is the allocation released when the index is dropped. TLS is the separately observed residual after dropping the index. Flat/Disk after-drop residuals were zero. Allocation during serialization had fully returned before the after-save snapshot in every graph case.

| First n documents | Flat requested heap | Disk open requested heap | Approx requested heap | Approx retained TLS | Approx built delta including TLS |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8,192 | 25,509,896 | 278,536 | 28,105,096 | 16,384 | 28,121,480 |
| 10,000 | 31,078,536 | 278,536 | 34,981,544 | 32,768 | 35,014,312 |
| 100,000 | 310,228,232 | 2,228,232 | 342,180,848 | 262,144 | 342,442,992 |

These are not peak allocation counts. In particular the Disk builder is dropped before opening the file; only the open DiskIndex heap is measured. Its 25,231,392 / 30,800,032 / 308,000,032-byte files are mapped, but mappings and page residency are not accounted by this allocator. A small Disk heap must never be represented as total memory consumed by the data or the process.

| n | Mean layer-0 degree | Mean links/node across layers | Nodes by maximum level | Graph file bytes |
| --- | ---: | ---: | --- | ---: |
| 8,192 | 26.58654785 | 27.66296387 | 7,672 / 489 / 29 / 2 | 27,179,872 |
| 10,000 | 26.54870000 | 27.60690000 | 9,377 / 585 / 35 / 3 | 33,173,960 |
| 100,000 | 26.38885000 | 27.43127000 | 93,895 / 5,717 / 366 / 20 / 2 | 331,597,248 |

The graph was built on cosine real embeddings with explicit capacity(n), M16, efConstruction200 and seed7. The previous document's illustrative random vectors used L2 on Linux x86-64/Rust1.94, whereas this run used macOS ARM64/Rust1.98.1. The increased degree against its ~23.7 figure is an observation across different datasets/metric/toolchain/host, not a controlled estimate of a causal dataset effect or a regression.

## Source-derived memory interpretation

The engine stores 256 vectors per 768-dimensional chunk (the largest power-of-two row count fitting the 1 MiB target), so a fully allocated chunk requests **786,432 B**, not 1 MiB. Chunk payload is ceil(n/256) × 786,432: 25,165,824 / 31,457,280 / 307,494,912 B. Last-chunk unused vector storage is 0 / 737,280 / 294,912 B.

For these cases below the sibling reserve cap, source `vanedb/src/approx/storage.rs:29–34,56–59,79–87` and `approx/mod.rs:1092–1114` justify the non-neighbor component: vector chunk payload + 24 × ceil(n/256) chunk headers + 13n for IDs/levels/deleted flags + the independently calibrated ID-map allocation. Subtracting that from the released-on-drop index heap leaves neighbor headers/list storage including real Vec spare capacities. This subtraction is a source-based interpretation, not an instrumented observation of each private field.

| n | Inferred neighbor headers/list allocation | B/node | Exact-fit neighbor geometry | B/node |
| --- | ---: | ---: | ---: | ---: |
| 8,192 | 2,553,472 | 311.703125 | 2,219,408 | 270.923828 |
| 10,000 | 3,114,768 | 311.476800 | 2,704,488 | 270.448800 |
| 100,000 | 31,148,320 | 311.483200 | 26,901,424 | 269.014240 |

Exact-fit geometry is `24n + 24 × total node-layers + 8 × total edges`, computed by the independent persisted-file parser. It includes a per-node Vec header, each per-layer Vec header and each neighbor index, but assumes capacity equals length. It cannot reveal Vec capacities and is not claimed as measured live heap. This run's ~311.5 B/node inferred actual neighbor allocation sits within the prior 300–360 B/node planning range for these particular fixture prefixes and parameters; it is not a universal upper bound.

The standard HashMap calibration matches `17 × buckets + 8` on this ARM64 target. Candidate FlatIndex's add_batch and open DiskIndex match that independently calibrated map plus their source-defined arrays exactly. The old study's `+16` was a Linux x86-64 control-group width. This 8-byte absolute difference was caught by the first calibration and resolved from Rust1.98.1's bundled hashbrown0.17.1 source, not patched as an unexplained fitting constant. Retained raw counts and the independent review preserve the correction.

## Reproduction and evidence boundaries

`README.md` gives the commands and measurement lifecycle. Source and binary hashes, toolchains/host, full fixture/model/corpus identity, PIDs, commands, candidate/probe locks and engine dependency closure are in `results/metadata.json`, `results/cargo-metadata.json`, and each case's JSON. The runner verifies source identity/cleanliness before and after the run, and its `complete.json` records successful completion. Individual VNDB files remain retained in the original task workspace; the repository includes their independently verified SHA-256 and graph/disk details, with source to regenerate them. Large generated files are not committed.

No timing, RSS, cache state, peak memory, physical-device success, 1M/10M scaling result, mapped HNSW implementation or quantized mode was measured. No PR, protected-main merge, publication, capacity-scope decision or release-gate waiver was performed. The remaining gates in the [release readiness record](../../release/0.2.0-readiness.md) still apply.
