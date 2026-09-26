# Real-embedding allocator probe

This scratch study addresses open question 2 in the candidate's `docs/research/capacity.md`: requested heap bytes and graph link density on the now-hosted real-embedding fixture. It does not measure timing, RSS, peak memory, cold-cache behavior, quantized or mapped graphs, or any platform capacity limit. It does not complete issue #210 or approve a release.

Source is the unmodified engine candidate `35758c6ccac0b9e657be8d80e84d1744378df120`, path dependency `../vanedb-020-integration/vanedb`, with feature `disk`. The full original fixture SHA-256 is `4676911521f13dc6592d102cf3b4fcace85c6b62b3011e9bd57557dfc9c01f2d`: 100,000 document rows and 1,000 held-out queries, 768 dimensions, nomic-embed-text-v1.5 / BeIR nq. Each case inserts the first n document rows and their original IDs; queries are not inserted. No subset is asserted to be a random or representative sample.

The scripts preserve their measured scratch layout. A copy placed inside a repository's documentation tree is an evidence archive, **not** a directly runnable in-tree crate. To reproduce elsewhere, copy this text/source directory out as `020-capacity-fixture-probe` and prepare this sibling layout:

```text
study-root/
  020-capacity-fixture-probe/    # these source/text files, preserving src/
  vanedb-020-integration/       # clean vanedb checkout at exact source SHA above
  vanedb-pr-214/bench/compare/fixtures/embeddings.vnef
```

`vanedb-pr-214` here is just the fixture directory name retained by the script; it need not be a Git checkout. Obtain the fixture from [the pinned fixture release](https://github.com/vanedb/vanedb/releases/download/compare-fixture-v1/embeddings.vnef) and verify the SHA above. Obtain the source from [the engine commit](https://github.com/vanedb/vanedb/commit/35758c6ccac0b9e657be8d80e84d1744378df120). Both checks are also enforced at run time. The source checkout must retain its `.git` identity, candidate Cargo.lock and fixture metadata file. Avoid renaming these directories unless intentionally producing a new probe revision and recording its hashes.

Measured cases are n=8192, 10000, 100000 for FlatIndex, DiskIndex, ApproxIndex. All use cosine on the exact original f32 vector bits. ApproxIndex uses explicit capacity(n), M=16, construction beam=200, search beam=50 and seed=7. Flat and Approx use add_batch. Disk builder uses serial add/save and is dropped before measuring the open mapped index. Its mapped bytes are deliberately excluded from requested heap counts.

## Reproduce the retained experiment

This directory preserves the exact probe source used for the measurement,
including its historical sibling paths. Copy it to a scratch directory before
running; it is not part of the engine Cargo workspace. The source and raw JSON
retain original absolute paths as provenance, not as required output locations.

Create this layout:

```text
scratch/
  020-capacity-fixture-probe/                 # copy this directory here
  vanedb-020-integration/                    # engine checkout at the SHA above
  vanedb-pr-214/bench/compare/fixtures/
    embeddings.vnef                         # full pinned fixture, not another checkout
```

Fetch the exact engine commit from `https://github.com/vanedb/vanedb.git`.
Download the fixture from the [official compare-fixture-v1 release](https://github.com/vanedb/vanedb/releases/download/compare-fixture-v1/embeddings.vnef)
and verify the SHA-256 above. The probe and runner both refuse a different
fixture; the runner also requires the exact clean engine commit and matching
engine dependency lock entries. The recorded run used macOS ARM64 and Rust
1.98.1; other compiler/target layouts may produce different heap counts.

Run from `scratch/020-capacity-fixture-probe/`:

```sh
cargo build --release --locked --offline
target/release/capacity-fixture-probe --self-test
python3 -m unittest -v test_verify_files.py
python3 run_probe.py --out results-new
```

Each output directory must be fresh; retained outputs are never overwritten or deleted by the runner. `--counts 32 --out calibration-new` runs a small structural calibration with the same pinned input. A cached Rust dependency set is needed for `--offline`. Omit offline only if fetching the exact committed lockfile is intended. Python uses only its standard library (3.11+ for tomllib/file_digest).

`src/main.rs` overrides every GlobalAlloc entry point, including alloc_zeroed. It counts requested layout sizes on successful allocations/reallocations only. The self-test exercises nonzero allocations, zeroing, growth, shrinkage, deallocation, and simulated null allocation/reallocation accounting without issuing an OOM request. This measures neither allocator headers/rounding nor stack storage, mmap pages, or process-wide resident memory.

`run_probe.py` launches a fresh process for every (index kind, n) case. The full fixture is checksummed and parsed before the baseline; the selected prefix vector/ID arrays and arguments remain alive through every snapshot, while full-file parsing temporaries are dropped before the baseline. Serialization occurs after the build snapshot and must leave retained heap unchanged. JSON formatting occurs after all snapshots. Dropping the index supplies a second count: built minus after-drop is the allocation released with the index; after-drop minus baseline separately records retained scratch. ApproxIndex's thread-local VisitedBuffer remains after the index drop. Flat/Disk residuals must be zero.

A separate stateless-hasher HashMap<u64,usize> is built and dropped before each index baseline to calibrate this standard-library/architecture allocation layout. Rust 1.98.1's bundled hashbrown0.17.1 uses an 8-byte NEON control group on this ARM64 host, whereas the old Linux x86-64 report used a 16-byte group. Actual calibrated bytes and usable capacity are retained; formulas must not silently copy the old +16 term.

`verify_files.py` independently parses persisted graphs with explicit field widths and read bounds, validates header parameters, IDs and every vector byte against the pinned fixture, exact EOF/empty native RNG continuation, maximum/entry levels, degree caps, duplicate/self/out-of-range edges and neighbor levels. Its exact-fit neighbor formula describes serialized geometry interpreted as hypothetical exact-fit Vec capacities; it is not an observation of actual Vec capacity. Disk checks independently verify its v1 header, exact file length, IDs and vector bytes. `test_verify_files.py` uses an independently constructed tiny graph, every truncation position and mutated headers/edges.

The runner verifies actual engine HEAD and tracked-source cleanliness before and after the campaign, retains both lockfiles, resolves the engine dependency closure through cargo metadata and requires identical locked package identities/checksums. Probe-only SHA-256/JSON dependencies are outside that engine closure. Results include individual commands, PIDs, compiler/host metadata, source/binary hashes and independent file checksums.

**Build-profile annotation:** the actual build command was `cargo build --release --locked --offline` (optimized release profile; see build.log). `results/metadata.json` field `rustc_cfg` is explicitly the output of bare `rustc --print cfg`, describing compiler defaults/host target selection; its `debug_assertions` line does not describe the release binary. It is useful for aarch64/little-endian/NEON selection, not proof of per-crate build flags. No engine or probe source changed after the full campaign began.

See [REPORT.md](REPORT.md), `results/`, and independent [QA-REVIEW.md](QA-REVIEW.md).
This repository retains text evidence and executable probe source. The binary,
large generated VNDB files, and initial calibration scratch outputs remain in
the original task workspace; they are not committed here. Their absence does
not turn file-validation JSON into independent proof of unretained bytes: the
QA report records the independent check while those files were available,
and a fresh run regenerates and validates them. Raw per-file sizes and SHA-256
values are retained in `results/*file-verification.json`. The initial failed
formula assumption and ARM control-tail correction are explained in the report.

## Safe calibration rerun

CodeQL flagged a raw-pointer write in the original calibration self-test. The
current source replaces that test with safe owned `Vec<u8>` operations for
allocation, zeroing, growth, shrinkage and deallocation; simulated null-result
accounting checks remain separate. The global allocator wrapper and measured
workload are unchanged. The alert was not dismissed or suppressed.

All nine cases were rerun after the correction. This archive's `results/` now
contains the fresh run from the original workspace's `results-safe-calibration/`.
Every heap delta, calibrated map count, graph statistic and generated-file
checksum is identical to the earlier run. Absolute live counters increased by
17 bytes because the retained output-path argument is longer; that common
offset cancels from every delta.

[The semantic comparison](safe-calibration-semantic-diff.json) and
[independent fix QA](QA-SAFE-CALIBRATION-REVIEW.md) and
[analyst review](ANALYST-SAFE-CALIBRATION-REVIEW.md) document the correction.
`QA-REVIEW.md` and `ANALYST-REVIEW.md` retain the earlier campaign's identities
as historical evidence; current measured source and binary hashes are in
`results/metadata.json`. Original source and text outputs remain in the preceding Git revision. The
original binary and large files remain in the task workspace. The raw-pointer
self-test is not retained as active source in this directory.
