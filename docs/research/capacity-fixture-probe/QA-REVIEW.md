# Independent QA review — fixture capacity probe

**PASS. No open Critical/Major findings.** Final counting source, runner, parser, binary and nine retained result cases were reviewed. Two calibration assumptions were corrected before the full run: dependency checking now covers the engine's resolved closure, and hash-table control tail is host-specific. Neither correction changes engine source. No full graph construction was duplicated for QA.

Candidate: `35758c6ccac0b9e657be8d80e84d1744378df120`, `/Users/anton/code/codex/vanedb-020-integration`, independently checked clean after completion. Scope is requested live allocation bytes and graph geometry for actual pinned embedding prefixes. This report makes no timing, RSS, peak-memory, device-capacity, recall, mobile, or controlled historical-comparison claim.

## Findings and resolved risks

- **PASS — allocator:** `src/main.rs:14–60` counts successful ordinary/zeroed allocations, successful realloc growth/shrink by the size difference, and deallocation. Null results preserve the counter. Accounting performs no heap allocation. Independently executed `--self-test` exercises all these paths, zero initialization and data preservation, including simulated null-result accounting without inducing OOM.
- **PASS — scopes:** `src/main.rs:148–248` loads fixture and keeps input arrays alive before baseline, isolates map calibration, drops disk builder before open measurement, samples graph before save, asserts unchanged live heap after save, and separates retained post-drop bytes from released bytes. JSON/reporting runs after snapshots. Nine result PIDs are distinct; every case ran in a fresh process.
- **PASS — provenance:** `run_probe.py:27–29,46–76,98–99` checks exact candidate HEAD and tracked cleanliness, retains lockfiles and Cargo metadata, checks resolved engine package identities/checksums, records source/binary hashes, then checks source again. QA independently recomputed all five source-file hashes and binary hash, compared retained/current locks, and verified actual clean source HEAD. Probe-only SHA2 dependencies are correctly outside the engine closure check. The earlier all-name comparison falsely rejected `cpufeatures 0.2.17`; it was corrected before final runs.
- **PASS — parser:** `verify_files.py:30–107` validates native Rust VNDB v2 graph header, count/capacity/parameters, IDs/vector bytes, flags/levels, layer caps, exact EOF, entry/max level, edge bounds, self/duplicate edges and target-node levels. Restricting continuation to encoding 1 with zero payload is intentional for this probe, not a general-purpose reader claim.
- **PASS — historical formula correction:** section 4.4's Linux x86_64 `17 × buckets + 16` formula is not a portable 64-bit formula. Actual host Rust source predicts `17 × buckets + 8`; no constant was fitted to make the measurements pass. Details below.
- **Interpretation constraint:** `results/metadata.json` field `rustc_cfg` is bare `rustc --print cfg`, not the exact release invocation; it includes compiler-default `debug_assertions`. The executable came from the retained release build. Treat that field as compiler/target defaults and retain the implementer's separate profile annotation.

## Independent executions and fixture identity

Fixture `/Users/anton/code/codex/vanedb-pr-214/bench/compare/fixtures/embeddings.vnef` independently passed full streamed SHA256 `4676911521f13dc6592d102cf3b4fcace85c6b62b3011e9bd57557dfc9c01f2d`; header VNEF v1, 100000 document rows, 1000 queries, 768 dimensions, cosine, reserved 0; exact length 311072028 bytes. All runs use the first n document rows in fixture order and original IDs.

Independent bounded checks:

- Executed allocator self-test successfully.
- Executed parser suite: 3/3 tests, including all 184 truncation prefixes of handcrafted graph.
- Parsed candidate canonical `l2_rng1.vndb`: nodes-by-level `[1,2]`, degree sums `[6,2]`, layers 5, links 8, exact-fit neighbor representation 256 bytes. Additional independently constructed duplicate-edge, above-target-level edge and nonfinite-vector mutations all rejected.
- Wrote and ran a separate inline Python parser **without importing `verify_files.py`** over all three retained graph files. Independently checked headers, every ID/vector byte against fixture, all caps/topology/levels, exact EOF, degree histograms, nodes by level, layer/link counts and file SHA256 against retained verification JSON.
- Independently checked all three disk files' full header, ID and vector regions and exact file length against fixture; independently checked all flat/disk byte formulas, raw snapshot arithmetic and zero residuals.

## Host-specific formula evidence

Host: aarch64-apple-darwin, Rust 1.98.1 (Homebrew). Local sysroot source root:
`/opt/homebrew/Cellar/rust/1.98.1/lib/rustlib/src/rust/library`.

`std/Cargo.toml:23` selects hashbrown 0.17.1; `std/src/collections/hash/map.rs:4` imports it. In bundled `vendor/hashbrown-0.17.1/src/control/group/mod.rs:24–33`, aarch64 + neon + little endian selects NEON; independently read compiler cfg confirms these conditions. `control/group/neon.rs:14–21` defines a 64-bit `uint8x8_t` group and its width. `src/raw.rs:221–223` allocates aligned element bytes plus buckets plus group width. For 16-byte `(u64,usize)` entries aligned to 8, the formula is exactly `17 × buckets + 8`.

For these n, buckets is the next power of two at least `ceil(n × 8 / 7)`. Flat one-batch payload is `n × (8 + 4 × 768)`. Disk file is `32 + n × (8 + 4 × 768)`. The independently derived host formulas agree to the byte:

| n | Flat requested heap | Disk map heap | Disk file | Approx vector chunk bytes | Final chunk padding |
|---:|---:|---:|---:|---:|---:|
| 8192 | 25509896 | 278536 | 25231392 | 25165824 | 0 |
| 10000 | 31078536 | 278536 | 30800032 | 31457280 | 737280 |
| 100000 | 310228232 | 2228232 | 308000032 | 307494912 | 294912 |

The preliminary QA table, derived from the historical x86_64 study, was 8 bytes higher for flat and disk heap only. Calibration exposed this platform assumption; actual installed standard-library source independently explains it.

## Graph geometry and ownership checks

| n | Build live delta | Released on index drop | Retained after drop | Layers | Links | Mean degree 0 |
|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 28121480 | 28105096 | 16384 | 8745 | 226615 | 26.5865478515625 |
| 10000 | 35014312 | 34981544 | 32768 | 10664 | 276069 | 26.5487 |
| 100000 | 342442992 | 342180848 | 262144 | 106517 | 2743127 | 26.38885 |

Nodes by level: 8192 → `[7672,489,29,2]`; 10000 → `[9377,585,35,3]`; 100000 → `[93895,5717,366,20,2]`. Every saved degree histogram also matched the independent parser, not merely its mean.

Source `vanedb/src/approx/storage.rs:29–34,79–87` gives 256 vectors per 768-d chunk, each chunk 786432 bytes. Let `c = ceil(n / 256)`. Empty reservation is `37n + 24c`: 303872, 370960, 3709384 bytes; all measured values agree. The built-minus-drop residue matches `2 × next_power_of_two(n)` from thread-local `Vec<u16>` growth (`approx/mod.rs:51–92`). This supports the TLS interpretation while retaining the measured residue separately.

Subtracting source-derived vector chunks, chunk headers `24c`, non-neighbor sibling arrays `13n`, and calibrated map bytes from released-on-drop heap yields the **neighbor-storage allocation residual**, including node Vec headers:

| n | Neighbor residual bytes | Bytes/node | Serialized exact-fit neighbor bytes | Exact-fit bytes/node |
|---:|---:|---:|---:|---:|
| 8192 | 2553472 | 311.703125 | 2219408 | 270.923828125 |
| 10000 | 3114768 | 311.4768 | 2704488 | 270.4488 |
| 100000 | 31148320 | 311.4832 | 26901424 | 269.01424 |

Exact-fit is `24n + 24 × layers + 8 × links`. This is a hypothetical packed-capacity Vec representation computed from saved lengths, not an observed allocation capacity. The residual is derived by subtraction; the probe does not instrument each graph Vec individually. Allocation bytes exclude allocator overhead, stack, mapped pages and RSS. Save and parsing occur outside the build sample; transient build peaks are not measured.

## Exact-content identity reviewed

Full-run `results/metadata.json` source identities matched current files on independent recomputation:

- `src/main.rs`: `de8db76c0140f2266a432bd6b50d143b8dcbd9649a01433e7eb877913a3a4491`
- `Cargo.toml`: `753f731e7fc0b9da4b1329ad08f77950f334a3914d98128c39b7265cec6c8538`
- `Cargo.lock`: `c325fe51eb13103fe2ba53934245fc615e4a04a2b5e5bd181fd82b9a3469fa58`
- `verify_files.py`: `ade34d0da1d0e892cfb2e1436b70fb47c880caa72978860085e4986c0e68fc5b`
- `run_probe.py`: `29e7f8d0d11d02d678641b033f0868323fe93d1aafcf031b1f9e336a2780bd2c`
- Release executable: `086304919cd32dc8a102015d517210cefc8d958a2202083f99f88ef878effab0`

`results/complete.json` covers all three counts and all three kinds. This is evidence for capacity study open question 2 on this one fixture/metric/seed/configuration; open question 1's physical-device resident-memory/latency/recall work remains separate.
