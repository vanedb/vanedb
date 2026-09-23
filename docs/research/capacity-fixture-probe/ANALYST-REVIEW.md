# Independent analyst review — real-fixture allocator study

Reviewed 23 September 2026 against candidate `35758c6ccac0b9e657be8d80e84d1744378df120` and the completed nine-case results.

## Findings, risks and limits

No blocking correctness or methodological finding. One P3 documentation precision issue was sent to the implementer: README.md:22 originally said the full fixture's buffers remain alive throughout the snapshots. `src/main.rs:99–133` drops full-file parsing temporaries when `fixture()` returns; the selected prefix vectors/IDs and arguments remain alive (`src/main.rs:247–248`). The implementer corrected this wording before final review; no open finding remains and measured deltas are unaffected.

The following limits are material and correctly preserved in REPORT.md:

- Built requested heap includes retained thread-local scratch. The released-on-drop figure and residual are separately measured (`src/main.rs:189–205,233–244`). The residual agrees with `Vec<u16>` VISITED growth in candidate `vanedb/src/approx/mod.rs:51–84`: 16,384 / 32,768 / 262,144 bytes. It is not valid to describe the entire build delta as index-owned memory.
- Neighbor-list allocation is inferred by subtracting source-defined non-neighbor allocations, not directly measured field by field. Chunk payload is 256 × 768 × 4 bytes; chunk-list headers, 13n sibling-array bytes and the target-calibrated map must all be subtracted. Candidate `vanedb/src/approx/storage.rs:29–34,56–59,79–87` and `approx/mod.rs:1092–1114` justify this calculation for these counts below RESERVE_CAP. The empty-index deltas also equal 37n + 24 × ceil(n/256), consistent with these reservations.
- Persisted degree geometry cannot reveal original Vec capacities. `verify_files.py:106–107` correctly labels 24n + 24 × node-layers + 8 × links as an exact-fit model. It is a lower bound for the observed representation, not live allocation or a compact-format implementation result.
- Fixture prefixes are neither independent draws nor demonstrated representative corpus samples. The three nested prefixes support this pinned fixture/parameter observation, not universal link-density estimates, confidence intervals, extrapolation or an upper bound.
- The earlier random/L2/Linux x86-64/Rust 1.94 study differs in metric, corpus, host and toolchain. Its ~23.7 versus the new ~26.4–26.6 layer-0 degree is descriptive, not a controlled causal effect. REPORT.md:23 states this correctly.
- Requested layout bytes exclude allocator overhead, stack, mappings and residency. Flat/Disk zero residual and the Disk map formula do not establish total process memory, peak build memory or physical-device capacity. None of timing, RSS, recall, device capacity or release readiness is established here.

## Independent checks completed

I read the analyst role baseline and candidate AGENTS.md, reviewed the allocator/runner/parser, and independently checked all completed outputs. I did not modify probe/engine source or rebuild the 100k graph.

1. Rehashed the full original VNEF fixture and confirmed its exact recorded identity, model/corpus metadata and candidate tracked-source clean SHA.
2. Rehashed all five recorded probe source/manifests and the executable; all match results/metadata.json. Compared both retained lockfiles with current originals and independently traversed the saved cargo metadata dependency closure from vanedb. Every reachable package identity/source/checksum is present in the candidate lock, including rand 0.10.2; no engine dependency drift was found. The path dependency points to the reviewed candidate.
3. Checked all nine snapshot arithmetic identities, save equality, per-case HashMap calibration and Flat/Disk formulas. On this target map bytes equal 17 × buckets + 8; Flat adds 3080n bytes and Disk adds zero requested heap beyond that map. The +8 is host-specific, and no cross-platform +16 assumption is applied.
4. Wrote an independent temporary decoder using Python struct without importing verify_files.py. All three graph files independently reproduce node-layer counts, degree sums, degree histograms, edge bounds/no-self/no-duplicates/target-level invariants, exact EOF and exact-fit arithmetic.
5. Independently checked all six persisted graph/disk files against original fixture prefix IDs and f32 bytes, requested header identities and retained SHA-256. No query row is inserted. The fixture identity fixes actual input bytes; the model name is provenance from the candidate metadata, not a claim that inference was rerun.
6. Recomputed the source-derived non-neighbor and neighbor residual figures. Results are retained in `analyst-independent-calculations.json`. They agree with REPORT.md.
7. Reviewed the final REPORT.md/README.md interpretation. Build-profile annotation correctly distinguishes bare rustc --print cfg from the optimized release binary. Completion is bounded to the allocator/link-density study and capacity open question 2; open question 1 remains separate.

## Independently recalculated results

All values are decimal bytes. Neighbor residual includes actual reserved neighbor headers/list storage under the stated source-based subtraction; exact-fit is a geometry lower bound.

| n | Released with index | Retained after drop | Mean layer-0 degree | Neighbor residual | Exact-fit neighbor bytes |
|---:|---:|---:|---:|---:|---:|
| 8,192 | 28,105,096 | 16,384 | 26.58654785 | 2,553,472 | 2,219,408 |
| 10,000 | 34,981,544 | 32,768 | 26.54870000 | 3,114,768 | 2,704,488 |
| 100,000 | 342,180,848 | 262,144 | 26.38885000 | 31,148,320 | 26,901,424 |

At 100k this is 311.4832 inferred neighbor-allocation bytes/node versus 269.01424 exact-fit bytes/node. The difference represents representation slack under the source assumptions; persisted degrees alone cannot assign that slack among particular Vec capacities.

## Exact identities

- Engine source: `35758c6ccac0b9e657be8d80e84d1744378df120`
- Fixture SHA-256: `4676911521f13dc6592d102cf3b4fcace85c6b62b3011e9bd57557dfc9c01f2d`
- Binary SHA-256: `086304919cd32dc8a102015d517210cefc8d958a2202083f99f88ef878effab0`
- Compiler: rustc 1.98.1, commit `48a229ceaefd4985c50990b14116b6d856af0985`, aarch64-apple-darwin, LLVM 22.1.8. Host recorded as macOS-27.0-arm64.
- main.rs SHA-256: `de8db76c0140f2266a432bd6b50d143b8dcbd9649a01433e7eb877913a3a4491`
- Cargo.toml SHA-256: `753f731e7fc0b9da4b1329ad08f77950f334a3914d98128c39b7265cec6c8538`
- Cargo.lock SHA-256: `c325fe51eb13103fe2ba53934245fc615e4a04a2b5e5bd181fd82b9a3469fa58`
- verify_files.py SHA-256: `ade34d0da1d0e892cfb2e1436b70fb47c880caa72978860085e4986c0e68fc5b`
- run_probe.py SHA-256: `29e7f8d0d11d02d678641b033f0868323fe93d1aafcf031b1f9e336a2780bd2c`
- disk-8192.vndb SHA-256: `f5c405fad5222bcb01d18af12767b49bf4411e0835c1e5e824f19bd9c43871e2`
- approx-8192.vndb SHA-256: `313e5c28226729d9e825fe4f3c54f8a02348104e04c0223dc161749be6332c6f`
- disk-10000.vndb SHA-256: `3ede6854ceae452f89de777336def71e55d92a791cc33704d6c772e7dd130419`
- approx-10000.vndb SHA-256: `12f3b2ad3036ca8573e1cecdcb4b6dfd6b4713812461a63e9aab0f4fac81d76f`
- disk-100000.vndb SHA-256: `704dcace49bf7a0ae7f805efb773e20f4f94ab82e6ac777597b4d7629103edcd`
- approx-100000.vndb SHA-256: `e0af799fcf87d13459987fb7bcc165f8957b134c030cddc72937bd83d799db52`

Outcome: PASS for the completed, pinned requested-heap and graph-geometry study, subject to the stated interpretation boundaries. This review provides no merge or release approval.
