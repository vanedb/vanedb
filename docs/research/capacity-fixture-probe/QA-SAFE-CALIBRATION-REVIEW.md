# Independent QA — safe allocator calibration correction

**PASS for the local calibration correction and rerun equivalence. No open actionable QA findings.** This review does not assert that hosted CodeQL has cleared; the next exact PR-head CI must establish that separately. Earlier exact-head documentation approvals do not automatically approve a changed PR head.

The reported `rust/access-invalid-pointer` alert targets the old calibration's explicit raw pointer access. The revised `src/main.rs:62–101` uses safe owned `Vec<u8>` operations instead. This review does not dismiss the alert or claim the old pattern was safe based solely on prior successful execution.

## Exact source and methodological scope

- New source SHA256: `469a9a7ff25ef4405ce54b80aba0affc2f2e731601ff3fa138bfb2d6f8448d08`.
- New release executable SHA256: `1902809276275ac676ffa2a9505561cce3576acacdb7b4e4700609d2f4bc18c3`.
- Engine candidate remains clean `35758c6ccac0b9e657be8d80e84d1744378df120`; fixture remains pinned `4676911521f13dc6592d102cf3b4fcace85c6b62b3011e9bd57557dfc9c01f2d`.
- Independently compared source against `historical-before-safe-calibration/src/main.rs`: all bytes before `fn self_test()` and from `fn fixture(` onward are identical. The counting allocator wrapper, fixture parsing, map calibration, measurement lifecycle and report generation are unchanged. Only the separately dispatched `--self-test` body changed.
- Original campaign `results/` and historical source/executable remain separately retained. Fresh campaign is `results-safe-calibration/`; old data were not relabeled as a run of the new executable.

## Calibration review and independent bounded execution

The self-test reserves 113 bytes in a Vec, observes it with `black_box`, and asserts both actual capacity and requested live-byte delta. It then requests capacity 257, verifies preserved content, truncates/shrinks to capacity 17, verifies contents and byte delta, drops ownership and asserts the original global baseline. A separate zero-initialized Vec is similarly observed, checked and dropped. Successful-path assertions allocate no report strings between snapshots; printing occurs afterward. Simulated failed allocation/reallocation helper calls use null results and an isolated atomic counter, without dereferencing pointers or issuing OOM requests.

The zeroed-Vec check exercises the zeroed allocation path on this recorded compiler: installed Rust 1.98.1 source `/opt/homebrew/Cellar/rust/1.98.1/lib/rustlib/src/rust/library/alloc/src/vec/spec_from_elem.rs:49–51` specializes u8 zero elements to `RawVec::with_capacity_zeroed_in`. This is compiler-specific source evidence, not a promise that all future Vec implementations use the same path.

Independently ran the exact new executable's `--self-test`: PASS. Independently reran `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v test_verify_files.py`: 3/3 PASS. No graph build was launched by QA.

## Independent full-campaign equivalence check

An independent inline Python comparison, without using the implementer's comparison script, verified:

- All nine fresh case records retain exactly the original fields/values apart from PID, output file path and absolute live-counter values. Every heap delta, map calibration, parameter, file length, ownership result and TLS residual is identical.
- Each case's five absolute snapshots (baseline, empty, built, after-save, after-drop) all moved by exactly +17 bytes. This constant offset corresponds to the longer retained output argument; it cancels from every measured difference. It is not hidden measurement drift.
- Streamed SHA256 of each of the six actual saved VNDB files matches its original actual file and both campaigns' recorded file checksum. Every parser field, including graph levels, degree histograms, link counts and geometry, is identical after excluding only the file path. Original graph files were already independently parsed and validated against fixture bytes; byte-identical fresh outputs preserve that validation.
- Both `complete.json` contents match and cover 8192, 10000 and 100000 rows for flat/disk/approx. Nine fresh PIDs are distinct and every case stderr is empty.
- Every fresh metadata source hash and the executable hash independently matches current source/binary; engine HEAD and clean status were checked again.

The 100k approximate values remain build delta 342442992 B, released on index drop 342180848 B, and retained scratch 262144 B. These are requested allocation counts only. The correction introduces no timing, RSS, peak-memory, device-capacity, recall, release-gate or publication claim.

**Final local QA verdict: PASS for source hash `469a9a7ff25ef4405ce54b80aba0affc2f2e731601ff3fa138bfb2d6f8448d08` and fresh `results-safe-calibration/`.** An exact PR-head archive review and hosted CodeQL/required-gate result remain separate follow-up checks.
