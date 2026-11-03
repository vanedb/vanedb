# Complete November 3 interleaved attempt

All 1,418 raw files are in `raw-results.tar.gz`; `raw-file-index.json` records
individual sizes and SHA-256 hashes. `SHA256SUMS` covers every public file except
itself. The archive includes every Criterion round, stdout/stderr, five-second
host sample, preflight, lifecycle record and controller diff. Nothing was
selected or dropped based on favorable timings. No Android campaign ran here;
Android scripts in the archive were prepared but not executed.

`attempt-provenance.json` preserves the earlier controller setup hashes;
`post-run-validation.json` records the final executed controller hashes.
`run_native.py.diff` and `pause_indexer.py.diff` in the archive explain the final
changes from November 2: output paths/reservation, explicit controller PID and
fresh indexer identity discovery. Shared-host checks and thresholds are unchanged.
`corespotlightd` restoration is recorded in `criterion-indexer-pause.json` and
verified by post-run process inspection. The user-owned iOS simulator remained
running. Do not interpret emulator or shared-runner timings as native Linux AVX2.

From this directory:

```sh
shasum -a 256 -c SHA256SUMS
python3 verify_evidence.py
mkdir /tmp/vanedb-interleaved-recheck
tar -xzf raw-results.tar.gz -C /tmp/vanedb-interleaved-recheck
python3 /tmp/vanedb-interleaved-recheck/analyze_criterion.py
cmp criterion-analysis.json /tmp/vanedb-interleaved-recheck/criterion-analysis.json
```

Use a new empty extraction directory. Analysis reproduces the numbers without
running any benchmarks. The launch, finalization and packaging scripts contain
original local paths and are retained as provenance; do not blindly rerun them.
`package_attempt.py` rejects replacing an existing public evidence directory.
The report does not claim a qualifying performance pass.
