# November 2 performance evidence

Read the [acceptance report](../../0.2.0-performance-acceptance-2026-11-02.md)
for qualification. Report completeness/renderer PASS does not override failed
host-load screens. No full competitor or RFC 0010 performance gate passes.

## Inspectable records

- `probe-analysis.json`: all four fixed-graph processes and both paths.
- `criterion-analysis.json`: all six passes / 28 rows and Rust/C++ ratios.
- `index-confirm-analysis.json`: all eight runs, both planned blocks, C++ controls.
- `quality-summary.json`: counts, power, interval separation and indexer restoration.
- `cosine.json`, `l2.json`, `android-cosine.json`, `android-l2.json`: unchanged
  canonical harness JSON; matching `*-rendered.md` files have only excess EOF blank lines removed.
  Original renderer stdout remains in `final-validation.json`.
- `*-campaign.json`: exact commands, hashes, exit status and quality counters.
- `android-setup*.json`, `android-avd-creation.json`, `android-avd-config.ini`,
  `android-emulator-lifecycle.json`: setup failures, equivalent AVD and lifecycle.
- `*-index-files.json` and Android campaign `index-hashes` command outputs:
  checksums of task-generated indices before their removal.
- `*-verification.json`: report completeness and raw-round arithmetic checks.

Build commands, source/fixture/tool versions and the recording overlay are in
[November 1 preparation report](../../0.2.0-performance-acceptance-2026-11-01.md).
The full candidate is `e1761dcf4258011b446a361dc529da082ea2fc94`; baseline is
`51cb2d871538985aad933a13ebe50c13a0031dfa`. The docs commit is not measured code.

## Verify without running benchmarks

Work in a copy of this evidence directory. Python 3.11+ checks use only the standard
library. The archive contains relative regular-file paths, with deterministic
archive metadata and unchanged raw numeric records.

```sh
shasum -a 256 -c SHA256SUMS
python3 verify_archive.py
tar -xzf raw-rounds-and-load.tar.gz
python3 verify_probe.py
python3 analyze_criterion.py
python3 analyze_index.py
python3 summarize_quality.py
python3 verify_compare.py cosine.json compare-cosine.stderr.log /tmp/cosine-verify.json
python3 verify_compare.py l2.json compare-l2.stderr.log /tmp/l2-verify.json
python3 verify_compare.py android-cosine.json android-cosine.stderr.log /tmp/android-cosine-verify.json
python3 verify_compare.py android-l2.json android-l2.stderr.log /tmp/android-l2-verify.json
```

The analysis scripts recompute their JSON in that copy. `verify_archive.py`
checks every member against `archive-index.json`. To repeat renderer checks,
use the candidate checkout's `bench/compare/scripts/render_comparison_md.py`
with each canonical report and its committed fixture pin.

The `run_*`, setup, launch and indexer-control scripts are historical command
records. Their absolute paths and process identities belong to this session;
repeating measurements requires fresh process identities, source/binary/fixture
verification, AC and quiet-host checks. They are not part of the read-only
arithmetic verification above.

All raw host/guest records and logs, including failed **setup attempts**,
are in `raw-rounds-and-load.tar.gz`; no measured round was discarded. The
emulator logs redact only host ADB public-key lines. `archive-index.json`
records the redactions and public member hashes. `local-original-artifacts.sha256`
records original local input hashes before public metadata edits, log
redaction and rendered-table EOF normalization. Those transformed files
intentionally differ from that historical manifest. `SHA256SUMS` verifies the public files.
