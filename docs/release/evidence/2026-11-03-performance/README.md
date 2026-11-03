# November 3 interrupted performance attempt

Read the [report and gate checklist](../../0.2.0-performance-acceptance-2026-11-03.md).
The host screen failed after initial quiet preflight. Only baseline targets ran;
there is no complete baseline/candidate comparison or new performance gate PASS.

`summary.json` contains recomputable counts. `criterion-campaign.json` retains
commands, binary hashes and the interrupted controller status. `attempt-stop.json`
records the deliberate SIGINT and freshly checked process identity;
`post-stop-validation.json` records cleanup, clean source worktrees, existing
user work and tool versions. `host-inventory.json` includes AC, hardware and
OrbStack inventory. No applications/services were suspended and no VM started.

The archive retains every partial Criterion artifact, stdout/stderr log,
controller log and host-load sample from this attempt. No redactions or timing
round deletions were applied. Plain JSON, controller sources and their diffs are
retained alongside it. Prepared Android scripts were **not executed**. Absolute
paths and identities are historical records, not instructions to replay PIDs.

Verify in a scratch copy of this directory with Python 3.11 or newer:

```sh
shasum -a 256 -c SHA256SUMS
python3 verify_archive.py
tar -xzf raw-partial-results-and-load.tar.gz
python3 verify_attempt.py
```

The verifier checks archive member hashes and safe relative regular-file paths,
then recomputes CPU/power screens, per-target counters, preflight streaks,
process peaks, retained Criterion sample counts, controller hashes and recorded
cleanup. Verification PASS means the evidence reconciles; acceptance stays blocked.
The interrupted index target has partial data and is never promoted to a full run.

The previous complete campaigns and clean fixed-graph diagnostic remain in the
[November 2 evidence](../2026-11-02-performance/README.md). Build provenance and
the comparison recording overlay remain in the
[November 1 evidence](../../0.2.0-performance-acceptance-2026-11-01.md).
