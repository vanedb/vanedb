# November 3 release-readiness evidence

Read the [report and remaining gates](../../0.2.0-readiness-progress-2026-11-03.md).
Original candidate `e1761dcf4258011b446a361dc529da082ea2fc94` and wasm correction
`6e4fc2d88f2f248cb06becb24bcc0ef228323d22` are kept distinct. No artifact or
functional CI result is promoted to dedicated-hardware performance evidence.

- `dispatches.json` and `fix-remote-runs.json`: commands and source identities.
- `run-*.json`: exact-source workflow outcomes and step statuses; publication jobs skipped.
- `artifacts-*.json`, npm artifact metadata and `download-verification.json`:
  GitHub artifact identities, ZIP digests and downloaded payload checksums.
- `artifact-inventory-validation.json`, `capi-verification.json`: 28 wheels,
  one sdist, package checksum manifests, five-platform C inventory and signatures.
- `cross-realm-red.json`, `fix-validation.json`, `fix-final-validation.json`,
  `fixed-remote-npm-verification.json`: original failure and local/remote fix checks.
- `fix-source.json`, `fix-file-hashes.json`, `fix.patch`: the focused fix provenance.
- `tool-versions.json`, `demo-status.json`, `runner-inventory.json`: actual tooling,
  unchanged accepted demo and absence of a registered dedicated runner.

The raw archive contains local stdout/stderr, remote workflow logs, the independent
review's minimal reproducers and their outcomes. The review's `extra.mjs` test is
retained as `review-reproducer/extra-fixed-checks.mjs`. The exact source patch is retained inside the archive with its original
unified-diff whitespace. Full source changes live in PR #359. The technical review does not constitute final multi-role release sign-off.

Verify the public record in a scratch copy, using Python 3.11 or newer:

```sh
shasum -a 256 -c SHA256SUMS
python3 verify_archive.py
tar -xzf raw-validation-and-review.tar.gz
python3 verify_record.py
```

To recheck the locally downloaded bytes, keep this evidence directory intact and
pass the download root (containing `artifacts/`, `npm-artifact/` and
`fixed-npm-artifact/`) to:

```sh
python3 verify_downloads.py /path/to/020-release-readiness-20261103T082227Z
```

The exact C Sigstore verification command is retained in `capi-verification.json`.
It uses the candidate's `scripts/capi_release.py verify` with explicit version,
commit and branch ref. The minimal Promise reproducer runs in a temporary npm
consumer with the relevant tarball installed, using Node's
`--unhandled-rejections=strict --disallow-code-generation-from-strings` flags.

Artifact binaries are retained locally and in GitHub workflow artifacts, not
committed here. GitHub artifact retention is finite. This is candidate rehearsal
evidence, not the final stable release's durable artifact distribution or sign-off.
