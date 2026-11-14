# vanedb-compare

Competitor benchmark harness for RFC 0003 / #198.

Requires a Rust toolchain (MSRV 1.85+) and a C++17 compiler (`g++` preferred).

## Apple Silicon dedicated run (one command)

From the repository root, on the dedicated Apple Silicon machine, after the
machine-state checks below. This fetches the fixture, checks its checksum,
rebuilds the harness, measures cosine and squared L2, and pastes both tables.
Smoke timings from any other machine stay out of `bench/COMPARISON.md`.

```bash
bash bench/compare/scripts/maintainer_fill_host_comparison.sh
```

What that command does, in order:

1. **Machine state (RFC 0003 / `AGENTS.md`).** Refuses unless this is Darwin
   arm64, `pmset -g batt` shows `AC Power`, and Low Power Mode is off
   (`pmset -g`, `lowpowermode` not `1`). Stop other work first (browsers,
   builds, sync, indexing); the script cannot see that. It then sets
   `VANEDB_COMPARE_HW=apple-silicon` (override with that env var, still an
   `apple-*` label) and `VANEDB_COMPARE_DEDICATED=1`, and wraps every
   `cargo` invocation in `caffeinate -i` so the Mac does not idle-sleep.
   The run JSON records `git rev-parse --short HEAD`, so check out the
   commit you intend to publish before starting.
2. **Fixture download and checksum.** If
   `bench/compare/fixtures/embeddings.vnef` is missing, downloads
   `https://github.com/vanedb/vanedb/releases/download/compare-fixture-v1/embeddings.vnef`
   and checks the `embeddings.vnef` line in
   `bench/compare/fixtures/SHA256SUMS`:
   `4676911521f13dc6592d102cf3b4fcace85c6b62b3011e9bd57557dfc9c01f2d`.
3. **Build.** `cargo build --release --locked --manifest-path bench/compare/Cargo.toml`,
   then each metric via `cargo run --release --locked` with `--rounds 4`
   and `--markdown`. `--locked` uses `bench/compare/Cargo.lock`. The
   release rebuild is required: `--markdown` bakes `SHA256SUMS` in at
   compile time.
4. **Where the numbers go.** JSON lands in
   `bench/compare/runs/apple-silicon-cosine-*.json` and
   `bench/compare/runs/apple-silicon-l2-*.json`. The helper pastes them
   into [`../COMPARISON.md`](../COMPARISON.md) under
   `### Apple Silicon (dedicated laptop)`, subsections `#### Cosine` and
   `#### Squared L2`. Commit `COMPARISON.md` and those JSON files on a
   branch. Related to #242; leave that issue open (the Android tables are
   still empty).

The same helper on a native Linux AVX2 box fills the deferred Linux section
once that section has `#### Cosine` and `#### Squared L2` headings again.
Per-metric form (same AC-power check on `apple-*` labels):

```bash
bash bench/compare/scripts/record_publish_run.sh apple-silicon cosine
bash bench/compare/scripts/record_publish_run.sh apple-silicon l2
```

`--markdown` requires `VANEDB_COMPARE_HW`, `VANEDB_COMPARE_DEDICATED=1`,
dim=768, and the in-repo `embeddings.vnef` SUMS pin. Shared CI/cloud runners
(and Cursor cloud FS markers) are refused; GitHub Actions *self-hosted*
runners without `/opt/hostedtoolcache` are allowed. Android: see
[`ANDROID.md`](ANDROID.md). Actions → **Fill COMPARISON (self-hosted)** runs
the same helper.

Smoke / CI only (never paste into COMPARISON.md):

```bash
# Linux images whose `c++` is a clang without libstdc++ headers:
#   CC=/usr/bin/gcc CXX=/usr/bin/g++ 
cargo run --release --locked --manifest-path bench/compare/Cargo.toml -- run \
  --fixture bench/compare/fixtures/smoke.vnef --allow-smoke --rounds 1 --max-queries 4
```

Useful flags: `--engine vanedb --engine usearch`, `--metric cosine|l2`,
`--ef 16,32,50,100`, `--rounds 4`, `--markdown`.

See [`../COMPARISON.md`](../COMPARISON.md) for methodology, caveats, and where
to paste dedicated-hardware results. See `fixtures/README.md` for the embedding
fixture contract. Android notes: [`ANDROID.md`](ANDROID.md).
