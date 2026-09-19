# vanedb-compare

Competitor benchmark harness for RFC 0003 / #198.

Requires a Rust toolchain (MSRV 1.85+) and a C++17 compiler (`g++` preferred).

## One command (from the repository root)

Publish paste path (dedicated idle host only; after `embeddings.vnef` is
fetched **and** listed in `fixtures/SHA256SUMS`):

```bash
# Prefer the helper (Apple/Linux):
bash bench/compare/scripts/record_publish_run.sh linux-avx2 cosine

# Or equivalent:
VANEDB_COMPARE_HW=linux-avx2 VANEDB_COMPARE_DEDICATED=1 \
  cargo run --release --locked --manifest-path bench/compare/Cargo.toml -- run \
  --fixture bench/compare/fixtures/embeddings.vnef \
  --rounds 4 \
  --markdown \
  --json-out /tmp/compare-$(hostname).json
```

`--markdown` requires `VANEDB_COMPARE_HW`, `VANEDB_COMPARE_DEDICATED=1`,
dim=768, and the in-repo `embeddings.vnef` SUMS pin. Shared CI/cloud runners
(and Cursor cloud FS markers) are refused; GitHub Actions *self-hosted*
runners without `/opt/hostedtoolcache` are allowed. Android: see
[`ANDROID.md`](ANDROID.md). One-shot host fill:
`maintainer_fill_host_comparison.sh`, or Actions → **Fill COMPARISON
(self-hosted)**.

Smoke / CI only (never paste into COMPARISON.md):

```bash
cargo run --release --locked --manifest-path bench/compare/Cargo.toml -- run \
  --fixture bench/compare/fixtures/smoke.vnef --allow-smoke --rounds 1 --max-queries 4
```

Useful flags: `--engine vanedb --engine usearch`, `--metric cosine|l2`,
`--ef 16,32,50,100`, `--rounds 4`, `--markdown`.

See [`../COMPARISON.md`](../COMPARISON.md) for methodology, caveats, and where
to paste dedicated-hardware results. See `fixtures/README.md` for the embedding
fixture contract. Android notes: [`ANDROID.md`](ANDROID.md).
