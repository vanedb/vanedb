# vanedb-compare

Competitor benchmark harness for RFC 0003 / #198.

Requires a Rust toolchain (MSRV 1.85+) and a C++17 compiler (`g++` preferred).

## One command (from the repository root)

```bash
cargo run --release --locked --manifest-path bench/compare/Cargo.toml -- run \
  --fixture bench/compare/fixtures/embeddings.vnef \
  --rounds 4 \
  --markdown \
  --json-out /tmp/compare-$(hostname).json
```

Set `VANEDB_COMPARE_HW` to a short label (`apple-m4-pro`, `linux-avx2`,
`android-arm64-emulator`, …) before publishing a row.

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
