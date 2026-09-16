# vanedb-compare

Competitor benchmark harness for RFC 0003 / #198. Builds with:

```bash
cargo build --release --locked --manifest-path bench/compare/Cargo.toml
```

One-command run (defaults to `fixtures/embeddings.vnef`, else smoke):

```bash
cargo run --release --locked --manifest-path bench/compare/Cargo.toml -- run
```

Useful flags: `--engine vanedb --engine usearch`, `--metric cosine|l2`,
`--ef 16,32,50,100`, `--rounds 4`, `--markdown`, `--max-queries 32` (smoke).

See [`../COMPARISON.md`](../COMPARISON.md) for methodology, caveats, and where
to paste dedicated-hardware results. See `fixtures/README.md` for the embedding
fixture contract.
