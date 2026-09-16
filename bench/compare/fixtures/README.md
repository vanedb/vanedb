# Embedding fixtures

| File | Role |
|---|---|
| `smoke.vnef` | Deterministic harness smoke (768-d, 256 docs, 16 queries). **Not for publication.** |
| `embeddings.vnef` | Real `nomic-embed-text-v1.5` 100k×768 + 1k queries with nomic task prefixes over BeIR/nq (pinned revision). Generated once, never in CI. Too large for git — see [`docs/launch/0003-fixture-hosting.md`](../../docs/launch/0003-fixture-hosting.md). |
| `metadata.json` | Model, corpus, generator for `embeddings.vnef`. |
| `SHA256SUMS` | Pins fixture bytes the harness will accept. |

## Format (VNEF v1)

Little-endian:

1. magic `VNEF`
2. `u32` version (=1)
3. `u32` dim
4. `u32` n_docs
5. `u32` n_queries
6. `u32` metric_hint (0=L2, 1=cosine)
7. `u32` reserved
8. `n_docs * dim` × `f32` documents
9. `n_queries * dim` × `f32` queries
10. `n_docs` × `u64` ids

## Generate

```bash
# smoke (also: cargo run -- write-smoke)
cargo run --release --manifest-path bench/compare/Cargo.toml -- write-smoke

# full fixture (offline, once) — see docs/launch/0003-fixture-hosting.md
python3 bench/compare/scripts/generate_fixture.py --backend fastembed \
  --out-dir bench/compare/fixtures --text-batch 64 --max-chars 1500
# then host embeddings.vnef; commit metadata.json + SHA256SUMS only
```

Paste a dedicated-hardware JSON into COMPARISON.md with:

```bash
python3 bench/compare/scripts/render_comparison_md.py path/to/report.json
```
