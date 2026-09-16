# Embedding fixtures

| File | Role |
|---|---|
| `smoke.vnef` | Deterministic harness smoke (768-d, 256 docs, 16 queries). **Not for publication.** |
| `embeddings.vnef` | **Not in git yet.** Target: real `nomic-embed-text-v1.5` 100k×768 + 1k queries with nomic task prefixes over BeIR/nq (pinned revision). Generate once, host as a release asset, never in CI. See [`docs/launch/0003-fixture-hosting.md`](../../../docs/launch/0003-fixture-hosting.md). |
| `metadata.json` | Placeholder until generation finishes (`n_docs`/`n_queries` = 0, notes say pending). Replaced by the generator output when the publish fixture is finalized. |
| `SHA256SUMS` | Today pins `smoke.vnef` only. The `embeddings.vnef` line is added when the publish fixture is hosted. `--markdown` / `render_comparison_md.py` fail closed until that line exists in **this checkout** (beside-file SUMS alone is not enough; dim must be 768). |

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
  --out-dir /tmp/vnef-full --text-batch 64 --max-chars 1500
bash bench/compare/scripts/finalize_fixture.sh /tmp/vnef-full
# then host embeddings.vnef; commit metadata.json + SHA256SUMS only
```

Paste a dedicated-hardware JSON into COMPARISON.md with:

```bash
# JSON must have been recorded with VANEDB_COMPARE_DEDICATED=1 on that machine
# (shared_runner=false, dedicated_attested=true). Cloud/CI JSON is refused.
python3 bench/compare/scripts/render_comparison_md.py path/to/report.json
```
