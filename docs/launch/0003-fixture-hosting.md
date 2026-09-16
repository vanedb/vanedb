# Hosting `embeddings.vnef` (#198)

**Status:** published as prerelease tag `compare-fixture-v1`
(`https://github.com/vanedb/vanedb/releases/download/compare-fixture-v1/embeddings.vnef`).
sha256 `4676911521f13dc6592d102cf3b4fcace85c6b62b3011e9bd57557dfc9c01f2d`.

Consumers:

```bash
VANEDB_COMPARE_FIXTURE_URL=https://github.com/vanedb/vanedb/releases/download/compare-fixture-v1/embeddings.vnef \
  bash bench/compare/scripts/fetch_fixture.sh
```

## Regenerating (maintainers only)

The publish fixture is ~310 MiB and must not be committed. Steps if the pin
must be replaced:

1. Generate on a machine with ≥16 GiB RAM (streaming generator):

   ```bash
   python3 bench/compare/scripts/generate_fixture.py --backend fastembed \
     --out-dir /tmp/vnef-full --text-batch 64 --max-chars 1500
   ```

2. Copy into the tree (do **not** `git add` the `.vnef`):

   ```bash
   cp /tmp/vnef-full/embeddings.vnef bench/compare/fixtures/
   cp /tmp/vnef-full/metadata.json bench/compare/fixtures/
   # merge embeddings.vnef hash into fixtures/SHA256SUMS (keep smoke.vnef line)
   ```

   CI verifies **present** files only (`smoke.vnef` always; `embeddings.vnef`
   only after a local/fetch). Committing the embeddings hash without the blob
   is intentional and safe.
3. Create a prerelease asset (example tag `compare-fixture-v1`):

   ```bash
   gh release create compare-fixture-v1 \
     bench/compare/fixtures/embeddings.vnef \
     --title "Competitor fixture (RFC 0003)" \
     --notes "100k×768 BeIR/nq + nomic-embed-text-v1.5; see fixtures/metadata.json" \
     --prerelease
   ```

4. Point consumers at the asset URL in `bench/COMPARISON.md` /
   `fixtures/README.md`, then:

   ```bash
   VANEDB_COMPARE_FIXTURE_URL=https://github.com/vanedb/vanedb/releases/download/compare-fixture-v1/embeddings.vnef \
     bash bench/compare/scripts/fetch_fixture.sh
   ```

5. Commit only `metadata.json` + `SHA256SUMS` (and docs URL). Keep
   `embeddings.vnef` gitignored.

Helper (after a completed generator run under `/tmp/vnef-full`):

```bash
bash bench/compare/scripts/finalize_fixture.sh /tmp/vnef-full
UPLOAD_RELEASE=1 bash bench/compare/scripts/finalize_fixture.sh /tmp/vnef-full
```

The helper refuses incomplete VNEF bodies and `pending` metadata.
