# Recorded runs

JSON reports from dedicated-hardware runs go here. Name them
`{hardware_label}-{metric}-{YYYYMMDD}.json`.

**Evidence rule for #198:** before pasting into
[`../../COMPARISON.md`](../../COMPARISON.md), keep the matching JSON in this
directory (or attach it on the issue/PR). Files here are gitignored except
this README — commit them with `git add -f bench/compare/runs/*.json` (the
self-hosted fill workflow and `maintainer_fill_host_comparison.sh` tip do
this). Re-render only via
`scripts/render_comparison_md.py`, which refuses forged params, incomplete
engine sets, fixture hashes not listed in `fixtures/SHA256SUMS`, JSON with
`shared_runner=true`, JSON without `dedicated_attested=true`, and JSON whose
`host_*` facts contradict the HW label.

Record publish runs with the helper (preferred):

```bash
bash bench/compare/scripts/record_publish_run.sh linux-avx2 cosine
```

Or the raw form (host binds are still enforced by `--markdown`):

```bash
VANEDB_COMPARE_HW=linux-avx2 VANEDB_COMPARE_DEDICATED=1 \
  cargo run --release --locked --manifest-path bench/compare/Cargo.toml -- run \
  --fixture bench/compare/fixtures/embeddings.vnef --rounds 4 --markdown \
  --json-out runs/linux-avx2-cosine-$(date +%Y%m%d).json
```

Raw timings from CI or shared cloud runners must not be committed or pasted.
Android: use [`../ANDROID.md`](../ANDROID.md), not this helper.
