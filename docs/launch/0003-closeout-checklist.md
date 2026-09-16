# Closing #198 — maintainer closeout

Tip harness + publish gates are WIP on PR #212. **Do not close #198** until
every box below has evidence.

## 1. Publish fixture (AC2)

```bash
# If /tmp/vnef-full already finished on an agent host:
bash bench/compare/scripts/finalize_fixture.sh /tmp/vnef-full
UPLOAD_RELEASE=1 COMPARE_FIXTURE_TAG=compare-fixture-v1 \
  bash bench/compare/scripts/finalize_fixture.sh /tmp/vnef-full

git add bench/compare/fixtures/metadata.json bench/compare/fixtures/SHA256SUMS
# do NOT add embeddings.vnef
```

Update `bench/COMPARISON.md` / `fixtures/README.md` fetch URL to:

`https://github.com/vanedb/vanedb/releases/download/compare-fixture-v1/embeddings.vnef`

Verify:

```bash
VANEDB_COMPARE_FIXTURE_URL=https://github.com/vanedb/vanedb/releases/download/compare-fixture-v1/embeddings.vnef \
  bash bench/compare/scripts/fetch_fixture.sh
```

## 2. Dedicated hardware tables (AC3)

On each of **Apple Silicon**, **Linux AVX2**, and **Android ARM64** (device
preferred; emulator only if labelled), with the machine idle:

```bash
bash bench/compare/scripts/record_publish_run.sh linux-avx2 cosine
bash bench/compare/scripts/record_publish_run.sh linux-avx2 l2
# apple-m4-pro / android-arm64-device likewise
```

Paste only the harness `--markdown` output (or
`render_comparison_md.py` on that JSON) under the matching heading in
`bench/COMPARISON.md`. Keep the JSON under `bench/compare/runs/` or attach it
on the PR. **Never** paste cloud/CI timings.

## 3. Demo (AC5)

Follow [`0003-demo-update-checklist.md`](0003-demo-update-checklist.md). Reply
on #198 / #212 with the demo release URL.

## 4. Launch (AC6)

Only after §2 and §3: remove the do-not-publish banner from
[`0003-competitor-benchmark.md`](0003-competitor-benchmark.md) and switch
provisional tense to past tense with the real numbers.

## 5. Gate

Required CI Gate green on the tip that contains the filled COMPARISON + pinned
fixture sums.
