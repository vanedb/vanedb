# Closing #198 — maintainer closeout

Tip harness + publish gates are WIP on PR #212. **Do not close #198** until
every box below has evidence.

## 1. Publish fixture (AC2)

Full runbook: [`0003-fixture-hosting.md`](0003-fixture-hosting.md).

```bash
# Generate (once; needs ≥16 GiB RAM; streaming generator):
python3 bench/compare/scripts/generate_fixture.py --backend fastembed \
  --out-dir /tmp/vnef-full --text-batch 64 --max-chars 1500

# Or reuse a finished agent tree under /tmp/vnef-full, then:
bash bench/compare/scripts/finalize_fixture.sh /tmp/vnef-full
UPLOAD_RELEASE=1 COMPARE_FIXTURE_TAG=compare-fixture-v1 \
  bash bench/compare/scripts/finalize_fixture.sh /tmp/vnef-full

git add bench/compare/fixtures/metadata.json bench/compare/fixtures/SHA256SUMS
# do NOT add embeddings.vnef
```

`--markdown` / `render_comparison_md.py` fail closed until
`fixtures/SHA256SUMS` lists `embeddings.vnef` in this checkout (Round-8 repo
pin, dim must be 768). Finalize + commit the pin **before** dedicated HW runs.

After the release asset exists, set the fetch URL in `bench/COMPARISON.md` /
`fixtures/README.md` to:

`https://github.com/vanedb/vanedb/releases/download/compare-fixture-v1/embeddings.vnef`

Verify:

```bash
VANEDB_COMPARE_FIXTURE_URL=https://github.com/vanedb/vanedb/releases/download/compare-fixture-v1/embeddings.vnef \
  bash bench/compare/scripts/fetch_fixture.sh
```

## 2. Dedicated hardware tables (AC3)

### Apple Silicon / Linux AVX2 (host binary)

Idle dedicated machine only:

```bash
bash bench/compare/scripts/record_publish_run.sh linux-avx2 cosine
bash bench/compare/scripts/record_publish_run.sh linux-avx2 l2
bash bench/compare/scripts/record_publish_run.sh apple-m4-pro cosine
bash bench/compare/scripts/record_publish_run.sh apple-m4-pro l2
```

The helper refuses cloud/CI shells, refuses `apple-*` off Darwin arm64, refuses
`linux-avx2*` without a readable AVX2 `cpuinfo`, and **refuses `android-*`**
(see below). The harness `--markdown` path applies the same host binds
(OS/arch/AVX2/Android filesystem), so skipping the helper does not reopen
cross-class labels.

### Android ARM64

Follow [`../../bench/compare/ANDROID.md`](../../bench/compare/ANDROID.md) (NDK/adb on
device, or labelled emulator). Do **not** use `record_publish_run.sh` with an
`android-*` label on a laptop/server — that would mislabel host timings.

Paste only harness `--markdown` output (or `render_comparison_md.py` on that
JSON) under the matching heading in `bench/COMPARISON.md`. Keep JSON under
`bench/compare/runs/` or attach on the PR. **Never** paste cloud/CI timings.

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
