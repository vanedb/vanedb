# Closing #198 — maintainer closeout

Tip harness + publish gates are WIP on PR #212. **Do not close #198** until
every box below has evidence.

## 1. Publish fixture (AC2) — **done on tip**

Release asset:
`https://github.com/vanedb/vanedb/releases/download/compare-fixture-v1/embeddings.vnef`

Pinned sha256 `4676911521f13dc6592d102cf3b4fcace85c6b62b3011e9bd57557dfc9c01f2d`
in `bench/compare/fixtures/SHA256SUMS`. Metadata committed. Fetch verify:

```bash
VANEDB_COMPARE_FIXTURE_URL=https://github.com/vanedb/vanedb/releases/download/compare-fixture-v1/embeddings.vnef \
  bash bench/compare/scripts/fetch_fixture.sh
```

Rebuild `compare` after pulling this tip so the compile-time SUMS pin matches
before dedicated HW `--markdown` runs.

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

Follow [`0003-demo-update-checklist.md`](0003-demo-update-checklist.md). Apply
[`0003-obsidian-vane-search-0.2.0.patch`](0003-obsidian-vane-search-0.2.0.patch)
in the demo repo (this agent has no write access there), cut the `0.2.0`
release, and reply on #198 / #212 with the release URL.

## 4. Launch (AC6)

Only after §2 and §3: remove the do-not-publish banner from
[`0003-competitor-benchmark.md`](0003-competitor-benchmark.md) and switch
provisional tense to past tense with the real numbers.

## 5. Gate

Required CI Gate green on the tip that contains the filled COMPARISON + pinned
fixture sums.
