# Closing #198 — maintainer closeout

PR #212 tip has harness + hosted fixture. Re-check **Required CI Gate** on the
current tip after every push. **Do not close #198** until every box below has
evidence (AC3 HW tables + AC5 demo release still open).

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

## 2. Dedicated hardware tables (AC3) — **blocked for cloud agents**

`bench/COMPARISON.md` still has six `*Pending.*` cells (cosine+L2 × Apple /
Linux AVX2 / Android). This cloud agent has **0** connected Cursor
self-hosted workers and must not paste timings from shared runners
(`AGENTS.md`). Operator-owned **GitHub Actions self-hosted** runners are an
allowed path (see below).

### Apple Silicon / Linux AVX2 (host binary)

Idle dedicated machine only (after `git pull` of the tip with the fixture pin):

```bash
# One-shot for the current Apple Silicon or Linux AVX2 host:
bash bench/compare/scripts/maintainer_fill_host_comparison.sh

# Or per metric:
bash bench/compare/scripts/record_publish_run.sh linux-avx2 cosine
bash bench/compare/scripts/record_publish_run.sh linux-avx2 l2
bash bench/compare/scripts/record_publish_run.sh apple-m4-pro cosine
bash bench/compare/scripts/record_publish_run.sh apple-m4-pro l2
# Both metrics for one label:
bash bench/compare/scripts/record_both_metrics.sh linux-avx2
```

Or Actions → **Fill COMPARISON (self-hosted)** (`fill-comparison-self-hosted.yml`)
on a registered `runs-on: self-hosted` runner (never `ubuntu-latest` /
`macos-latest`). The harness and helpers treat `RUNNER_ENVIRONMENT=self-hosted`
without `/opt/hostedtoolcache` as dedicated; GitHub-hosted images still refuse.

The helper refuses cloud/CI shells (except the self-hosted exception above),
refuses `apple-*` off Darwin arm64, refuses `linux-avx2*` without a readable
AVX2 `cpuinfo`, and **refuses `android-*`** (see below). The harness
`--markdown` path applies the same host binds (OS/arch/AVX2/Android
filesystem), so skipping the helper does not reopen cross-class labels.

### Android ARM64

Follow [`../../bench/compare/ANDROID.md`](../../bench/compare/ANDROID.md) (NDK/adb on
device, or labelled emulator). Do **not** use `record_publish_run.sh` with an
`android-*` label on a laptop/server — that would mislabel host timings.

Prefer the gated slot filler (same policy as `--markdown` /
`render_comparison_md.py`):

```bash
python3 bench/compare/scripts/fill_comparison_slot.py \
  bench/compare/runs/<hw>-cosine-*.json \
  bench/compare/runs/<hw>-l2-*.json
```

Or paste harness `--markdown` output under the matching heading. Keep JSON under
`bench/compare/runs/` or attach on the PR. **Never** paste cloud/CI timings.
`maintainer_fill_host_comparison.sh` runs the filler automatically after both
metrics record.

## 3. Demo (AC5) — **maintainer apply required**

Applyable patch + one-shot are on **main** via
[#219](https://github.com/vanedb/vanedb/pull/219) (`eb5bf4d`):
[`0003-obsidian-vane-search-0.2.0.patch`](0003-obsidian-vane-search-0.2.0.patch)
(verified `git apply` + **98/98** tests + production `main.js` build) and
`docs/launch/maintainer_cut_demo_0.2.0.sh`. This agent cannot push/fork
`vanedb/obsidian-vane-search` (403 on git push and Contents API). One-shot
for a maintainer with write access (from a main checkout):

```bash
bash docs/launch/maintainer_cut_demo_0.2.0.sh
```

Or Actions → **Cut demo 0.2.0** with secret `DEMO_REPO_TOKEN`
(`cut-demo-0.2.0.yml` is on **main** via
[#220](https://github.com/vanedb/vanedb/pull/220)). Local script on main also
works without the secret if you have demo-repo write.

To let a **future** cloud agent cut AC5 itself:
[#215](https://github.com/vanedb/vanedb/pull/215) is **merged** (`ecec707` —
`.cursor/environment.json` `repositoryDependencies` includes
`github.com/vanedb/obsidian-vane-search`). Remaining: install the Cursor
GitHub App on `vanedb/obsidian-vane-search` with `contents:write`, then
**start a new** cloud agent (this run’s token will not pick up App scope)
and execute `bash docs/launch/maintainer_cut_demo_0.2.0.sh`. Until the App
has write, `cursor[bot]` still gets push **403** on the demo repo.

Maintainer request: https://github.com/vanedb/obsidian-vane-search/issues/18 —
run the script / workflow (or apply patch + tag `0.2.0`), reply on #198 / #212
with the release URL.

Staging artifacts (vanedb prerelease, not official demo release):
https://github.com/vanedb/vanedb/releases/tag/demo-0.2.0-staging
Until the official demo tag exists, AC5 stays open.

## 4. Launch (AC6)

Only after §2 and §3: remove the do-not-publish banner from
[`0003-competitor-benchmark.md`](0003-competitor-benchmark.md) and switch
provisional tense to past tense with the real numbers.

## 5. Gate — **Required CI Gate green on tip**

Tip CI (including Required CI Gate and `claude-review`) must stay green after
every COMPARISON paste / harness tip. Re-check the tip SHA before closing
#198. Do **not** close while six `*Pending.*` cells remain or while the
official demo `0.2.0` release URL is missing (staging on vanedb is not AC5).
