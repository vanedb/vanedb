# Closing #198 / #226 — maintainer closeout

Harness + fixture landed on **main** via
[#212](https://github.com/vanedb/vanedb/pull/212) (`bead8c0`). ARM64 proptest
flake fix: [#227](https://github.com/vanedb/vanedb/pull/227) (`9a555bc`,
Required CI Gate green).

**#198 was auto-closed by the #212 merge** while AC3/AC5 were still open.
Residual tracking: [#226](https://github.com/vanedb/vanedb/issues/226). This change
reopens #198 so acceptance stays on the original issue. **Do not close #226 / #198** until
every box below has evidence.

## 1. Publish fixture (AC2) — **done on main**

Release asset:
`https://github.com/vanedb/vanedb/releases/download/compare-fixture-v1/embeddings.vnef`

Pinned sha256 `4676911521f13dc6592d102cf3b4fcace85c6b62b3011e9bd57557dfc9c01f2d`
in `bench/compare/fixtures/SHA256SUMS`. Metadata committed. Fetch verify:

```bash
git checkout main && git pull
VANEDB_COMPARE_FIXTURE_URL=https://github.com/vanedb/vanedb/releases/download/compare-fixture-v1/embeddings.vnef \
  bash bench/compare/scripts/fetch_fixture.sh
```

Rebuild `compare` after pull so the compile-time SUMS pin matches before
dedicated HW `--markdown` runs.

## 2. Dedicated hardware tables (AC3) — **open (6× Pending)**

`bench/COMPARISON.md` still has six `*Pending.*` cells (cosine+L2 × Apple /
Linux AVX2 / Android). Cloud agents have **0** Cursor self-hosted workers and
must not paste timings from shared runners (`AGENTS.md`). Operator-owned
**GitHub Actions self-hosted** runners are allowed (see below).

### Apple Silicon / Linux AVX2 (host binary)

Idle dedicated machine only (from current `main`):

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

Or Actions → **Fill COMPARISON (self-hosted)** (`fill-comparison-self-hosted.yml`
is on the **default branch**) on a registered `runs-on: self-hosted` runner
(never `ubuntu-latest` / `macos-latest`). The harness and helpers treat
`RUNNER_ENVIRONMENT=self-hosted` without `/opt/hostedtoolcache` as dedicated;
GitHub-hosted images still refuse.

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
`bench/compare/runs/` (commit to main or a follow-up PR). **Never** paste
cloud/CI timings. `maintainer_fill_host_comparison.sh` runs the filler
automatically after both metrics record.

## 3. Demo (AC5) — **open (official 0.2.0 missing)**

Applyable patch + one-shot are on **main** via
[#219](https://github.com/vanedb/vanedb/pull/219):
[`0003-obsidian-vane-search-0.2.0.patch`](0003-obsidian-vane-search-0.2.0.patch)
and `docs/launch/maintainer_cut_demo_0.2.0.sh`. Cloud agents without demo-repo
write get push **403**. One-shot for a maintainer with write access:

```bash
bash docs/launch/maintainer_cut_demo_0.2.0.sh
```

Or Actions → **Cut demo 0.2.0** with secret `DEMO_REPO_TOKEN`
(`cut-demo-0.2.0.yml` on **main** via [#220](https://github.com/vanedb/vanedb/pull/220)).

To let a **future** cloud agent cut AC5 itself:
[#215](https://github.com/vanedb/vanedb/pull/215) merged
`repositoryDependencies` for `github.com/vanedb/obsidian-vane-search`. Remaining:
install the Cursor GitHub App on that repo with `contents:write`, then **start a
new** cloud agent (existing run tokens do not pick up App scope).

Maintainer request: https://github.com/vanedb/obsidian-vane-search/issues/18 —
reply on #226 (or reopened #198) with the official release URL.

Staging artifacts (vanedb prerelease, **not** official demo release):
https://github.com/vanedb/vanedb/releases/tag/demo-0.2.0-staging
Until the official demo tag exists, AC5 stays open.

## 4. Launch (AC6)

Only after §2 and §3: remove the do-not-publish banner from
[`0003-competitor-benchmark.md`](0003-competitor-benchmark.md) and switch
provisional tense to past tense with the real numbers.

## 5. Gate — **Required CI Gate green on main**

Re-check **Required CI Gate** on `main` after every COMPARISON paste. Do **not**
close #226 / #198 while six `*Pending.*` cells remain or while the official demo
`0.2.0` release URL is missing (staging on vanedb is not AC5).
