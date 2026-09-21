# Closing #198 / residual tracking — maintainer closeout

Status / benchmark-fill driver:
`bash docs/launch/maintainer_closeout_226.sh` (status or `--fill`).
For the demo, use the reviewed-PR sequence in §3; the historical cut path
applies an older patch and is not the current release candidate.

Harness + fixture landed on **main** via
[#212](https://github.com/vanedb/vanedb/pull/212) (`bead8c0`). ARM64 proptest
flake fix: [#227](https://github.com/vanedb/vanedb/pull/227) (`9a555bc`,
Required CI Gate green).

**#198 was auto-closed by the #212 merge** while AC3/AC5 were still open.
[#226](https://github.com/vanedb/vanedb/issues/226) tracked the residual until
[#238](https://github.com/vanedb/vanedb/pull/238) merged — GitHub's merge
auto-link treated a negated “close #\…” phrase in that PR body as a closing
keyword and marked #226 completed. Active residual issue:
**[#242](https://github.com/vanedb/vanedb/issues/242)**. Maintainers may also
reopen #198 / #226 manually if desired. GitHub has no auto-reopen keyword.

**PR / commit wording:** never write `close` / `closes` / `fix` / `fixes` /
`resolve` / `resolves` next to `#198`, `#226`, or `#242` (including negations
like “does not …” + those verbs + `#N`). Reference issues as “related to #242”
or “vanedb#242” until both boxes below have evidence URLs, then shut the issue
from the GitHub UI (not via merge keywords).

### Evidence matrix (goal / #198)

| Criterion | Status | Where to verify |
|---|---|---|
| Tip Required CI Gate green | **done** | `origin/main` (post-#212 lineage); Gate must stay green after COMPARISON pastes |
| Hostable `embeddings.vnef` + `SHA256SUMS` | **done** | [compare-fixture-v1](https://github.com/vanedb/vanedb/releases/tag/compare-fixture-v1) |
| Six dedicated-HW COMPARISON tables | **open** | `bench/COMPARISON.md` still has six `*Pending.*` — Apple / Linux AVX2 / Android × cosine+L2; never cloud/CI |
| Official Obsidian demo `0.2.0` | **open** | Missing tag; [staging](https://github.com/vanedb/vanedb/releases/tag/demo-0.2.0-staging) ≠ AC5 |
| ≥5 CTO→user→staff→senior→QA→AI PM rounds | **done** | ≥7 rounds posted on [#212](https://github.com/vanedb/vanedb/pull/212) (e.g. Rounds 23–34) |

Do not mark the goal or residual issues done while any row above is **open**.

## 1. Publish fixture (AC2) — **done on main**

Release assets:
`https://github.com/vanedb/vanedb/releases/download/compare-fixture-v1/embeddings.vnef`
and
`https://github.com/vanedb/vanedb/releases/download/compare-fixture-v1/SHA256SUMS`

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
on the **default branch**) on a registered `runs-on: self-hosted` runner
(never `ubuntu-latest` / `macos-latest`). The workflow opens a PR with
`COMPARISON.md` + run JSON (`main` is ruleset-protected — tip push cannot
land). The harness and helpers treat `RUNNER_ENVIRONMENT=self-hosted` without
`/opt/hostedtoolcache` as dedicated; GitHub-hosted images still refuse.

The helper refuses cloud/CI shells (except the self-hosted exception above),
refuses `apple-*` off Darwin arm64, refuses `linux-avx2*` without a readable
AVX2 `cpuinfo`, and **refuses `android-*`** (see below). The harness
`--markdown` path applies the same host binds (OS/arch/AVX2/Android
filesystem), so skipping the helper does not reopen cross-class labels.

### Android ARM64

Follow [`../../bench/compare/ANDROID.md`](../../bench/compare/ANDROID.md) (NDK/adb on
device, or labelled emulator). Pass `--out-dir` under a writable on-device path
(e.g. `/data/local/tmp/compare-out`) — the default build-host
`target/compare-out` is not writable after cross-compile push. Do **not** use
`record_publish_run.sh` with an `android-*` label on a laptop/server — that
would mislabel host timings.

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

The current candidate is [demo PR #20](https://github.com/vanedb/obsidian-vane-search/pull/20).
Follow [the updated demo checklist](0003-demo-update-checklist.md): independent
reviews and CI, real Obsidian/Ollama vault acceptance, then the approved
reviewed merged commit's annotated release tag. Automated tests and the
historical staging artifacts do not prove the real walkthrough or official
release. Record both evidence links on [#242](https://github.com/vanedb/vanedb/issues/242).

## 4. Launch (AC6)

Only after §2 and §3: remove the do-not-publish banner from
[`0003-competitor-benchmark.md`](0003-competitor-benchmark.md) and switch
provisional tense to past tense with the real numbers.

## 5. Gate — **Required CI Gate green on main**

Re-check **Required CI Gate** on `main` after every COMPARISON paste. Leave
[#242](https://github.com/vanedb/vanedb/issues/242) (and #198 / #226 if
reopened) open while six `*Pending.*` cells remain or while the official demo
`0.2.0` release URL is missing (staging on vanedb is not AC5).
