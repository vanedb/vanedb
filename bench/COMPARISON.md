# Competitor comparison (RFC 0003 / #198)

VaneDB against the in-process libraries a developer actually shortlists:
USearch, hnswlib, instant-distance, hnsw_rs, and sqlite-vec. This is a
**separate arm** from the C++-vs-Rust harness in [`../`](../); that harness
stays conformance-adjacent. Numbers here are only meaningful on dedicated
hardware with interleaved rounds — never from CI or a shared cloud runner
([`AGENTS.md`](../AGENTS.md) performance rules).

## Caveats (read before the tables)

These caveats are written **before** any published numbers, per the RFC and
the existing `bench/README.md` discipline.

1. **Dedicated hardware only.** Idle machine, plugged in, other work stopped.
   Interleaved A-B-C-… rounds (`--rounds ≥ 2`). Report medians and inter-pass
   spread. A single run is not a result. Local noise floor is ~3%; treat
   deltas inside that floor as noise.
2. **No CI / cloud timings.** Shared runners are noisy and contested. Do not
   paste their output into the result tables below.
3. **Real embeddings, fixed fixture.** Published rows use
   `fixtures/embeddings.vnef` (100k × 768-d, 1k queries), checksummed in
   `fixtures/SHA256SUMS` and hosted as prerelease
   [`compare-fixture-v1`](https://github.com/vanedb/vanedb/releases/tag/compare-fixture-v1)
   (see [`docs/launch/0003-fixture-hosting.md`](../docs/launch/0003-fixture-hosting.md)).
   Only `smoke.vnef` is committed; fetch the publish fixture before a real run:
   `bash bench/compare/scripts/fetch_fixture.sh`. Passages are truncated to
   `--max-chars 1500` at generation time (see `fixtures/metadata.json`). The
   smoke fixture is for harness checks only and must never appear in a
   published table.
4. **Parameter fairness.** Shared `M=16`, `ef_construction=200`, seed `42`
   where the engine exposes them. `instant-distance` hard-codes `M=32` and
   ignores per-query ef (one row at construction `ef_search` only). USearch
   and `hnsw_rs` do not expose construction RNG seeds — recorded in row notes.
   VaneDB build uses `add_batch` (same topology as serial add; wall time is
   not a sequential-insert peer — noted in the vanedb row). sqlite-vec is
   brute force (no ANN); it is the "you already ship SQLite"
   baseline, not an HNSW peer. Its **Build (s)** cell is temp-DB create + row
   insert time, not an in-RAM graph construction — read it as ingest cost,
   not as an HNSW peer. sqlite-vec also emits a single ef row (ef unused).
   Cosine publish runs **omit** sqlite-vec by
   default (vec0 has no cosine metric; the harness scan would mislabel); use
   `--metric l2` for native sqlite-vec, or `--force-sqlite-vec-cosine` only
   when you intentionally want the harness-side scan labelled as such.
5. **Recall is against f64 exact search** on the same fixture and metric.
   Graph luck still applies: one construction seed is one observation.
6. **Delete-then-search** is checked only for engines with a public delete
   API (VaneDB, USearch, hnswlib, sqlite-vec). Others report `n/a`.
7. **File size** is whatever that engine's native save path writes. Engines
   without a wired save path report `n/a`.
8. **RSS** is the process resident-set delta during that engine's build in
   this process. Later engines inherit allocator state; treat RSS as
   indicative, not a lab-grade isolate.

## Methodology

| Item | Value |
|---|---|
| Fixture | `nomic-embed-text` family, 768-d, BeIR/nq passages (see `fixtures/metadata.json`) |
| Metrics | cosine (native; sqlite-vec omitted unless forced) and squared L2 (separate run, includes sqlite-vec) |
| k | 10 |
| ef sweep | 16, 32, 50, 100 (instant-distance: construction ef only) |
| Rounds | interleaved, ≥2 on dedicated hardware |
| Engines | vanedb, usearch 2.21.0, hnswlib 0.8.0, instant-distance 0.6.1, hnsw_rs 0.3.4, sqlite-vec 0.1.6 (L2) |

Preferred command (binds HW label to the host OS/arch/CPU):

```bash
# One-shot Apple Silicon / Linux AVX2 (fetch + rebuild + both metrics + slot fill):
bash bench/compare/scripts/maintainer_fill_host_comparison.sh

# Or per metric:
bash bench/compare/scripts/record_publish_run.sh linux-avx2 cosine
bash bench/compare/scripts/record_publish_run.sh linux-avx2 l2
python3 bench/compare/scripts/fill_comparison_slot.py \
  bench/compare/runs/linux-avx2-cosine-*.json \
  bench/compare/runs/linux-avx2-l2-*.json
```

Equivalent raw form (same host binds are enforced inside `--markdown`):

```bash
VANEDB_COMPARE_HW=linux-avx2 VANEDB_COMPARE_DEDICATED=1 cargo run --release --locked \
  --manifest-path bench/compare/Cargo.toml -- run \
  --fixture bench/compare/fixtures/embeddings.vnef \
  --metric cosine \
  --rounds 4 \
  --markdown \
  --json-out bench/compare/runs/linux-avx2-cosine-$(date +%Y%m%d).json
```

`--markdown` refuses smoke/dev fixtures, unsigned files, basename other than
`embeddings.vnef`, fixture dim ≠ 768, any fixture whose sha256 is not the
`embeddings.vnef` line baked from the **in-repo**
`bench/compare/fixtures/SHA256SUMS` at compare compile time (beside-file SUMS
alone is not enough — run `finalize_fixture.sh`, then `git add`/`commit` that
pin and **rebuild** `compare` before pasteable runs), unset
`VANEDB_COMPARE_HW`, HW labels that do not match the current host (`apple-*` ⇒
Darwin aarch64, `linux-avx2*` ⇒ Linux+AVX2, `android-arm64-*` ⇒ on-device
Android), missing `VANEDB_COMPARE_DEDICATED=1`, `--rounds < 2`, non-canonical
`M`/`ef_construction`/`ef`/`k`/`seed`, engine cherry-picks, shrunk
`--max-queries`, `--force-sqlite-vec-cosine`, `--skip-delete`, `--skip-save`,
shared CI/cloud runner envs **and** Cursor cloud filesystem markers
`/opt/cursor` and `/exec-daemon` plus GitHub-hosted `/opt/hostedtoolcache`
(clearing `CURSOR_AGENT` / `CI` / `GITHUB_ACTIONS` is not enough). Exception:
GitHub Actions *self-hosted* runners (`RUNNER_ENVIRONMENT=self-hosted`)
without `/opt/hostedtoolcache` are treated as operator-owned dedicated
hardware (CI vars alone do not refuse). Also refused: a missing
`metadata.json`. Smoke fixtures require `--allow-smoke` and cannot
produce publishable markdown (classification is by `n_docs`/`n_queries` +
metadata, not filename). JSON recorded without `dedicated_attested=true` (or
with `shared_runner=true`), or with host facts that contradict the HW label, is
also refused by `render_comparison_md.py`.


Generate the full fixture once (not in CI):

```bash
python3 bench/compare/scripts/generate_fixture.py --backend fastembed \
  --out-dir bench/compare/fixtures
# host embeddings.vnef as a release asset; commit metadata.json + SHA256SUMS
```

Host the resulting `embeddings.vnef` as a GitHub Release asset (too large for
git), add its sha256 to `fixtures/SHA256SUMS`, then consumers fetch with:

```bash
VANEDB_COMPARE_FIXTURE_URL=https://github.com/vanedb/vanedb/releases/download/compare-fixture-v1/embeddings.vnef \
  bash bench/compare/scripts/fetch_fixture.sh
```

Maintainer closeout (fixture + three HW classes + demo):
[`docs/launch/0003-closeout-checklist.md`](../docs/launch/0003-closeout-checklist.md).
Dedicated-machine helper (Apple/Linux only):
`bench/compare/scripts/record_publish_run.sh`. Android: [`ANDROID.md`](compare/ANDROID.md).



## Engine versions (pin these in the run JSON)

| Engine | Access | Version pinned in harness |
|---|---|---|
| VaneDB | `vanedb` path dep | workspace crate version |
| USearch | `usearch` crate | 2.21.0 |
| hnswlib | vendored C++ + thin FFI | 0.8.0 |
| instant-distance | crate | 0.6.1 (`M=32` fixed) |
| hnsw_rs | crate | 0.3.4 |
| sqlite-vec | vendored amalgamation via rusqlite | 0.1.6 |

## Results

Fill **two** subsections under each hardware class: one for `--metric cosine`
(sqlite-vec omitted) and one for `--metric l2` (includes native sqlite-vec).
Paste the harness `--markdown` output, or run
`fill_comparison_slot.py` on the publish JSON, under the matching heading; do
not mix metrics in one table.

### Apple Silicon (dedicated laptop)

#### Cosine

*Pending.*

#### Squared L2

*Pending.*

### Linux x86-64 AVX2 (dedicated box)

#### Cosine

*Pending.*

#### Squared L2

*Pending.*

### Android ARM64 (device or emulator)

Label device vs emulator in `VANEDB_COMPARE_HW`. Physical device preferred.

#### Cosine

*Pending.*

#### Squared L2

*Pending.*

## Smoke verification (not a result)

The harness builds with `--locked` and runs every engine on
`fixtures/smoke.vnef`. That path exists to catch API breakage. Its timings
and recall are excluded from the tables above by policy.

## Demo and launch

- Demo application: [`obsidian-vane-search`](https://github.com/vanedb/obsidian-vane-search)
  (separate repository). Linked from the top-level README.
- Launch post draft: [`docs/launch/0003-competitor-benchmark.md`](../docs/launch/0003-competitor-benchmark.md)
  — publish only after maintainer review and after the three hardware classes
  above are filled.
