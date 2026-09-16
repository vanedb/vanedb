# Competitor comparison (RFC 0003 / #198)

VaneDB against the in-process libraries a developer actually shortlists:
USearch, hnswlib, instant-distance, hnsw_rs, and sqlite-vec. This is a
**separate arm** from the C++-vs-Rust harness in [`../`](../); that harness
stays conformance-adjacent. Numbers here are only meaningful on dedicated
hardware with interleaved rounds — never from CI or a shared cloud runner
([`AGENTS.md`](../../AGENTS.md) performance rules).

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
   `fixtures/SHA256SUMS`. Passages are truncated to `--max-chars` at generation
   time (default 1500–2000; see `fixtures/metadata.json`). The smoke fixture is
   for harness checks only and must never appear in a published table.
4. **Parameter fairness.** Shared `M=16`, `ef_construction=200`, seed `42`
   where the engine exposes them. `instant-distance` hard-codes `M=32` and
   ignores per-query ef (one row at construction `ef_search` only). USearch
   and `hnsw_rs` do not expose construction RNG seeds — recorded in row notes.
   sqlite-vec is brute force (no ANN); it is the "you already ship SQLite"
   baseline, not an HNSW peer. Cosine publish runs **omit** sqlite-vec by
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

One command (from the **repository root**):

```bash
VANEDB_COMPARE_HW=linux-avx2 cargo run --release --locked \
  --manifest-path bench/compare/Cargo.toml -- run \
  --fixture bench/compare/fixtures/embeddings.vnef \
  --rounds 4 \
  --markdown \
  --json-out runs/$(hostname)-$(date +%Y%m%d).json
```

`--markdown` refuses smoke/dev fixtures, unsigned files, unset
`VANEDB_COMPARE_HW`, `--rounds < 2`, shrunk `--max-queries`, and
`--force-sqlite-vec-cosine`. Smoke fixtures require `--allow-smoke` and cannot
produce publishable markdown (classification is by `n_docs`/`n_queries` +
metadata, not filename).


Generate the full fixture once (not in CI):

```bash
python3 bench/compare/scripts/generate_fixture.py --backend fastembed \
  --out-dir bench/compare/fixtures
# host embeddings.vnef as a release asset; commit metadata.json + SHA256SUMS
```

Host the resulting `embeddings.vnef` as a GitHub Release asset (too large for
git), add its sha256 to `fixtures/SHA256SUMS`, then consumers fetch with:

```bash
VANEDB_COMPARE_FIXTURE_URL=https://github.com/vanedb/vanedb/releases/download/<tag>/embeddings.vnef \
  bash bench/compare/scripts/fetch_fixture.sh
```



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
Paste the harness `--markdown` output (or `render_comparison_md.py`) under the
matching heading; do not mix metrics in one table.

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
