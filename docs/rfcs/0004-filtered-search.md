# RFC 0004: Filtered search

- Status: accepted (2026-09-13)
- Milestone: 0.2.0
- Tracking issue: #199
- Supersedes / superseded by: none

## Problem

Every search returns the nearest stored ids with no way to restrict the
candidate set. Issue #91 named metadata filtering as "the other big gap" and
ranked it second only to delete; the README tells users to over-fetch and
filter client-side. Filtering combined with vector search is the most common
request in every embedded vector project's tracker (usearch #348, sqlite-vec
#165, objectbox-dart #658) and is on every 2026 on-device checklist. Client
side over-fetch degrades silently: with a 1% selective filter, `k = 10` needs
a beam of roughly 1,000 to find ten matches, and the caller has no way to know
how many it missed.

`SearchParams<'a>` already carries an unused lifetime reserved for exactly
this: a borrowed filter.

## Decision

Add a per-query filter to `SearchParams` that restricts which ids may appear
in results. The graph walk still traverses filtered-out nodes (they may be the
only route between live neighbourhoods, as with tombstones), and only result
acceptance is filtered. Exact indexes apply the filter inside the scan. All
four bindings expose it in the same release.

## Design

### Rust

```rust
pub enum Filter<'a> {
    /// Accept an id when the predicate returns true.
    Predicate(&'a (dyn Fn(u64) -> bool + Sync)),
    /// Accept only these ids. Sorted, deduplicated; the constructor checks.
    Allow(&'a [u64]),
    /// Accept every id except these. Sorted, deduplicated.
    Deny(&'a [u64]),
}

impl<'a> SearchParams<'a> {
    pub fn filter(self, filter: Filter<'a>) -> Self;
    /// Upper bound on automatic beam widening; default 4 × ef_search.
    pub fn max_ef_search(self, ef: usize) -> Self;
}
```

- `ApproxIndex::search_with` applies the filter when a candidate is offered to
  the result heap. Traversal is unchanged, so recall for the accepted subset
  matches unfiltered recall at the same effective beam.
- **Automatic widening.** If the walk ends with fewer than `k` accepted results
  and fewer than `max_ef_search` nodes visited, the search restarts with the
  beam doubled, up to `max_ef_search`. The result carries `SearchResult`s only;
  the number of restarts is not exposed in 0.2.0 (the `#[non_exhaustive]`
  struct allows adding it later).
- `FlatIndex::search_with` and `DiskIndex::search_with` take the same
  `SearchParams` and apply the filter inside the streaming top-k, so an exact
  filtered search costs one scan.
- `Allow` and `Deny` slices are checked by binary search; they are the fast
  path for callers that already computed a candidate set (a SQL query, a
  metadata index).

### Python

`search(query, k, *, ef_search=None, filter=None, allow_ids=None, deny_ids=None)`.
`filter` is a callable `int -> bool`; the binding re-acquires the GIL per
call, so the docstring states that `allow_ids` / `deny_ids` (numpy `uint64`
arrays or sequences) are the fast path. Passing both `filter` and an id list
is a `ValueError`.

### WebAssembly

`search(query, k, ef_search?, { allow?: BigUint64Array, deny?: BigUint64Array, predicate?: (id: bigint) => boolean })`.
A JavaScript predicate crosses the boundary per candidate; the README states
the cost.

### C ABI

New functions, existing ones unchanged (RFC 0002 rule):

```c
typedef bool (*vanedb_rs_filter_fn)(uint64_t id, void *user_data);
size_t vanedb_rs_index_search_filtered(const vanedb_rs_index *, const float *query,
    size_t k, size_t ef_search, vanedb_rs_filter_fn filter, void *user_data,
    const uint64_t *allow, size_t allow_len, const uint64_t *deny, size_t deny_len,
    uint64_t *ids, float *distances);
```

plus `_store_` and `_disk_` variants. A null `filter` with empty lists is the
unfiltered search. A panic inside the callback is caught at the boundary and
reported as `VANEDB_RS_PANIC`.

### Errors

An unsorted or duplicated `Allow`/`Deny` slice is `VaneError::Validation`.
A filter is never an error on an empty index.

## Decisions recorded

- 2026-09-13: automatic beam widening with the 4 × `ef_search` default cap
  accepted (decision 4).
- 2026-09-13: a Python callable predicate is allowed, documented as the slow
  path, with id lists as the fast path (decision 5).

## Alternatives rejected

- **Filter during traversal (skip filtered nodes entirely).** Rejected: known
  to disconnect the graph under selective filters; hnswlib and usearch both
  filter at acceptance for this reason.
- **Metadata stored in the index and a query language.** Rejected for 0.2.0:
  a payload column is RFC 0009, and filtering on it is a later step. A
  predicate plus id lists serves every persona in #91 today with the caller's
  own metadata store.
- **No automatic widening.** Rejected: without it a selective filter returns
  fewer than `k` with no signal, which is the client-side over-fetch problem
  restated.

## Compatibility and migration

- Additive. `search` and `search_with` without a filter behave exactly as
  before; the unfiltered path has no new branch in the inner loop (checked
  by the existing criterion benches, interleaved).
- No file-format change. Graph construction unchanged, so the HNSW invariants
  and cross-engine conformance in `AGENTS.md` are untouched.
- The C ABI adds functions; `VANEDB_RS_ABI_VERSION` is not bumped.

## Acceptance criteria

- [ ] `Filter` and `SearchParams::filter` / `max_ef_search` in Rust with
      docs and examples.
- [ ] `tests/search_correctness.rs` extended: filtered exact search equals the
      `f64` reference restricted to the allowed set, for all three metrics and
      the existing twelve dimensions.
- [ ] Graph test: recall@10 of filtered search at 50%, 10% and 1% selectivity
      measured against exact filtered search; the widening cap documented with
      the measured recall at each selectivity.
- [ ] Tombstoned entries are never returned regardless of filter.
- [ ] Python, WebAssembly and C ABI surfaces with tests; the ctypes example
      gains a filtered call.
- [ ] Interleaved A-B-A bench shows no regression on unfiltered
      `index_search`, `store_search`, `disk_search` (within noise floor).
- [ ] README replaces the "over-fetch and filter client-side" advice.
- [ ] `CHANGELOG.md` entry.

## Evidence required before the claim

Recall-versus-selectivity table from a dedicated machine on the embedding
fixture of RFC 0003, not uniform random vectors.

## Out of scope

Payload storage and filtering on stored metadata (RFC 0009). Range search by
distance threshold, which fits the same `SearchParams` later.
