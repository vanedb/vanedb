# RFC 0004: Filtered search

- Status: implemented (0.2.0, unreleased)
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
    /// Accept only these ids. Sorted, deduplicated; search validates them.
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

- `ApproxIndex::search_with` keeps the routing beam independent of the filter
  and offers scored, live, matching candidates to a separate bounded result
  heap. Excluded nodes remain traversable. Filtering does not guarantee the
  same recall as unfiltered search; measure recall for the accepted subset.
- **Automatic widening.** If the walk ends with fewer than `k` accepted results
  and fewer than `max_ef_search` nodes visited, the search restarts with the
  beam doubled, up to `max_ef_search`. Effective widths are at least `k` and
  capped by the number of stored nodes; arithmetic saturates. The cap bounds
  beam width, not total visited nodes or distance evaluations: one expansion
  may score a whole neighbor list. Exhausting the cap can return fewer than
  `k` matches even when more exist. The result carries `SearchResult`s only;
  the number of restarts is not exposed in 0.2.0 (the `#[non_exhaustive]`
  struct allows adding it later).
- `FlatIndex::search_with` and `DiskIndex::search_with` take the same
  `SearchParams` and apply the filter inside the streaming top-k, so an exact
  filtered search costs one scan.
- `Allow` and `Deny` slices are checked by binary search; they are the fast
  path for callers that already computed a candidate set (a SQL query, a
  metadata index).

### Python

`ApproxIndex.search(query, k, *, ef_search=None, filter=None, allow_ids=None, deny_ids=None, max_ef_search=None)`.
Exact indexes accept the three filter keywords without the beam options.
`filter` is a callable `int -> bool`; the binding re-acquires the GIL per
call, so the docstring states that `allow_ids` / `deny_ids` (numpy `uint64`
arrays or sequences) are the fast path. Passing both `filter` and an id list
is a `ValueError`, as is combining allow and deny lists. Callback and truth-value
conversion exceptions propagate unchanged; partial results are discarded.

### WebAssembly

`ApproxIndex.search(query, k, efSearchOrOptions?)` preserves the existing
numeric third argument and also accepts
`{ efSearch?, maxEfSearch?, allow?, deny?, predicate? }`.
`FlatIndex.search(query, k, options?)` accepts the three filter fields.
Allow/deny lists are JavaScript arrays of lossless IDs or `BigUint64Array`; use
`BigUint64Array` for the full `u64` range. Combining filters is an error.
Predicate exceptions propagate unchanged and discard partial results.
A JavaScript predicate crosses the boundary per candidate; the README states
the cost.

### C ABI

New functions, existing ones unchanged (RFC 0002 rule):

```c
typedef bool (*vanedb_rs_filter_fn)(uint64_t id, void *user_data);
size_t vanedb_rs_index_search_filtered(vanedb_rs_index *, const float *query,
    size_t k, size_t ef_search, vanedb_rs_filter_fn filter, void *user_data,
    const uint64_t *allow, size_t allow_len, const uint64_t *deny, size_t deny_len,
    uint64_t *ids, float *distances);
```

plus `_store_` and `_disk_` variants. Null callback and list pointers select
unfiltered search. A non-null list pointer selects that filter even at length
zero: an empty allow list matches nothing, an empty deny list matches all.
Null list pointers require zero lengths; at most one filter may be supplied.
Invalid combinations fail without writing output buffers. The C API uses the
default four-times beam cap; it does not expose a separate maximum parameter.
A Rust panic from a `C-unwind` callback is caught at the boundary and reported
as `VANEDB_RS_PANIC`; foreign exceptions and `longjmp` must not cross the callback.

### Errors

An unsorted or duplicated `Allow`/`Deny` slice is `VaneError::Validation`.
Valid filters on an empty index return no matches without invoking a predicate.
Malformed filters are rejected on empty indexes too, so validation does not
depend on whether data has been inserted.

Predicates must remain deterministic for a query and consult external metadata.
They run under the searched index's read lock and must not call methods on
that same index (including from another thread they wait for). Nested searches
on a different index are supported. ID lists avoid callback overhead and these
reentrancy constraints.

## Decisions recorded

- 2026-09-13: automatic beam widening with the 4 × `ef_search` default cap
  accepted (decision 4).
- 2026-09-13: a Python callable predicate is allowed, documented as the slow
  path, with id lists as the fast path (decision 5).
- 2026-09-20 review: specify empty C lists by pointer presence, reject conflicting
  filters consistently, propagate binding callback errors, and document beam
  and callback limits. Preserve the existing WebAssembly third argument.

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
  before; unfiltered search does not allocate the filtered routing heap.
  Check the existing criterion search benches in interleaved runs before
  claiming that its performance is unchanged.
- No file-format change. Graph construction unchanged, so the HNSW invariants
  and cross-engine conformance in `AGENTS.md` are untouched.
- The C ABI adds functions; `VANEDB_RS_ABI_VERSION` is not bumped.

## Acceptance criteria

- [x] `Filter` and `SearchParams::filter` / `max_ef_search` in Rust with
      docs and examples.
- [x] `tests/search_correctness.rs` extended: filtered exact search equals the
      `f64` reference restricted to the allowed set, for all three metrics and
      the existing twelve dimensions.
- [x] Graph test: recall@10 of filtered search at 50%, 10% and 1% selectivity
      measured against exact filtered search; the widening cap documented with
      the measured recall at each selectivity.
- [x] Tombstoned entries are never returned regardless of filter.
- [x] Python, WebAssembly and C ABI surfaces with tests; the ctypes example
      gains a filtered call.
- [x] Interleaved A-B-A bench shows no regression on unfiltered
      `index_search`, `store_search`, `disk_search` (within noise floor).
- [x] README replaces the "over-fetch and filter client-side" advice.
- [x] `CHANGELOG.md` entry.

## Evidence required before the claim

Recall-versus-selectivity table from a dedicated machine on the embedding
fixture of RFC 0003, not uniform random vectors.

The [validation record](../release/0.2.0-filtered-search-validation.md) contains
the pinned real-embedding fixture, default and tuned recall tables, reproduction
commands, regression-test coverage, and unfiltered benchmark evidence.

## Out of scope

Payload storage and filtering on stored metadata (RFC 0009). Range search by
distance threshold, which fits the same `SearchParams` later.
