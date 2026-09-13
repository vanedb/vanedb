# RFC 0010: Write-path gaps

- Status: accepted (2026-09-13)
- Milestone: 0.2.0
- Tracking issues: #77, #109 (milestone 0.2.0)
- Supersedes / superseded by: none

## Problem

The published benchmark snapshot (`bench/README.md`) shows the Rust engine
trailing the frozen C++ engine on `store_add` (1.58x) and `disk_build`
(2.18x), while matching or leading on every search path and kernel. Both gaps
are diagnosed: issue #109 attributes about 40 ns per add to SipHash on the
internal `u64` id maps where C++ uses identity hashing; issue #77 records the
`disk_build` gap after #71 buffered the builder's writes. These are the only
rows where a reader of the snapshot sees Rust lose, and RFC 0003 will publish
the same operations against third parties.

## Decision

Close both gaps with implementation changes that alter no API, format or
graph construction choice, and verify with interleaved A-B-A runs.

## Design

- **Identity hashing for id maps (#109).** Replace the default hasher on the
  `u64 -> slot` and `slot -> u64` maps with an identity or multiplicative
  hasher (`nohash-hasher`-style, implemented in-crate to avoid a dependency).
  Ids are caller-chosen and may be adversarial, so the map must not degrade to
  O(n) on sequential or crafted keys: use a Fibonacci-multiplicative finaliser
  rather than the raw identity, and add a test that inserts 1M sequential ids
  and 1M ids sharing low bits within a time bound relative to random ids.
- **`disk_build` (#77).** Profile the builder with the streaming path of RFC
  0008 in mind: the remaining cost after #71 is expected in per-row validation
  (finite check, duplicate check) and the id-set insert. Vectorise the finite
  check over the batch, and use the same hasher change for the id set.
- **`store_add`.** `FlatIndex` growth strategy and the same id-map change;
  measure before deciding on chunked storage, which `ApproxIndex` already has.

## Alternatives rejected

- **Accept the gap and annotate the rows.** Rejected: the rows are on the
  operations every user runs first, and RFC 0003 will show them beside
  competitors.
- **Change M, reverse-link policy, or insertion order to speed up build.**
  Rejected: `AGENTS.md` fixes those as performance-sensitive conformance
  choices; this RFC does not touch graph construction.

## Compatibility and migration

None visible. Files, results and iteration orders are unchanged; a test
asserts identical saved bytes before and after on the conformance fixtures.

## Acceptance criteria

- [ ] Hasher change with the adversarial-key test.
- [ ] Interleaved A-B-A(-B) runs on a dedicated machine show `store_add` and
      `disk_build` within 1.1x of the C++ engine, or the RFC is amended with
      the measured floor and the reason.
- [ ] No regression on any search row within the noise floor.
- [ ] Saved-bytes identity test on the fixtures.
- [ ] `bench/README.md` snapshot refreshed and the ‡ annotations removed or
      updated; issues #77 and #109 closed by the PRs.

## Evidence required before the claim

Interleaved runs on dedicated hardware, per `AGENTS.md`. Never a CI delta.

## Out of scope

Graph construction changes, parallel build, the streaming builder itself
(RFC 0008).
