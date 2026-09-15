# RFCs

Design records for changes that alter a public API, an ABI, a file format, a
cross-binding contract, or the release process. One file per change, reviewed
as a pull request, kept after the work lands so the reasoning stays with the
code.

Anything smaller (a bug, a nit, a one-function refactor) is an issue, not an
RFC.

## Lifecycle

Every RFC starts with a status line. Allowed values:

| Status | Meaning |
|---|---|
| `draft` | proposed; not yet reviewed |
| `accepted` | decided; work may start |
| `implemented` | shipped; the RFC names the release |
| `parked` | accepted or draft, deliberately not scheduled; the RFC says why |
| `superseded by NNNN` | replaced by a later RFC |
| `rejected` | decided against; the RFC says why |

An RFC that is neither implemented nor parked within two milestones of
acceptance gets a `parked` line and a reason.

## Where the pieces live

- **Decision and design**: here, in `docs/rfcs/NNNN-short-name.md`.
- **Work**: one GitHub issue per deliverable, linking the RFC, carrying the
  acceptance checklist, assigned to a milestone. Closed by the PR that ships it
  with `Closes #N`.
- **Release**: a GitHub milestone per minor version. The milestone page is the
  release plan; `docs/release/<version>-readiness.md` is its evidence.
- **Index**: [`docs/ROADMAP.md`](../ROADMAP.md) lists every RFC with status and
  milestone and states the order of work. It contains no task detail.

## Writing one

Copy [`0000-template.md`](0000-template.md). Keep the sections; delete none.
Say what changes, why the alternatives lose, what an existing user or file has
to do, and how a reviewer can tell the work is done. Evidence requirements are
part of the acceptance criteria, not a footnote: a feature that cannot be
verified on the hardware it claims does not ship with the claim.

## Index

| RFC | Title | Status | Milestone | Issues |
|---|---|---|---|---|
| [0001](0001-cuda-after-initial-release.md) | CUDA after the initial release | accepted; ordering amendment accepted 2026-09-13 | none | #205 |
| [0002](0002-c-abi-distribution.md) | C ABI distribution | accepted | 0.2.0 (stages 1 and 5), 0.3.0 (stages 2 to 4) | #193 #194 #195 #196 #197 |
| [0003](0003-competitor-benchmark-and-demo.md) | Competitor benchmark and demo | accepted | 0.2.0 | #198 |
| [0004](0004-filtered-search.md) | Filtered search | accepted | 0.2.0 | #199 |
| [0005](0005-quantized-storage.md) | Quantized storage | accepted | 0.3.0 | #200 |
| [0006](0006-wasm-persistence.md) | WebAssembly persistence | accepted | 0.2.0 | #201 |
| [0007](0007-mobile-sdks.md) | Mobile SDKs | accepted | 0.3.0 | #202 |
| [0008](0008-streaming-disk-build-and-mapped-graph.md) | Streaming disk build and mapped graph | draft; direction accepted, gated on #210 | 0.4.0 | #203 #210 |
| [0009](0009-payload-column.md) | Optional payload column | accepted | 0.4.0 | #204 |
| [0010](0010-write-path-gaps.md) | Write-path gaps | accepted | 0.2.0 | #77 #109 |
| [0011](0011-api-vocabulary-before-1-0.md) | API vocabulary before 1.0 | accepted | 0.2.0 | #206 |
| [0012](0012-platform-support-policy.md) | Platform support policy | draft | 0.2.0 | #207 |
| [0013](0013-vndb-v3-container.md) | VNDB v3 container format | accepted | 0.3.0 | #209 |
