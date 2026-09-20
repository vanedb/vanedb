# VaneDB roadmap

The 0.1.1 scope is the Rust engine and its Python, C and WebAssembly bindings,
with CPU search. The experimental macOS `gpu-metal` feature exposes distance
scans, accelerates no index and no binding exposes it (#208);
finish-versus-delete is decided after 0.3.0 (#257). The C++ engine remains
frozen reference code.

[`LIMITS.md`](LIMITS.md) records what 0.1.1 can hold and what it does not do.

This page is an index. Designs live in [`docs/rfcs/`](rfcs/README.md), work
lives in GitHub issues assigned to a milestone, and each release's evidence
lives in [`docs/release/`](release/). [`MARKET_ANALYSIS.md`](MARKET_ANALYSIS.md)
(2026-09-13) is the analysis behind the ordering below; it is analysis, not a
decision, until the corresponding RFC is accepted.

## Order of work

| Order | RFC | Milestone | Status |
|---|---|---|---|
| 0 | [0003 Competitor benchmark and demo](rfcs/0003-competitor-benchmark-and-demo.md) | 0.2.0 | accepted |
| 1 | [0004 Filtered search](rfcs/0004-filtered-search.md) | 0.2.0 | implemented |
| 2 | [0006 WebAssembly persistence](rfcs/0006-wasm-persistence.md) | 0.2.0 | implemented |
| 3 | [0002 C ABI distribution](rfcs/0002-c-abi-distribution.md), stages 1 and 5 | 0.2.0 | accepted |
| 4 | [0010 Write-path gaps](rfcs/0010-write-path-gaps.md) | 0.2.0 | accepted |
| 4 | [0011 API vocabulary before 1.0](rfcs/0011-api-vocabulary-before-1-0.md) | 0.2.0 | accepted |
| 4 | [0012 Platform support policy](rfcs/0012-platform-support-policy.md) | 0.2.0 | accepted |
| 5 | [0013 VNDB v3 container format](rfcs/0013-vndb-v3-container.md) | 0.3.0 | accepted |
| 5 | [0005 Quantized storage](rfcs/0005-quantized-storage.md) | 0.3.0 | accepted |
| 6 | [0002 C ABI distribution](rfcs/0002-c-abi-distribution.md), stages 2 to 4 | 0.3.0 | accepted |
| 7 | [0007 Mobile SDKs](rfcs/0007-mobile-sdks.md) | 0.3.0 | accepted |
| 8 | [0008 Streaming disk build and mapped graph](rfcs/0008-streaming-disk-build-and-mapped-graph.md) | 0.4.0 | draft; direction accepted, gated on the [capacity study](research/capacity.md) |
| 9 | [0009 Optional payload column](rfcs/0009-payload-column.md) | 0.4.0 | accepted |
| after the above | [0001 CUDA after the initial release](rfcs/0001-cuda-after-initial-release.md) | none assigned | accepted; ordering amendment accepted 2026-09-13 |

Milestones: [0.2.0](https://github.com/vanedb/vanedb/milestone/1),
[0.3.0](https://github.com/vanedb/vanedb/milestone/2),
[0.4.0](https://github.com/vanedb/vanedb/milestone/3).

## CUDA on NVIDIA GPUs

Required after the initial release (decision of 2026-09-07), scheduled after
RFCs 0002 to 0007 (amendment of 2026-09-13), no version or date assigned, and
not claimable without the hardware, correctness, lifecycle and performance
evidence listed in [RFC 0001](rfcs/0001-cuda-after-initial-release.md). That
RFC is the only place the requirements are written.

## Mobile physical-device verification

Simulator and emulator runs are the accepted evidence for 0.1.x. Physical
iPhone and Android runs are an acceptance criterion of
[RFC 0007](rfcs/0007-mobile-sdks.md) and must be recorded before any
device-support claim.

## Decided for 0.2: `get` returns `None` instead of raising

Recorded with its precedent survey in
[RFC 0011](rfcs/0011-api-vocabulary-before-1-0.md), which also settles the
count spelling, `Metric` ergonomics and the `ef_search` shape.
