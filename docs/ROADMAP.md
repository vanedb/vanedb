# VaneDB roadmap

The 0.1.1 scope is the Rust engine and its Python, C and WebAssembly bindings,
with CPU search and the existing macOS Metal feature. The C++ engine remains
frozen reference code.

This page is an index. Designs live in [`docs/rfcs/`](rfcs/README.md), work
lives in GitHub issues assigned to a milestone, and each release's evidence
lives in [`docs/release/`](release/). [`MARKET_ANALYSIS.md`](MARKET_ANALYSIS.md)
(2026-09-13) is the analysis behind the ordering below; it is analysis, not a
decision, until the corresponding RFC is accepted.

## Order of work

| Order | RFC | Milestone | Status |
|---|---|---|---|
| 0 | [0003 Competitor benchmark and demo](rfcs/0003-competitor-benchmark-and-demo.md) | 0.2.0 | draft |
| 1 | [0004 Filtered search](rfcs/0004-filtered-search.md) | 0.2.0 | draft |
| 2 | [0006 WebAssembly persistence](rfcs/0006-wasm-persistence.md) | 0.2.0 | draft |
| 3 | [0002 C ABI distribution](rfcs/0002-c-abi-distribution.md), stages 1 and 5 | 0.2.0 | draft |
| 4 | [0010 Write-path gaps](rfcs/0010-write-path-gaps.md) | 0.2.0 | draft |
| 5 | [0005 Quantized storage](rfcs/0005-quantized-storage.md) | 0.3.0 | draft |
| 6 | [0002 C ABI distribution](rfcs/0002-c-abi-distribution.md), stages 2 to 4 | 0.3.0 | draft |
| 7 | [0007 Mobile SDKs](rfcs/0007-mobile-sdks.md) | 0.3.0 | draft |
| 8 | [0008 Streaming disk build and mapped graph](rfcs/0008-streaming-disk-build-and-mapped-graph.md) | 0.4.0 | draft |
| 9 | [0009 Optional payload column](rfcs/0009-payload-column.md) | 0.4.0 | draft |
| after the above | [0001 CUDA after the initial release](rfcs/0001-cuda-after-initial-release.md) | none assigned | accepted; ordering amendment draft |

Milestones: [0.2.0](https://github.com/vanedb/vanedb/milestone/1),
[0.3.0](https://github.com/vanedb/vanedb/milestone/2),
[0.4.0](https://github.com/vanedb/vanedb/milestone/3).

## CUDA on NVIDIA GPUs

CUDA is a required follow-up after the initial release, as agreed on
September 7, 2026. It has no assigned release version or delivery date, and
the unimplemented Rust stub is excluded from 0.1.1. A feature flag or kernel
source alone does not establish support. The implementation, real-hardware CI,
correctness, lifecycle, error-behaviour and end-to-end performance
requirements that must be met before CUDA is advertised are recorded in
[RFC 0001](rfcs/0001-cuda-after-initial-release.md); nothing may claim CUDA
support without that evidence. RFC 0001 also carries a draft amendment that
schedules CUDA after RFCs 0002 to 0007 and scopes a first target to NVIDIA
Jetson.

## Mobile device follow-up

Mobile verification starts with iOS simulators and Android emulators,
including Android ARM64 with 16 KiB pages. Physical-device acceptance follows
before broader device-support claims, and is now part of
[RFC 0007](rfcs/0007-mobile-sdks.md)'s acceptance criteria: run the same
all-metric C ABI lifecycle, graph persistence and mapped search checks on an
iPhone and an Android device, recording hardware, OS, source revision and
results. Simulator or emulator success must never be presented as
physical-device evidence.

## Open API question for 0.2: should `get` return `None` instead of raising?

An independent precedent survey done for vanedb#153 found that `.get()`
returning `None` on a miss is more settled across keyed stores than raising:
py-lmdb, plyvel, rocksdict, python-rocksdb, redis-py, usearch and
`collections.abc.Mapping.get` all do it. That makes a *raising* method named
`get` the outlier — independently of which exception it raises.

`vanedb`'s `get`/`get_vector` raise; the frozen C++ package is inconsistent
with itself here (`FlatIndex.get` and `DiskIndex.get` return `None`,
`ApproxIndex.get_vector` throws), so there is no single convention on that side
to match.

The first release retains this behavior because the Rust core has no
`Option`-returning accessor to bind, `get`/`get_vector` are the cross-engine
spelling settled in #85, and `contains` is already the non-raising probe. The
choice of `KeyError` makes the mismatch more conspicuous rather than
less, because `KeyError` is the exception `dict.get` exists to avoid.

Revisit deliberately: either accept the divergence and document `get` as
raising with `contains` as the probe, or add a `try_get`-style accessor to the
core first so every binding can offer both. Do not change one binding alone.
