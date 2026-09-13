# RFC 0001: CUDA after the initial release

- Status: accepted (decision of 2026-09-07; ordering amendment accepted
  2026-09-13)
- Milestone: none assigned
- Tracking issue: #205 (amendment decision)
- Supersedes / superseded by: none

## Problem

The 0.1.1 scope is the Rust engine and its Python, C and WebAssembly bindings,
with CPU search and the existing macOS Metal feature. CUDA on NVIDIA GPUs was
excluded, and the unimplemented Rust stub was removed from the release. Users
with NVIDIA hardware cannot accelerate vector workloads through the engine.

## Decision

CUDA is a required follow-up after the initial release. It has no assigned
release version or delivery date. A feature flag or kernel source alone does
not establish support; the requirements below must be met before CUDA is
advertised, and no CUDA claim may be made without the evidence they name.

## Requirements before advertising CUDA support

These are the requirements recorded in the roadmap on 2026-09-07, unchanged.

- Implement the usable Rust API and document which operations and bindings are
  accelerated. Publish the supported OS/architecture, NVIDIA GPU architectures,
  CUDA toolkit and driver requirements, including the combinations actually
  verified. Check for unsupported configurations and explain how to resolve them.
- Add CI that builds the feature and runs it on real NVIDIA hardware. Record the
  GPU, driver, toolkit and platform with each result. A CPU-only compile check is
  insufficient for a CUDA release.
- Compare squared L2, cosine distance and negative dot product against the CPU
  reference with documented floating-point tolerances and result ordering. Cover
  empty inputs, zero and uneven dimensions, dimension mismatches, zero vectors,
  large and tiny finite values, non-finite inputs, ties and boundary-sized batches.
- Verify context, stream and allocation lifetimes, repeated construction and
  destruction, concurrent use, and cleanup after partial initialization or failed
  operations. Exercise allocation exhaustion and GPU/driver errors without
  leaving invalid state or leaked resources.
- Specify observable error behavior and when CPU fallback is available. Make the
  selected execution path visible to callers and preserve metric, ordering and
  validation semantics when falling back. Unsupported hardware and unavailable
  drivers must have a documented outcome.
- Measure end-to-end performance on dedicated NVIDIA hardware, including host
  preparation, allocation, transfers and synchronization. Report cold and reused
  workloads, corpus and query sizes, memory use, correctness/recall, CPU baselines
  and repeated-run variation. State where GPU use helps and where its overhead
  makes CPU execution preferable; kernel-only timing is not an application gain.

## Amendment: ordering (accepted 2026-09-13)

[`MARKET_ANALYSIS.md`](../MARKET_ANALYSIS.md) found that CUDA contradicts the
stated positioning: the edge audience runs on NPUs, mobile GPUs and CPU SIMD,
no on-device competitor offers CUDA, and the server audience that wants it uses
FAISS or cuVS. The amendment:

1. CUDA is scheduled behind RFCs 0002 to 0007 (C ABI distribution, competitor
   benchmark, filtered search, quantized storage, WebAssembly persistence,
   mobile SDKs). "Required" is retained; "high priority" is not.
2. If CUDA is scoped, the first target is NVIDIA Jetson, which is an edge
   platform and matches the positioning. Discrete workstation GPUs follow only
   with a demonstrated application gain over the CPU path on the same host.
3. A `FlatIndex` brute-force scan is the first accelerated operation. Graph
   construction and graph search on the GPU are out of scope until the scan
   shows an end-to-end gain.

The amendment changed `AGENTS.md`'s "required, high-priority
follow-up" wording to "required follow-up, scheduled after RFCs 0002 to 0007".
It changes none of the requirements above.

## Alternatives rejected

- **Ship a CUDA feature flag now and harden later.** Rejected on 2026-09-07:
  a flag that does nothing is a false claim.
- **Drop CUDA.** Rejected: NVIDIA edge hardware (Jetson) is inside the target
  audience, and a documented path to it has value even without a date.

## Compatibility and migration

No public API or file format changes until an implementation RFC exists. That
RFC must specify how the selected execution path is visible to callers in every
binding.

## Acceptance criteria

For the amendment:

- [x] Maintainer accepted the ordering on 2026-09-13 (#205).
- [x] `AGENTS.md` and `README.md` wording updated to match.
- [x] `docs/ROADMAP.md` lists CUDA after RFCs 0002 to 0007.

For CUDA itself: the requirements section above, each item with recorded
evidence.

## Evidence required before the claim

Real NVIDIA hardware, named GPU, driver, toolkit and platform, per run. Never a
CPU-only compile, never a CI timing.

## Out of scope

Metal. The existing `gpu-metal` feature exposes distance scans without
accelerating an index. Decision 14 (2026-09-13): the README claim is removed
now (#208), the feature flag stays, and finish-versus-delete is decided after
0.3.0.
