# VaneDB roadmap

The 0.1.1 scope is the Rust engine and its Python, C and WebAssembly bindings,
with CPU search and the existing macOS Metal feature. CUDA is a required,
high-priority follow-up after the initial release, as agreed on September 7, 2026. It has no
assigned release version or delivery date. The C++ engine remains frozen
reference code.

## CUDA on NVIDIA GPUs — required after the initial release

Enable applications with NVIDIA hardware to accelerate vector workloads through
the Rust engine. CUDA is not supported in 0.1.1; the unimplemented Rust stub is
excluded from that release. A feature flag or kernel source alone does not
establish support.

Before advertising CUDA support, complete these requirements:

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

## Mobile device follow-up

Mobile verification starts with iOS simulators and Android emulators,
including Android ARM64 with 16 KiB pages. Physical-device acceptance follows
before broader device-support claims. Run the same all-metric C ABI lifecycle, graph persistence and mapped
search checks on an iPhone and an Android device, recording hardware, OS, source
revision and results. Simulator or emulator success must never be presented as
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

The first stable release retains this behavior because the Rust core has no
`Option`-returning accessor to bind, `get`/`get_vector` are the cross-engine
spelling settled in #85, and `contains` is already the non-raising probe. The
choice of `KeyError` makes the mismatch more conspicuous rather than
less, because `KeyError` is the exception `dict.get` exists to avoid.

Revisit deliberately: either accept the divergence and document `get` as
raising with `contains` as the probe, or add a `try_get`-style accessor to the
core first so every binding can offer both. Do not change one binding alone.
