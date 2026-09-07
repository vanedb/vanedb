# VaneDB roadmap

The 1.0.0 scope is the Rust engine and its Python, C and WebAssembly bindings,
with CPU search and the existing macOS Metal feature. CUDA is a required,
high-priority follow-up after 1.0.0, as agreed on September 7, 2026. It has no
assigned release version or delivery date. The C++ engine remains frozen
reference code.

## CUDA on NVIDIA GPUs — required after 1.0.0

Enable applications with NVIDIA hardware to accelerate vector workloads through
the Rust engine. CUDA is not supported in 1.0.0; the unimplemented Rust stub is
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
