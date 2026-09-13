# RFC 0007: Mobile SDKs

- Status: accepted (2026-09-13)
- Milestone: 0.3.0
- Tracking issue: #202
- Supersedes / superseded by: none

## Problem

The tagline says "edge AI". The only mobile surface is a C header and a
shared library verified on an iOS simulator and Android emulators. A Swift or
Kotlin developer evaluating on-device vector search compares against ObjectBox
(Swift, Kotlin, Dart), Couchbase Lite (Swift, Kotlin), USearch (Swift, Java)
and Google's AI Edge RAG SDK (Kotlin, Flutter). A C header is not on their
list. Practitioner reports agree the algorithm is the easy part and the
lifecycle (backgrounding, memory pressure, encrypted storage, file backup) is
the work; a language SDK is where that work can live.

RFC 0002 produces the binaries this RFC wraps.

## Decision

Ship a Swift package and a Kotlin library generated from one interface
definition with `uniffi`, layered on the RFC 0002 xcframework and AAR. Each
exposes the three index types, batch add, search with per-query beam width
and filter (RFC 0004), delete, upsert, compact, save and load. Physical-device
acceptance is part of this RFC's evidence, not a later promise.

## Design

### Binding generator

`uniffi` (Mozilla) over a new crate `vanedb-uniffi` that depends on `vanedb`
directly, not on the C ABI. Reasons: one `.udl`/proc-macro interface produces
Swift and Kotlin with typed errors mapped from `VaneError`, `u64` ids become
`UInt64`/`ULong`, and buffers cross as `[Float]`/`FloatArray` without a hand
written JNI layer. The C ABI stays the surface for C and C++ consumers.

### Swift

- `VaneDB` Swift package with a `binaryTarget` (the uniffi-built xcframework,
  distinct from the C ABI xcframework) and a source target with the generated
  Swift.
- Errors: `VaneError` as a Swift `Error` enum mirroring the Rust variants.
- Conveniences in hand-written Swift on top of the generated layer: `Sendable`
  wrappers, `[Float]` and `MLMultiArray` inputs, `URL`-based save/load, and a
  documented pattern for `UIApplication` background transitions (compact and
  save on `didEnterBackground`).
- Deployment: iOS 15, macOS 12 (uniffi's Swift runtime floor is checked at
  implementation time and recorded).

### Kotlin

- `io.vanedb:vanedb` AAR with the generated Kotlin and the uniffi-built
  native libraries per ABI. uniffi Kotlin uses JNA; the size cost is measured
  and recorded, and if it exceeds 1 MB per ABI the RFC is amended to a
  hand-written JNI shim before release.
- Coroutine-friendly: search and batch add are `suspend`-free and thread-safe,
  documented for `Dispatchers.Default`.
- `File`-based save/load; a documented pattern for `onTrimMemory`.

### Shared

- Same acceptance lifecycle as `vanedb-capi/tests/acceptance.c`, ported to a
  Swift XCTest and a Kotlin instrumented test: all three metrics, graph
  persistence, mapped search on a file copied from the app bundle.
- Sample apps: one SwiftUI and one Compose screen that embed sentences with an
  on-device model (Core ML / EmbeddingGemma via LiteRT) and search them. The
  embedding step is the sample's, not the SDK's.

### Physical-device evidence

Moved here from the roadmap's mobile follow-up: run the acceptance lifecycle
on an iPhone and on an Android device, recording hardware, OS, source revision
and results. Simulator or emulator success is never presented as device
evidence. The 0.3.0 readiness record carries these runs or states their
absence.

## Decisions recorded

- 2026-09-13: `uniffi` accepted over hand-written bindings (decision 9);
  Flutter and React Native deferred (decision 10).
- An Apple developer account is available to the maintainer, so xcframework
  signing, TestFlight sample distribution and physical iPhone runs are not
  blocked on an account.

## Alternatives rejected

- **Hand-written Swift and Kotlin over the C ABI.** Rejected for the first
  release: two hand-maintained bindings drift, as the Python and C surfaces
  did before #85/#86. Revisit only if uniffi's runtime cost is unacceptable.
- **Flutter / Dart now.** Deferred: add when a Dart user asks; ObjectBox owns
  that niche today.
- **React Native.** Deferred: `op-sqlite` plus `sqlite-vec` is entrenched;
  the wasm package covers Expo web, and a Nitro module can wrap the C ABI
  later.

## Compatibility and migration

- Additive: new crate, new packages. The core and C ABI are unchanged.
- The uniffi crate's own release tag (`vanedb-mobile-vX.Y.Z`) follows the
  existing per-distribution tag scheme.

## Acceptance criteria

- [ ] `vanedb-uniffi` crate with the full index surface and typed errors.
- [ ] Swift package builds with SwiftPM; XCTest acceptance passes on the iOS
      simulator and on macOS in CI.
- [ ] Kotlin AAR builds with Gradle; instrumented acceptance passes on the
      x86-64 and ARM64 emulators in CI; JNA size cost recorded.
- [ ] Physical-device runs recorded for one iPhone and one Android device
      (hardware, OS, revision, result) in `docs/release/0.3.0-readiness.md`,
      or the record states explicitly that they did not happen.
- [ ] Sample apps build and are linked from the README.
- [ ] Package sizes recorded; `README.md` bindings table gains Swift and
      Kotlin rows with what each supports.
- [ ] `CHANGELOG.md` entry.

## Evidence required before the claim

Device runs as above. Without them the README says "verified on simulator and
emulator", as it does today.

## Out of scope

Encrypted storage, sync, Flutter, React Native, on-device embedding models.
