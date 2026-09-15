# RFC 0002: C ABI distribution

- Status: accepted (2026-09-13)
- Milestone: 0.2.0 (stages 1 and 5), 0.3.0 (stages 2, 3 and 4)
- Tracking issues: #193 (stage 1), #194 (stage 2), #195 (stage 3), #196 (stage 4), #197 (stage 5)
- Supersedes / superseded by: none

## Problem

`vanedb-capi` already has a sound C ABI: opaque `vanedb_rs_*` handles, an
integer metric, a thread-local error channel with sixteen codes, a panic
boundary, `vanedb_rs_version()`, a cbindgen header regenerated and diffed in
CI, per-platform zips with a `compatibility.json`, and 16 KiB page alignment
for Android. What it lacks is everything that lets a C or C++ consumer use it
without Cargo and without reading a README:

- no `find_package`, no `pkg-config`, no recorded list of the system libraries
  a static link needs;
- no Apple xcframework, no Swift Package Manager binary target, no Android AAR;
- the shared library exports every Rust symbol, so two Rust-built libraries in
  one process can collide;
- no ABI-break gate: a changed signature is caught only if a consumer notices;
- artifacts are attached to CI runs, not to releases, and are unsigned.

The C++ engine's one remaining advantage over the Rust engine is that a
header-only library drops into any native build. This RFC closes that gap
without porting anything to C++.

## Decision

Ship the C ABI as a proper prebuilt distribution: a stable integer ABI version,
an exported-symbol allowlist, CMake and pkg-config consumer files, an Apple
xcframework with a SwiftPM binary target, an Android AAR with prefab modules,
musl and Windows GNU static libraries, vcpkg and Conan recipes, and signed
release assets. Every artifact is exercised by a consumer project that uses
only the installed layout.

## Design

### Stage 1: contract and consumer files (0.2.0)

- `VANEDB_RS_ABI_VERSION` integer macro and `uint32_t vanedb_rs_abi_version(void)`.
  Bumped only on an incompatible change. `VANEDB_RS_VERSION` stays as the
  semver string.
- Handles become integers. Every `vanedb_rs_store*`, `vanedb_rs_index*` and
  `vanedb_rs_disk*` value crossing the boundary is a `uint64_t` id looked up
  in a process-wide table, not a pointer from `Box::into_raw`. An unknown,
  stale or truncated id fails the lookup and returns a new status code,
  `VANEDB_RS_INVALID_HANDLE`; today it is dereferenced. Motivation: a Python
  `ctypes` caller that omits `restype` truncates the returned pointer to a C
  `int` (observed: `x0 = 0xc7d590` for a live `0x100c7d390`) and crashes in
  `vanedb_rs_store_add`; the same happens with any FFI that guesses types.
  Cost: one lookup under a sharded lock per call, below the search noise
  floor. Also gives double-free and use-after-free detection and a
  `vanedb_rs_handle_count()` for leak tests. This is the one incompatible
  change in stage 1 and the reason `VANEDB_RS_ABI_VERSION` starts at 1.
- Rule, written into the header comment: signatures never change. New behaviour
  arrives as a new function (`_ex` or `_v2`); old functions stay. No struct
  crosses the boundary; if one ever does, its first field is `size_t size`.
- Exported-symbol allowlist generated from the header by a script in
  `scripts/`: a GNU linker version script for Linux and Android, an
  `-exported_symbols_list` for Apple, a `.def` for Windows. Only `vanedb_rs_*`
  is exported from the shared library. The staticlib is additionally
  post-processed so that `nm --defined-only --extern-only` shows only
  `vanedb_rs_*` and the symbols `--print native-static-libs` requires.
- `lib/cmake/vanedb/vanedbConfig.cmake`, `vanedbConfigVersion.cmake`,
  `vanedbTargets.cmake` providing imported targets `vanedb::shared` and
  `vanedb::static`, with `INTERFACE_LINK_LIBRARIES` on the static target set
  from `cargo rustc --print native-static-libs` at package time.
- `lib/pkgconfig/vanedb.pc` with `Libs`, `Libs.private`, `Cflags`.
- `abidiff` (libabigail) in CI on Linux against the previous tagged release's
  shared object; a removed or changed symbol fails the job. The first tagged
  baseline is 0.2.0 itself.
- The header compiles as C99, C11 and C++17 under
  `-Wall -Wextra -pedantic -Werror` (Clang, GCC, MSVC `/W4 /WX`).
- Release profile for the crate: `lto = "fat"`, `codegen-units = 1`,
  `opt-level = 3`; the shared library is stripped, the static library is not.
  `panic = "unwind"` is retained: `panic = "abort"` would make the existing
  `catch_unwind` boundary inert and abort the host process.

### Stage 2: Apple (0.3.0)

- Targets: `aarch64-apple-darwin`, `x86_64-apple-darwin`, `aarch64-apple-ios`,
  `aarch64-apple-ios-sim`, `x86_64-apple-ios`, `aarch64-apple-ios-macabi`.
- `vanedb.xcframework` built from static libraries: one slice per platform,
  simulator slices lipo'd, headers and a `module.modulemap` included so Swift
  can import the C module directly.
- `Package.swift` at the repository root with a `binaryTarget` whose URL is the
  release asset and whose checksum is computed at release time. The repository
  URL is then the SwiftPM package URL.
- Deployment targets: macOS 11.0 on arm64, 10.12 on x86_64 (as today), iOS 13.

### Stage 3: Android (0.3.0)

- Targets: `aarch64-linux-android`, `armv7-linux-androideabi`,
  `x86_64-linux-android`, `i686-linux-android`, API 21, built with `cargo-ndk`.
- An AAR containing `prefab/modules/vanedb/` with the shared library per ABI,
  the header, and `abi.json`. Gradle consumers with `prefab true` get
  `find_package(vanedb CONFIG)` in their CMake for free.
- 16 KiB page alignment retained for every ABI; `check_android_elf.py` runs on
  each.
- Published to GitHub Packages (Maven) at release; Maven Central once a
  namespace is verified.

### Stage 4: other native ecosystems (0.3.0)

- Static libraries for `x86_64-unknown-linux-musl`,
  `aarch64-unknown-linux-musl`, `x86_64-pc-windows-gnu`, with their own
  `compatibility.json`.
- A vcpkg port and a Conan recipe, each downloading the release zip and
  installing the CMake config. Both live in-repo under `packaging/`.

### Stage 5: releases, not CI artifacts (0.2.0)

- All C ABI artifacts (zips, xcframework, AAR, `SHA256SUMS`, SBOM from
  `cargo cyclonedx`) attached to the GitHub Release of the
  `vanedb-crate-vX.Y.Z` tag. Signed with Sigstore `cosign` keyless signing;
  the release notes state how to verify.
- `vanedb-capi/README.md` points at the release, not at "a successful CI run".

## Decisions recorded

- 2026-09-13: static-only xcframework, `abidiff` gate, keyless `cosign`
  signing, and stage 5 before the mobile stages accepted (decision 7).
- 2026-09-15: integer handles accepted over a pointer registry (which
  cannot catch a truncated pointer that collides with a live one) and over
  documenting `restype` (which leaves the crash reachable by every future
  binding author). Tracked in #193.

## Alternatives rejected

- **Port the engine to header-only C++.** Rejected by decision in issue #100
  and `AGENTS.md`: the C++ engine is frozen reference code.
- **`uniffi` now, C ABI packaging later.** Rejected: the generated Swift and
  Kotlin bindings (RFC 0007) sit on exactly these binaries; packaging comes
  first.
- **Shared libraries only in the xcframework.** Rejected: iOS consumers
  overwhelmingly link statically, and a dynamic framework needs code signing
  by the consumer.
- **`panic = "abort"` for size.** Rejected: see stage 1.

## Compatibility and migration

- Function names and signatures are unchanged except that handle-typed
  parameters and returns become `uint64_t`. Consumers that stored the
  opaque pointer type recompile; consumers that never dereferenced it need
  no source change. `vanedb_rs_abi_version()` is additive.
- The existing zips keep their layout and gain `lib/cmake` and
  `lib/pkgconfig`.
- Consumers who linked the raw `.so` or `.dylib` from a CI artifact continue
  to work; the README stops recommending it.
- `AGENTS.md` invariants are untouched: no format, kernel, or graph change.

## Acceptance criteria

Stage 1 (#193):

- [ ] `vanedb_rs_abi_version()` and `VANEDB_RS_ABI_VERSION` exist and agree.
- [ ] Handles are `uint64_t` ids; a test passes a truncated, freed and random
      id to every entry point and gets `VANEDB_RS_INVALID_HANDLE`, never a
      crash; `vanedb-py` is unaffected (PyO3, not the C ABI); the README's
      C ABI section gains a `ctypes` snippet that sets `restype` and
      `argtypes`, for callers who bypass the header.
- [ ] Exported symbols of the shared library on Linux, macOS and Windows are
      exactly the `vanedb_rs_*` set; a CI step asserts it with `nm`/`dumpbin`.
- [ ] A consumer project using only `find_package(vanedb)` builds, links both
      imported targets, and passes `acceptance.c` on Linux x86-64, Linux ARM64,
      macOS ARM64, macOS x86-64 and Windows x64.
- [ ] A consumer using only `pkg-config --cflags --libs vanedb` does the same
      on Linux and macOS.
- [ ] Header compiles warning-free as C99, C11 and C++17 on Clang, GCC and MSVC.
- [ ] `abidiff` job present; documented baseline procedure for the first tag.
- [ ] Release profile applied; shared-library size before and after recorded
      in the README.

Stage 2 (#194):

- [ ] xcframework built in CI with the six slices; `lipo -info` recorded.
- [ ] A SwiftPM package that depends on the repository URL imports the module
      and runs the acceptance lifecycle on an iOS simulator and on macOS.
- [ ] `Package.swift` checksum updated by the release workflow.

Stage 3 (#195):

- [ ] AAR built in CI with four ABIs; `check_android_elf.py` passes on each.
- [ ] A Gradle project consuming the AAR through prefab runs the acceptance
      lifecycle on the x86-64 emulator (existing job) and the ARM64 emulator.

Stage 4 (#196):

- [ ] musl and Windows GNU static libraries packaged with `compatibility.json`.
- [ ] vcpkg port and Conan recipe install and pass the consumer test.

Stage 5 (#197):

- [ ] Release workflow attaches every artifact, checksums and SBOM to the
      GitHub Release and signs them; verification command in the notes.
- [ ] `vanedb-capi/README.md` and `docs/release/RELEASING.md` updated.

## Evidence required before the claim

Consumer acceptance on the recorded CI hosts and simulators, as today. Physical
iPhone and Android device runs remain RFC 0007's follow-up; nothing here claims
them.

## Out of scope

Swift and Kotlin language bindings (RFC 0007). Changes to the ABI's functions.
