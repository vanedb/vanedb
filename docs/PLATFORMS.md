# Platform support

This page is the only place VaneDB asserts a support tier
([RFC 0012](rfcs/0012-platform-support-policy.md)). The README summarises it,
release notes cite it, and each C archive's `compatibility.json` remains the
per-artifact evidence. Every row below names the CI job or recorded run that
backs it; a platform without such evidence is not listed above the source
tier, whatever it happens to work on.

During 0.x, APIs and persistence formats may change in a minor release. This
page records what is proven today, not a 1.x commitment.

## Tiers

| Tier | Meaning |
|---|---|
| **Tested** | built and tested in CI on the platform itself, on every pull request |
| **Built and verified on emulation** | built in CI; acceptance runs on a simulator or emulator, never on a physical device |
| **Source** | expected to build from source; no artifact and no CI |

A platform moves up a tier only when CI proves it. It moves down with a line
in the [change log](#change-log) and one minor release of notice, except when
the provider removes the platform, as with [Intel macOS](#intel-macos-exit),
where the notice is the date this page records the decision.

### Tested

Every pull request runs `cargo test --workspace --exclude vanedb-py --features
disk --locked`, the C consumer acceptance and the C archive packaging on each
row of this table (`rust-ci.yml`, `test-native`). The engine that ships is the
Rust engine; the frozen C++ reference has its own matrix in `cpp-ci.yml` and
makes no claim here.

| Platform | Rust target | Runner | Artifacts | Notes |
|---|---|---|---|---|
| Linux x86-64 | `x86_64-unknown-linux-gnu` | `ubuntu-24.04` | C archive; wheels (glibc and musl) | glibc 2.34 floor for the archive; AddressSanitizer job covers the AVX2 kernels and the C ABI on this platform only |
| Linux ARM64 | `aarch64-unknown-linux-gnu` | `ubuntu-24.04-arm` | C archive; wheels (glibc and musl) | NEON kernels tested here, unsanitized |
| macOS ARM64 | `aarch64-apple-darwin` | `macos-latest` | C archive; wheels | tested with `disk,gpu-metal`; deployment target 11.0 |
| macOS Intel (x86-64) | `x86_64-apple-darwin` | `macos-15-intel` | C archive; wheels | deployment target 10.12; **until August 2027**, see below |
| Windows x64 | `x86_64-pc-windows-msvc` | `windows-latest` | C archive; wheels | MSVC toolchain, Universal CRT |
| WebAssembly | `wasm32-unknown-unknown` | `ubuntu-latest` | npm package (Node and browser halves) | tested in Node.js and headless Chrome (`wasm-pack test`); the packaged web artifact and the merged npm package run in Chrome, Firefox and WebKit through Playwright |

Python wheels are in this tier with one qualification. The full matrix (Linux
x86-64 and ARM64 as `manylinux_2_17` and `musllinux_1_2`, macOS ARM64 and
macOS Intel, Windows x64, each for Python 3.11, 3.12, 3.13 and 3.14: 28 wheels) is
built, installed and run through the test suite by the release workflow
(`publish-rust.yml`) on every tag and rehearsal, with the musl wheels tested in
`python:<version>-alpine` containers. A pull request tests one Linux x86-64
wheel on Python 3.11, on the newest and the minimum supported maturin, plus
`cargo check` of the bindings crate. The Python floor is `requires-python
>= 3.11`.

### Built and verified on emulation

| Platform | Rust target | CI evidence |
|---|---|---|
| iOS ARM64 | `aarch64-apple-ios` (device build), `aarch64-apple-ios-sim` | core and C ABI built with `disk` on every pull request; C ABI acceptance runs in an iOS ARM64 simulator on the newest installed runtime (`scripts/test_ios.py`, compiled against the iOS 14.0 simulator target) |
| Android ARM64 (`arm64-v8a`) | `aarch64-linux-android` | built with `cargo-ndk` 4.1.2 and NDK r26b on every pull request, linked with 16 KiB page alignment and checked by `scripts/check_android_elf.py`; the runtime bundle is uploaded as `vanedb-android-arm64-runtime`. The recorded runtime pass is the 0.1.1 release run on an Android 15 emulator (API 35, 16,384-byte pages) with the CI-built binary: [release evidence](release/0.1.1-readiness.md). That run is not repeated per pull request |
| Android x86-64 | `x86_64-linux-android` | built as above; C ABI acceptance runs in an Android x86-64 emulator (API 29) on every pull request |

None of this establishes behaviour on an iPhone or an Android device. Physical
devices enter this tier only with a recorded run, which is an acceptance
criterion of [RFC 0007](rfcs/0007-mobile-sdks.md).

### Source

Expected to build from source; nothing is published for them and nothing
runs in CI.

- Any Rust tier-1 or tier-2 target with `std`. `vanedb` has no platform-
  specific dependencies beyond `memmap2` for the `disk` feature.
- musl Linux for the Rust crate and the C ABI, until
  [RFC 0002](rfcs/0002-c-abi-distribution.md) stage 4 (#196) packages and
  tests musl static libraries. (musl Python wheels are already in the tested
  tier; see above.)
- Windows with the GNU toolchain (`x86_64-pc-windows-gnu`), same RFC and
  stage.
- Intel macOS from August 2027: the sdist and `cargo build`.

Not supported: CUDA is not implemented and is not claimable without the
evidence in [RFC 0001](rfcs/0001-cuda-after-initial-release.md). `gpu-metal`
builds and is tested only on macOS ARM64.

## Floors

| Floor | Value | How it is held |
|---|---|---|
| Rust | MSRV 1.85 | `rust-version` in every crate; the `msrv` job runs `cargo check` on the 1.85.0 toolchain on Linux and macOS, including `wasm32-unknown-unknown` and `gpu-metal`. Tests run on stable. Raised only in a minor release with a changelog entry |
| glibc, C archives | 2.34 | `scripts/package_capi.py` reads the packaged library's `GLIBC_*` symbol versions, records the minimum in `compatibility.json` and fails the build above 2.34 |
| glibc, Python wheels | `manylinux_2_17` tag (glibc 2.17) | maturin builds and audits the wheel against that tag; musl wheels carry `musllinux_1_2`. The suite runs on the runner's own glibc, not on a 2.17 system |
| macOS, C archives | 11.0 on ARM64, 10.12 on x86-64 | `package_capi.py` inspects the Mach-O `minos` and rejects any other value; `MACOSX_DEPLOYMENT_TARGET` is set per leg in `test-native` |
| macOS, wheels | 11.0 on ARM64, 10.12 on x86-64 | maturin's defaults for each architecture; the floor is in the wheel's `macosx_*` platform tag. The release validator checks the `macosx` prefix and count, not the version, so read the tag (#52 is why this matters) |
| Android | API 21 | the C ABI acceptance is compiled with the NDK's `*-android21-clang`; every Android library is linked with `max-page-size=16384` and rejected by `check_android_elf.py` otherwise (a 4 KiB probe library is checked to fail). Emulator acceptance per pull request runs on API 29; the 16 KiB-page runtime evidence is the recorded Android 15 run above |
| iOS | 13 for the C ABI | the deployment target [RFC 0002](rfcs/0002-c-abi-distribution.md) declares. CI evidence is the simulator run on the newest installed runtime; no run on iOS 13 itself is recorded. The Swift package of RFC 0007, not yet shipped, will target iOS 15 |
| Windows | MSVC toolchain, Universal CRT | the C archive imports `VCRUNTIME140` and the Universal CRT, listed in `compatibility.json`. **A minimum Windows version is not verified**: consumer acceptance runs on `windows-latest`, and the PE subsystem version does not establish one |
| Python | 3.11 to 3.14 | `requires-python >= 3.11` in `vanedb-py/pyproject.toml`; the release matrix builds and tests each of the four versions on every tested platform |
| Node.js | 22 tested; package declares `engines.node >= 18` | the publish workflow builds and tests on Node 22; pull requests use the runner's default Node. Nothing runs on Node 18 or 20, so the `engines` range is a declaration, not a tested floor |
| Browsers | current Chrome, Firefox and WebKit | the versions Playwright and `wasm-pack` install at run time; no minimum browser version is verified |
| Page size | 4 KiB on Linux runners; 16 KiB on Android | `DiskIndex` maps files and Android loads the C ABI as a shared library, so page size matters. Linux x86-64 and ARM64 tests run on 4 KiB kernels; the Android libraries are 16 KiB-aligned and were run once on 16 KiB pages (above). Linux kernels with 64 KiB pages are untested |

## Intel macOS exit

GitHub's `macos-15-intel` hosted runner, the last Intel macOS runner, retires
in August 2027 with no successor
([actions/runner-images#13045](https://github.com/actions/runner-images/issues/13045)).
Decision, recorded by RFC 0012 on 2026-09-15 and taking #47's rule that both
publish workflows change together:

- From that date there are no `macosx_*_x86_64` wheels and no `macos-x86_64`
  C archive. The wheel matrix drops from 28 to 24 wheels, `macosx` from 8 to 4
  and each `cp3xx` tag from 7 to 6; the C archives drop from five to four.
- The sdist and `cargo build` keep working on Intel Macs, which move to the
  source tier.
- No untested wheel is published in its place. Cross-compiling on ARM
  runners, `universal2` wheels and a self-hosted Intel Mac were each
  rejected: an artifact nothing executes is worse than none, and #52 showed
  a wrong macOS wheel looks fine until it does not.
- The wheel-count and platform-tag tripwires in both publish workflows and
  `.github/scripts/check_wheel_matrix.py`, plus the `macos-x86_64` legs of
  `test-native`, change in one pull request. That change is a separate issue
  scheduled before August 2027; this page is the notice.

Until then, macOS x86-64 stays in the tested tier: every pull request still
runs the full suite and the C consumer acceptance on `macos-15-intel`.

## Keeping this page honest

- `vanedb/tests/platforms.rs` fails when the README names a platform this
  page does not, when the README stops linking here, or when this page loses
  a tier, the floors, the exit date or the change log.
- `scripts/package_capi.py` fails a C archive whose glibc floor rose or whose
  macOS deployment target changed.
- `.github/scripts/validate_python_release.py` fails a release whose wheel
  count or platform tags disagree with the matrix above.
- A claim that does not appear in a workflow, a script or a recorded run in
  `docs/release/` does not belong on this page.

## Change log

Dated; newest first. Tier moves and floor changes are recorded here and in
[`CHANGELOG.md`](../CHANGELOG.md).

- **2026-09-20** — First publication, implementing RFC 0012 (accepted
  2026-09-15). Tiers and floors record what CI already proved; no platform
  changed tier. Records the Intel macOS exit: native x86-64 macOS wheels and
  C archives end when `macos-15-intel` retires in August 2027, with sdist and
  source builds remaining. One minor release of notice does not apply, since
  the provider is removing the platform.
