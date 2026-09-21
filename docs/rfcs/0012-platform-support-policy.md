# RFC 0012: Platform support policy

- Status: implemented (0.2.0, unreleased)
- Milestone: 0.2.0
- Tracking issue: #207; resolves #47
- Supersedes / superseded by: none

## Problem

What VaneDB supports is scattered: the README's platform paragraph, the
release notes, `compatibility.json` inside each C archive, the MSRV in
`Cargo.toml`, the glibc floor in `package_capi.py`, and the deployment
targets in the publish workflows. Nothing states which of those are tested,
which are built and untested, and which are source-only. Issue #47 found that
GitHub's last Intel macOS runner retires in August 2027 with no successor, so
the Intel macOS wheel matrix has an end date that no document records.
Enterprise buyers ask for exactly this document; `SECURITY.md` exists and its
sibling does not.

## Decision

Publish `docs/PLATFORMS.md` with three tiers and a dated change log, make
every platform claim in the README point at it, and record the Intel macOS
decision: native x86-64 macOS wheels and C archives are dropped when the
`macos-15-intel` runner retires, with the sdist and source builds remaining
supported. No untested wheel is published in its place.

## Design

### Tiers

| Tier | Meaning | Today |
|---|---|---|
| **Tested** | built and tested in CI on the platform, every PR | Linux x86-64 and ARM64 (glibc 2.34+), macOS ARM64 (11.0+), macOS x86-64 (10.12+, until August 2027), Windows x64 (MSVC); Python 3.11 to 3.14 on each; Node current LTS; Chrome, Firefox, WebKit headless |
| **Built and verified on emulation** | built in CI, acceptance run on a simulator or emulator | iOS ARM64 (simulator), Android ARM64 and x86-64 (emulator, 16 KiB pages) |
| **Source** | expected to build from source; no artifact, no CI | any Rust tier-1/tier-2 target with `std`; musl Linux until RFC 0002 stage 4 promotes it |

Physical iOS and Android devices enter the second tier only with a recorded
run (RFC 0007). A platform moves up a tier only when CI proves it and down
with a changelog line and one minor release of notice, except when the
provider removes the platform (the Intel macOS case), where the notice is
this document's date.

### Floors

- Rust MSRV 1.85, checked in CI on that exact toolchain; raised only in a
  minor release with a changelog entry.
- glibc 2.34 for prebuilt Linux artifacts; `package_capi.py` rejects an
  increase.
- macOS deployment targets 11.0 (arm64), 10.12 (x86-64) until the x86-64 exit.
- Android API 21; iOS 13 for the C ABI, iOS 15 for the Swift package (RFC
  0007).
- Windows: MSVC toolchain, Universal CRT; a minimum Windows version is not
  verified and the document says so.

### Intel macOS exit

From the `macos-15-intel` retirement date (August 2027 per
actions/runner-images#13045): no `macosx_*_x86_64` wheels, no
`macos-x86_64` C archive. The sdist and `cargo build` keep working on Intel
Macs and remain in the source tier. Cross-compiled and `universal2` artifacts
are rejected for the reason recorded in #47: an artifact nothing executes is
worse than none. The wheel-count and platform-tag tripwires in both publish
workflows and `check_wheel_matrix.py` change in the same PR.

### Where the claims live

`docs/PLATFORMS.md` is the only place a tier is asserted. The README links
it; release notes cite it; `compatibility.json` remains the per-artifact
evidence. A CI test asserts that every platform named in the README appears
in `PLATFORMS.md`.

## Decisions recorded

- 2026-09-15: accepted as written, including the Intel macOS exit: no
  native x86-64 macOS wheels or C archives after the `macos-15-intel` runner
  retires in August 2027, no cross-compiled or `universal2` substitute, sdist
  and source builds remain supported.
- 2026-09-20: `docs/PLATFORMS.md` published, README and `SECURITY.md` link
  it, `vanedb/tests/platforms.rs` holds the two in sync. #47 closed with the
  decision recorded; the publish-matrix change is #258. Four qualifications
  the page records where the "Today" table above simplifies: the full Python
  wheel matrix is tested by the release workflow rather than on every pull
  request (a pull request tests one Linux x86-64 wheel on Python 3.11); musl
  Python wheels are already in the tested tier while the musl C ABI stays in
  the source tier until RFC 0002 stage 4; "Android ARM64 and x86-64
  (emulator, 16 KiB pages)" is per-pull-request acceptance on an API 29
  x86-64 emulator, with 16 KiB pages exercised only in the recorded 0.1.1
  ARM64 run; and "Node current LTS" is Node 22, the version the publish
  workflow tests.

## Alternatives rejected

- **Cross-compile Intel macOS on ARM runners.** Rejected: untested wheels,
  and #52 showed a wrong macOS wheel looks fine until it does not.
- **`universal2` wheels.** Rejected: doubles size for every user and still
  never executes the x86-64 half.
- **A self-hosted Intel Mac.** Rejected: indefinite hardware for a shrinking
  segment.

## Compatibility and migration

No code change. Intel Mac users lose prebuilt artifacts in August 2027 and
keep source installs. Both publish workflows change together (#47's rule).

## Acceptance criteria

- [x] `docs/PLATFORMS.md` published with the tiers, floors, exit date, and a
      dated change log.
- [x] README platform paragraph replaced by a summary and a link.
- [x] CI test: README platform names appear in `PLATFORMS.md`
      (`vanedb/tests/platforms.rs`).
- [x] #47 closed with the decision recorded (2026-09-20); the
      publish-workflow change is #258, scheduled before August 2027.
- [x] `SECURITY.md` links `PLATFORMS.md` for "supported versions" context.

## Evidence required before the claim

None beyond the existing CI; this RFC records what is already proven and
names what is not.

## Out of scope

Adding platforms. musl promotion is RFC 0002 stage 4; physical devices are
RFC 0007.
