# Android ARM64 runs (RFC 0003 third hardware class)

The competitor harness links usearch (C++), hnswlib (C++), and bundled SQLite.
A labelled Android ARM64 result is an acceptance criterion of #198.

## Preferred: physical device

1. Cross-compile `vanedb-compare` for `aarch64-linux-android` with the NDK,
   or build on-device in Termux with a full Rust+C++ toolchain.
2. Push `embeddings.vnef` + `SHA256SUMS` to the device.
3. Run with `VANEDB_COMPARE_HW=android-arm64-device` and `--rounds ≥ 2`.
4. Pull the JSON and paste markdown into `bench/COMPARISON.md`.

## Acceptable: emulator (must be labelled)

```text
VANEDB_COMPARE_HW=android-arm64-emulator
```

Use an ARM64 system image (not x86_64 with translation) when claiming ARM64.
State the AVD / API level in the COMPARISON.md notes for that section.

## Not acceptable

- Shared CI runners
- x86_64 Android emulator without an ARM64 label
- Smoke fixture numbers

Until a recorded Android run exists, the Android section of COMPARISON.md
stays `*Pending.*`.
