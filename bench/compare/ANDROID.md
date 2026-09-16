# Android ARM64 runs (RFC 0003 third hardware class)

The competitor harness links usearch (C++), hnswlib (C++), and bundled SQLite.
A labelled Android ARM64 result is an acceptance criterion of #198.

## Preferred: physical device

### Cross-compile from Linux (NDK r26+)

```bash
# One-time
rustup target add aarch64-linux-android
# Point at your NDK (example paths — adjust):
export ANDROID_NDK_HOME="${ANDROID_NDK_HOME:-$HOME/Android/Sdk/ndk/26.1.10909125}"
export CC_aarch64_linux_android="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android24-clang"
export CXX_aarch64_linux_android="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android24-clang++"
export AR_aarch64_linux_android="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-ar"
export CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER="$CC_aarch64_linux_android"

# hnswlib / usearch need a C++17 toolchain; prefer the NDK clang++ above.
cargo build --release --locked --manifest-path bench/compare/Cargo.toml \
  --target aarch64-linux-android
```

Push binary + fixture:

```bash
adb push bench/compare/target/aarch64-linux-android/release/compare /data/local/tmp/
adb push bench/compare/fixtures/embeddings.vnef /data/local/tmp/
adb push bench/compare/fixtures/SHA256SUMS /data/local/tmp/
adb push bench/compare/fixtures/metadata.json /data/local/tmp/
adb shell 'cd /data/local/tmp && VANEDB_COMPARE_HW=android-arm64-device \
  ./compare run --fixture embeddings.vnef --rounds 4 --markdown \
  --json-out device-$(date +%Y%m%d).json'
adb pull /data/local/tmp/device-*.json bench/compare/runs/
```

Paste the markdown section into `bench/COMPARISON.md` (Android section).

### On-device (Termux)

Full Rust + `clang`/`clang++` with libc++. Same `cargo build --release --locked
--manifest-path bench/compare/Cargo.toml`, then run as above with
`VANEDB_COMPARE_HW=android-arm64-device`.

## Acceptable: emulator (must be labelled)

```text
VANEDB_COMPARE_HW=android-arm64-emulator
```

Use an **ARM64** system image (not x86_64 with translation) when claiming ARM64.
State the AVD name and API level in the COMPARISON.md notes for that section.

## Not acceptable

- Shared CI runners
- x86_64 Android emulator without an ARM64 label
- Smoke fixture numbers

Until a recorded Android run exists, the Android section of COMPARISON.md
stays `*Pending.*`.
