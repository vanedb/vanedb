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

Push binary + fixture (include the **committed** repo `SHA256SUMS` pin):

```bash
adb push bench/compare/target/aarch64-linux-android/release/compare /data/local/tmp/
adb push bench/compare/fixtures/embeddings.vnef /data/local/tmp/
adb push bench/compare/fixtures/SHA256SUMS /data/local/tmp/
adb push bench/compare/fixtures/metadata.json /data/local/tmp/
```

Do **not** rely on `--markdown` on-device as the primary paste path: prefer
recording JSON on-device **without** `--markdown` (still set
`VANEDB_COMPARE_HW` + `VANEDB_COMPARE_DEDICATED=1`), `adb pull` the JSON, then
on a checkout that has the committed publish pin:

```bash
adb shell 'cd /data/local/tmp && VANEDB_COMPARE_HW=android-arm64-device \
  VANEDB_COMPARE_DEDICATED=1 \
  ./compare run --fixture embeddings.vnef --metric cosine --rounds 4 \
  --json-out device-cosine-$(date +%Y%m%d).json'
adb shell 'cd /data/local/tmp && VANEDB_COMPARE_HW=android-arm64-device \
  VANEDB_COMPARE_DEDICATED=1 \
  ./compare run --fixture embeddings.vnef --metric l2 --rounds 4 \
  --json-out device-l2-$(date +%Y%m%d).json'
adb pull /data/local/tmp/device-cosine-*.json bench/compare/runs/
adb pull /data/local/tmp/device-l2-*.json bench/compare/runs/
python3 bench/compare/scripts/render_comparison_md.py bench/compare/runs/<pulled-cosine>.json
python3 bench/compare/scripts/render_comparison_md.py bench/compare/runs/<pulled-l2>.json
```

Optional on-device `--markdown`: rebuild `compare` **after** committing the
`embeddings.vnef` line to `fixtures/SHA256SUMS` so the compile-time pin is
baked into the Android binary, then push that binary + fixture + SUMS + meta.
There is no env override for the pin.

Paste each markdown section into the matching Cosine / Squared L2 slot under
Android in `bench/COMPARISON.md`.

### On-device (Termux)

Full Rust + `clang`/`clang++` with libc++. Same `cargo build --release --locked
--manifest-path bench/compare/Cargo.toml`, then run as above with
`VANEDB_COMPARE_HW=android-arm64-device VANEDB_COMPARE_DEDICATED=1`.

## Acceptable: emulator (must be labelled)

```text
VANEDB_COMPARE_HW=android-arm64-emulator
VANEDB_COMPARE_DEDICATED=1
```

Use an **ARM64** system image (not x86_64 with translation) when claiming ARM64.
State the AVD name and API level in the COMPARISON.md notes for that section.

## Not acceptable

- Shared CI runners
- x86_64 Android emulator without an ARM64 label
- Smoke fixture numbers

Until a recorded Android run exists, the Android section of COMPARISON.md
stays `*Pending.*`.
