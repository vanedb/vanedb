#!/usr/bin/env bash
# Build and/or run the C acceptance consumer on an Android device.
set -euo pipefail
cd "$(dirname "$0")/.."

mode=${1:-build-and-run}
case "$mode" in
  build-and-run|--build-only|--run-only) ;;
  *) echo 'Usage: test_android.sh [--build-only|--run-only]' >&2; exit 1 ;;
esac
abi=${VANEDB_ANDROID_ABI:-x86_64}
case "$abi" in
  x86_64) rust_target=x86_64-linux-android ;;
  arm64-v8a) rust_target=aarch64-linux-android ;;
  *) echo "Unsupported Android ABI: $abi" >&2; exit 1 ;;
esac
runtime_dir="${VANEDB_ANDROID_RUNTIME_DIR:-${CARGO_TARGET_DIR:-target}/android-$abi-runtime}"

if [[ "$mode" != --run-only ]]; then
  case "$(uname -s)" in
    Linux) ndk_host=linux-x86_64 ;;
    Darwin) ndk_host=darwin-x86_64 ;;
    *) echo 'Building requires a Linux or macOS NDK host' >&2; exit 1 ;;
  esac
  library_dir="${CARGO_TARGET_DIR:-target}/$rust_target/release"
  mkdir -p "$runtime_dir"
  "${ANDROID_NDK_HOME:?}/toolchains/llvm/prebuilt/$ndk_host/bin/${rust_target}21-clang" \
    -std=c11 -Wall -Wextra -Werror -I vanedb-capi/include \
    -Wl,-z,max-page-size=16384 -Wl,-z,common-page-size=16384 \
    vanedb-capi/tests/acceptance.c -L "$library_dir" \
    -lvanedb_capi -o "$runtime_dir/acceptance"
  cp "$library_dir/libvanedb_capi.so" "$runtime_dir/"
fi
python3 scripts/check_android_elf.py "$runtime_dir/acceptance" "$runtime_dir/libvanedb_capi.so"
[[ "$mode" != --build-only ]] || exit 0

device_abi=$(adb shell getprop ro.product.cpu.abi | tr -d '\r')
page_size=$(adb shell getconf PAGE_SIZE | tr -d '\r')
if [[ "$device_abi" != "$abi" ]]; then
  echo "Expected an $abi device, found $device_abi" >&2
  exit 1
fi
if [[ -n "${VANEDB_ANDROID_PAGE_SIZE:-}" && "$page_size" != "$VANEDB_ANDROID_PAGE_SIZE" ]]; then
  echo "Expected $VANEDB_ANDROID_PAGE_SIZE-byte pages, found $page_size" >&2
  exit 1
fi
echo "Android acceptance: $device_abi, $page_size-byte pages"

device_dir=$(adb shell mktemp -d /data/local/tmp/vanedb.XXXXXX | tr -d '\r')
# Only remove the directory created by this test, including if acceptance fails.
[[ "$device_dir" =~ ^/data/local/tmp/vanedb\.[a-zA-Z0-9]+$ ]]
trap 'adb shell rm -rf "$device_dir"' EXIT
adb push "$runtime_dir/acceptance" "$device_dir/acceptance"
adb push "$runtime_dir/libvanedb_capi.so" "$device_dir/"
adb shell chmod 700 "$device_dir/acceptance"
adb shell "LD_LIBRARY_PATH=$device_dir $device_dir/acceptance $device_dir"
