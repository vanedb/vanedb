#!/usr/bin/env bash
# Run after cargo-ndk builds the x86_64 C ABI and an emulator is ready.
set -euo pipefail
cd "$(dirname "$0")/.."

"${ANDROID_NDK_HOME:?}/toolchains/llvm/prebuilt/linux-x86_64/bin/x86_64-linux-android21-clang" \
  -std=c11 -Wall -Wextra -Werror -I vanedb-capi/include \
  vanedb-capi/tests/acceptance.c -L target/x86_64-linux-android/release \
  -lvanedb_capi -o target/android-acceptance

device_dir=$(adb shell mktemp -d /data/local/tmp/vanedb.XXXXXX | tr -d '\r')
# Only remove the directory created by this test, including if acceptance fails.
[[ "$device_dir" =~ ^/data/local/tmp/vanedb\.[a-zA-Z0-9]+$ ]]
trap 'adb shell rm -rf "$device_dir"' EXIT
adb push target/android-acceptance "$device_dir/acceptance"
adb push target/x86_64-linux-android/release/libvanedb_capi.so "$device_dir/"
adb shell chmod 700 "$device_dir/acceptance"
adb shell "LD_LIBRARY_PATH=$device_dir $device_dir/acceptance $device_dir"
