#!/usr/bin/env bash
# Record one publishable compare run on dedicated hardware.
# Usage: record_publish_run.sh <hw-label> <cosine|l2> [extra compare args...]
# Example: record_publish_run.sh linux-avx2 cosine
# Android ARM64 must use ANDROID.md (adb/NDK on-device) — this wrapper refuses
# android-* labels so host timings cannot be mislabelled.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
HW="${1:?hw label required (apple-* or linux-avx2*)}"
METRIC="${2:?metric required (cosine|l2)}"
shift 2
FIX="$ROOT/bench/compare/fixtures/embeddings.vnef"
if [[ ! -f "$FIX" ]]; then
  echo "missing $FIX — fetch or finalize the publish fixture first" >&2
  echo "see docs/launch/0003-fixture-hosting.md" >&2
  exit 1
fi

case "$HW" in
  android-*)
    echo "refusing android-* in record_publish_run.sh — use bench/compare/ANDROID.md \
(adb/NDK on-device or labelled emulator). Host cargo run timings are not Android." >&2
    exit 1
    ;;
  apple-*)
    if [[ "$(uname -s)" != "Darwin" ]]; then
      echo "refusing apple-* on $(uname -s); need Darwin Apple Silicon host" >&2
      exit 1
    fi
    arch="$(uname -m)"
    if [[ "$arch" != "arm64" && "$arch" != "aarch64" ]]; then
      echo "refusing apple-* on arch=$arch; need Apple Silicon (arm64)" >&2
      exit 1
    fi
    ;;
  linux-avx2*)
    if [[ "$(uname -s)" != "Linux" ]]; then
      echo "refusing linux-avx2* on $(uname -s)" >&2
      exit 1
    fi
    if [[ ! -r /proc/cpuinfo ]]; then
      echo "refusing linux-avx2*: cannot read /proc/cpuinfo to verify AVX2" >&2
      exit 1
    fi
    if ! grep -qw avx2 /proc/cpuinfo; then
      echo "refusing linux-avx2*: /proc/cpuinfo has no avx2 flag" >&2
      exit 1
    fi
    ;;
  *)
    echo "hw label must start with apple- or linux-avx2- (android via ANDROID.md)" >&2
    exit 1
    ;;
esac

OUT_DIR="$ROOT/bench/compare/runs"
mkdir -p "$OUT_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$OUT_DIR/${HW}-${METRIC}-${STAMP}.json"
export VANEDB_COMPARE_HW="$HW"
export VANEDB_COMPARE_DEDICATED=1
# Refuse to run if this shell looks like CI/cloud (belt + suspenders).
# FS markers survive `env -u CURSOR_AGENT` on Cursor cloud VMs.
if [[ "${CI:-}" == "true" || "${CI:-}" == "1" \
   || "${GITHUB_ACTIONS:-}" == "true" || "${GITHUB_ACTIONS:-}" == "1" \
   || "${GITLAB_CI:-}" == "true" || "${CIRCLECI:-}" == "true" \
   || -n "${CURSOR_AGENT:-}" || -n "${CODESPACES:-}" \
   || -e /opt/cursor || -e /exec-daemon ]]; then
  echo "refusing record_publish_run.sh under CI/cloud env (or Cursor cloud FS markers)" >&2
  exit 1
fi
cd "$ROOT"
cargo run --release --locked --manifest-path bench/compare/Cargo.toml -- run \
  --fixture "$FIX" \
  --metric "$METRIC" \
  --rounds 4 \
  --markdown \
  --json-out "$OUT" \
  "$@"
echo "wrote $OUT"
echo "Re-render later with: python3 bench/compare/scripts/render_comparison_md.py $OUT"
