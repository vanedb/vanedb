#!/usr/bin/env bash
# Record one publishable compare run on dedicated hardware.
# Usage: record_publish_run.sh <hw-label> <cosine|l2> [extra compare args...]
# Example: record_publish_run.sh linux-avx2 cosine
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
HW="${1:?hw label required (apple-*, linux-avx2*, android-arm64-*)}"
METRIC="${2:?metric required (cosine|l2)}"
shift 2
FIX="$ROOT/bench/compare/fixtures/embeddings.vnef"
if [[ ! -f "$FIX" ]]; then
  echo "missing $FIX — fetch or finalize the publish fixture first" >&2
  exit 1
fi
OUT_DIR="$ROOT/bench/compare/runs"
mkdir -p "$OUT_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$OUT_DIR/${HW}-${METRIC}-${STAMP}.json"
export VANEDB_COMPARE_HW="$HW"
export VANEDB_COMPARE_DEDICATED=1
# Refuse to run if this shell looks like CI/cloud (belt + suspenders).
if [[ "${CI:-}" == "true" || "${CI:-}" == "1" || "${GITHUB_ACTIONS:-}" == "true" || -n "${CURSOR_AGENT:-}" ]]; then
  echo "refusing record_publish_run.sh under CI/cloud env" >&2
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
