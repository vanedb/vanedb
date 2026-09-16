#!/usr/bin/env bash
# Record both cosine and L2 publish rows for one host HW label.
# Usage: record_both_metrics.sh <apple-*|linux-avx2*>
# Android: use ANDROID.md (this wrapper refuses android-*).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
HW="${1:?hw label required (apple-* or linux-avx2*)}"
shift || true
SCRIPT="$(cd "$(dirname "$0")" && pwd)/record_publish_run.sh"
bash "$SCRIPT" "$HW" cosine "$@"
bash "$SCRIPT" "$HW" l2 "$@"
echo "Both metrics recorded for $HW under bench/compare/runs/"
echo "Fill COMPARISON.md with: python3 bench/compare/scripts/fill_comparison_slot.py \\"
echo "  bench/compare/runs/${HW}-cosine-*.json bench/compare/runs/${HW}-l2-*.json"
