#!/usr/bin/env bash
# One-shot maintainer path for Apple/Linux dedicated HW (#198 AC3).
# Does NOT run Android (see ANDROID.md). Refuses cloud/CI shells.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

# Matches publish.rs::shared_runner_signals — GHA self-hosted without
# /opt/hostedtoolcache is operator-owned dedicated hardware (AC3 path).
# shellcheck source=refuse_shared_runner.sh
source "$(cd "$(dirname "$0")" && pwd)/refuse_shared_runner.sh"
compare_refuse_if_shared_runner || exit 1

URL="${VANEDB_COMPARE_FIXTURE_URL:-https://github.com/vanedb/vanedb/releases/download/compare-fixture-v1/embeddings.vnef}"
if [[ ! -f bench/compare/fixtures/embeddings.vnef ]]; then
  echo "fetching publish fixture…"
  VANEDB_COMPARE_FIXTURE_URL="$URL" bash bench/compare/scripts/fetch_fixture.sh
fi

echo "rebuilding compare (compile-time SHA256SUMS pin)…"
cargo build --release --locked --manifest-path bench/compare/Cargo.toml

HW=""
case "$(uname -s)/$(uname -m)" in
  Darwin/arm64|Darwin/aarch64) HW="${VANEDB_COMPARE_HW:-apple-silicon}" ;;
  Linux/*)
    if grep -qw avx2 /proc/cpuinfo 2>/dev/null; then
      HW="${VANEDB_COMPARE_HW:-linux-avx2}"
    else
      echo "refusing: Linux host has no avx2" >&2
      exit 1
    fi
    ;;
  *)
    echo "refusing: unsupported host $(uname -s)/$(uname -m); use ANDROID.md for Android" >&2
    exit 1
    ;;
esac

echo "recording $HW cosine + l2…"
bash bench/compare/scripts/record_both_metrics.sh "$HW"

COS_JSON="$(ls -1t bench/compare/runs/"${HW}"-cosine-*.json 2>/dev/null | head -n 1 || true)"
L2_JSON="$(ls -1t bench/compare/runs/"${HW}"-l2-*.json 2>/dev/null | head -n 1 || true)"
if [[ -n "$COS_JSON" && -n "$L2_JSON" ]]; then
  echo "filling bench/COMPARISON.md from $COS_JSON + $L2_JSON…"
  python3 bench/compare/scripts/fill_comparison_slot.py "$COS_JSON" "$L2_JSON"
  echo "Next: commit on a branch + open a PR (main is PR-protected); do not push tip."
  echo "  git switch -c bench/fill-\$(uname -s)-\$(date +%Y%m%d) &&"
  echo "  git add bench/COMPARISON.md bench/compare/runs/ &&"
  echo "  git commit -m 'bench(compare): fill COMPARISON (#226)' && git push -u origin HEAD"
else
  echo "Next: python3 bench/compare/scripts/fill_comparison_slot.py <cosine.json> <l2.json>"
  echo "Or paste --markdown output into bench/COMPARISON.md under matching headings."
fi
