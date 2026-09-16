#!/usr/bin/env bash
# One-shot maintainer path for Apple/Linux dedicated HW (#198 AC3).
# Does NOT run Android (see ANDROID.md). Refuses cloud/CI shells.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

if [[ -e /opt/cursor || -e /exec-daemon || -n "${CURSOR_AGENT:-}" || -n "${CODESPACES:-}" \
   || "${CI:-}" == "true" || "${CI:-}" == "1" || "${GITHUB_ACTIONS:-}" == "true" ]]; then
  echo "refusing: this is a CI/cloud shell — run on idle dedicated hardware" >&2
  exit 1
fi

URL="${VANEDB_COMPARE_FIXTURE_URL:-https://github.com/vanedb/vanedb/releases/download/compare-fixture-v1/embeddings.vnef}"
if [[ ! -f bench/compare/fixtures/embeddings.vnef ]]; then
  echo "fetching publish fixture…"
  VANEDB_COMPARE_FIXTURE_URL="$URL" bash bench/compare/scripts/fetch_fixture.sh
fi

echo "rebuilding compare (compile-time SHA256SUMS pin)…"
cargo build --release --locked --manifest-path bench/compare/Cargo.toml

HW=""
case "$(uname -s)/$(uname -m)" in
  Darwin/arm64|Darwin/aarch64) HW="${VANEDB_COMPARE_HW:-apple-m4-pro}" ;;
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
echo
echo "Next: paste the printed markdown into bench/COMPARISON.md under the matching"
echo "Cosine / Squared L2 headings for this HW class, commit, and push."
