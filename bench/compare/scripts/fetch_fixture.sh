#!/usr/bin/env bash
# Download the published embeddings.vnef and verify SHA256SUMS.
# Usage: scripts/fetch_fixture.sh [URL]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FIXTURES="$ROOT/fixtures"
URL="${1:-${VANEDB_COMPARE_FIXTURE_URL:-}}"
if [[ -z "$URL" ]]; then
  echo "Set VANEDB_COMPARE_FIXTURE_URL or pass the release-asset URL" >&2
  exit 1
fi
mkdir -p "$FIXTURES"
curl -fL --retry 4 --retry-delay 4 -o "$FIXTURES/embeddings.vnef" "$URL"
(cd "$FIXTURES" && shasum -a 256 -c SHA256SUMS)
echo "ok: $FIXTURES/embeddings.vnef"
