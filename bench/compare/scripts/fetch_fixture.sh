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
if [[ ! -f "$FIXTURES/SHA256SUMS" ]]; then
  echo "missing $FIXTURES/SHA256SUMS — commit the sums file before fetching" >&2
  exit 1
fi
if ! grep -E '(^| )embeddings\.vnef$' "$FIXTURES/SHA256SUMS" >/dev/null; then
  echo "embeddings.vnef is not listed in SHA256SUMS" >&2
  exit 1
fi
# Verify only the embeddings line so a missing smoke file in this checkout is fine.
awk '$2=="embeddings.vnef" || $2=="*embeddings.vnef" {print}' "$FIXTURES/SHA256SUMS" \
  | (cd "$FIXTURES" && shasum -a 256 -c -)
echo "ok: $FIXTURES/embeddings.vnef"
