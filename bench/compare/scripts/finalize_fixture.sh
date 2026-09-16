#!/usr/bin/env bash
# Finalize a completed /tmp/vnef-full (or $1) generate_fixture.py run into
# bench/compare/fixtures/{metadata.json,SHA256SUMS} and optionally upload a
# GitHub Release asset. Does NOT git-add embeddings.vnef (gitignored).
set -euo pipefail
# scripts/ -> compare/
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${1:-/tmp/vnef-full}"
FIX="$ROOT/fixtures"
VNEF="$SRC/embeddings.vnef"
META="$SRC/metadata.json"
test -f "$VNEF" || { echo "missing $VNEF" >&2; exit 1; }
test -f "$META" || { echo "missing $META" >&2; exit 1; }
# Refuse incomplete files: header may claim 100k before the body is finished.
python3 - <<PY
import json, struct, pathlib, sys
p = pathlib.Path("$VNEF")
raw = p.read_bytes()
if raw[:4] != b"VNEF":
    sys.exit("bad magic")
if len(raw) < 28:
    sys.exit("too short")
ver, dim, n_docs, n_queries, _metric, _res = struct.unpack_from("<IIIIII", raw, 4)
need = 28 + (n_docs + n_queries) * dim * 4 + n_docs * 8
if len(raw) != need:
    sys.exit(f"incomplete vnef: size={len(raw)} need={need} n_docs={n_docs} n_queries={n_queries} dim={dim}")
if dim != 768:
    sys.exit(f"dim={dim} != 768 (RFC publish fixture)")
if n_docs < 100_000 or n_queries < 1000:
    sys.exit(f"too small for publish: n_docs={n_docs} n_queries={n_queries}")
meta = json.loads(pathlib.Path("$META").read_text())
if "pending" in meta.get("notes","").lower() or "pending" in meta.get("model","").lower():
    sys.exit("metadata still marked pending")
if int(meta["n_docs"]) != n_docs or int(meta["n_queries"]) != n_queries:
    sys.exit(f"metadata mismatch meta={meta['n_docs']}/{meta['n_queries']} file={n_docs}/{n_queries}")
if int(meta.get("dim", -1)) != dim:
    sys.exit(f"metadata dim mismatch meta={meta.get('dim')} file={dim}")
print(f"ok size={len(raw)} dim={dim} docs={n_docs} queries={n_queries}")
PY
cp "$META" "$FIX/metadata.json"
cp "$VNEF" "$FIX/embeddings.vnef"
# macOS ships shasum; Linux ships sha256sum (and often shasum via Perl).
if command -v sha256sum >/dev/null 2>&1; then
  HASH="$( (cd "$FIX" && sha256sum embeddings.vnef) )"
elif command -v shasum >/dev/null 2>&1; then
  HASH="$( (cd "$FIX" && shasum -a 256 embeddings.vnef) )"
else
  echo "need sha256sum or shasum to pin embeddings.vnef" >&2
  exit 1
fi
# Keep smoke line; replace or append embeddings line.
TMP="$(mktemp)"
if [[ -f "$FIX/SHA256SUMS" ]]; then
  grep -vE '(^|[[:space:]\*])embeddings\.vnef$' "$FIX/SHA256SUMS" >"$TMP" || true
else
  : >"$TMP"
fi
echo "$HASH" >>"$TMP"
mv "$TMP" "$FIX/SHA256SUMS"
echo "updated $FIX/metadata.json and SHA256SUMS:"
cat "$FIX/SHA256SUMS"
if [[ "${UPLOAD_RELEASE:-0}" == "1" ]]; then
  TAG="${COMPARE_FIXTURE_TAG:-compare-fixture-v1}"
  if gh release view "$TAG" -R vanedb/vanedb >/dev/null 2>&1; then
    gh release upload "$TAG" "$FIX/embeddings.vnef" -R vanedb/vanedb --clobber
  else
    gh release create "$TAG" "$FIX/embeddings.vnef" \
      -R vanedb/vanedb \
      --title "Competitor fixture (RFC 0003)" \
      --notes "100k×768 BeIR/nq + nomic-embed-text-v1.5; see bench/compare/fixtures/metadata.json" \
      --prerelease
  fi
  echo "uploaded release asset $TAG"
fi
echo "Next: git add metadata.json SHA256SUMS (+ docs URL); do NOT add embeddings.vnef"
