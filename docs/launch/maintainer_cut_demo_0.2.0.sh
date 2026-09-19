#!/usr/bin/env bash
# Cut the official obsidian-vane-search 0.2.0 release for vanedb#198 / AC5.
#
# Requires: git push access to vanedb/obsidian-vane-search, network, Node 20+
# (unless --skip-tests; the demo repo Release workflow still builds assets).
# Cloud agents without demo-repo write cannot run this successfully (403).
#
# Usage (from any clone of vanedb, or with VANEDB_ROOT set):
#   bash docs/launch/maintainer_cut_demo_0.2.0.sh
#   bash docs/launch/maintainer_cut_demo_0.2.0.sh --skip-tests
#   DEMO_REPO_TOKEN=ghp_... bash docs/launch/maintainer_cut_demo_0.2.0.sh --skip-tests
#
# Or: Actions → "Cut demo 0.2.0" → Run workflow (needs repo secret DEMO_REPO_TOKEN).
set -euo pipefail

SKIP_TESTS=0
for arg in "$@"; do
  case "$arg" in
    --skip-tests) SKIP_TESTS=1 ;;
    -h|--help)
      sed -n '1,16p' "$0"
      exit 0
      ;;
    *)
      echo "unknown arg: $arg" >&2
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VANEDB_ROOT="${VANEDB_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PATCH="$VANEDB_ROOT/docs/launch/0003-obsidian-vane-search-0.2.0.patch"
TAG="0.2.0"

if [[ -n "${DEMO_REPO_TOKEN:-}" ]]; then
  DEMO_URL="https://x-access-token:${DEMO_REPO_TOKEN}@github.com/vanedb/obsidian-vane-search.git"
elif [[ -z "${DEMO_URL:-}" ]]; then
  DEMO_URL="https://github.com/vanedb/obsidian-vane-search.git"
fi

redact_url() {
  sed -E 's#://[^/@]+@#://***@#g' <<<"$1"
}

if [[ ! -f "$PATCH" ]]; then
  echo "missing patch: $PATCH" >&2
  exit 1
fi

WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/vane-demo-0.2.0.XXXXXX")"
cleanup() { rm -rf "$WORKDIR"; }
trap cleanup EXIT

echo "==> clone $(redact_url "$DEMO_URL")"
git clone --depth 50 "$DEMO_URL" "$WORKDIR/repo"
cd "$WORKDIR/repo"

if ! git config user.email >/dev/null; then
  git config user.email "maintainers@vanedb.dev"
fi
if ! git config user.name >/dev/null; then
  git config user.name "vanedb maintainers"
fi

if git rev-parse "refs/tags/$TAG" >/dev/null 2>&1; then
  echo "tag $TAG already exists locally after clone; aborting" >&2
  exit 1
fi
if git ls-remote --tags origin "refs/tags/$TAG" | grep -q .; then
  echo "tag $TAG already exists on origin; aborting" >&2
  exit 1
fi

echo "==> apply $PATCH"
git apply "$PATCH"

if [[ "$SKIP_TESTS" -eq 0 ]]; then
  echo "==> npm ci && npm test && npm run build"
  npm ci
  npm test
  npm run build
else
  echo "==> --skip-tests: relying on demo repo release workflow for build"
fi

echo "==> commit + tag $TAG"
git add README.md manifest.json package.json versions.json
git commit -m "chore(release): 0.2.0 demo slice for vanedb#198"
git tag -a "$TAG" -m "obsidian-vane-search $TAG (vanedb#198)"

echo "==> push branch + tag (triggers .github/workflows/release.yml)"
BRANCH="release/$TAG-vanedb-198"
git push -u origin "HEAD:refs/heads/$BRANCH"
git push origin "refs/tags/$TAG"

echo
echo "Pushed tag $TAG. Release workflow will attach main.js + manifest.json + LICENSE."
echo "Watch: https://github.com/vanedb/obsidian-vane-search/actions"
echo "Expected URL: https://github.com/vanedb/obsidian-vane-search/releases/tag/$TAG"
echo "Reply on https://github.com/vanedb/vanedb/issues/226 with that URL when green."
