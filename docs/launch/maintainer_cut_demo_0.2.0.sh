#!/usr/bin/env bash
# Cut the official obsidian-vane-search 0.2.0 release for vanedb#198 / AC5.
#
# Requires: git push access to vanedb/obsidian-vane-search, network, Node 20+
# (unless --skip-tests; the demo repo Release workflow still builds assets).
# Cloud agents without demo-repo write cannot push (403); use DEMO_REPO_TOKEN
# or --dry-run to validate apply/test/tag without pushing.
#
# Usage (from any clone of vanedb, or with VANEDB_ROOT set):
#   bash docs/launch/maintainer_cut_demo_0.2.0.sh
#   bash docs/launch/maintainer_cut_demo_0.2.0.sh --skip-tests
#   bash docs/launch/maintainer_cut_demo_0.2.0.sh --dry-run
#   DEMO_REPO_TOKEN=ghp_... bash docs/launch/maintainer_cut_demo_0.2.0.sh --skip-tests
#
# Or: Actions → "Cut demo 0.2.0" → Run workflow (needs repo secret DEMO_REPO_TOKEN).
set -euo pipefail

SKIP_TESTS=0
DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --skip-tests) SKIP_TESTS=1 ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help)
      sed -n '1,18p' "$0"
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

# Cursor cloud (and similar) inject GH_TOKEN scoped to vanedb only. Using that
# credential for a public HTTPS clone of obsidian-vane-search yields 403; strip
# ambient tokens unless DEMO_REPO_TOKEN explicitly authenticates the clone.
clone_demo() {
  local dest="$1"
  if [[ -n "${DEMO_REPO_TOKEN:-}" ]]; then
    git clone --depth 50 "$DEMO_URL" "$dest"
  else
    env -u GH_TOKEN -u GITHUB_TOKEN -u GH_ENTERPRISE_TOKEN \
      git clone --depth 50 "$DEMO_URL" "$dest"
  fi
}

if [[ ! -f "$PATCH" ]]; then
  echo "missing patch: $PATCH" >&2
  exit 1
fi

WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/vane-demo-0.2.0.XXXXXX")"
cleanup() { rm -rf "$WORKDIR"; }
trap cleanup EXIT

echo "==> clone $(redact_url "$DEMO_URL")"
clone_demo "$WORKDIR/repo"
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

BRANCH="release/$TAG-vanedb-198"
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "==> --dry-run: skipping push of $BRANCH and tag $TAG"
  echo "Local tag ready: $(git rev-parse "$TAG")"
  echo "Expected URL after a real cut: https://github.com/vanedb/obsidian-vane-search/releases/tag/$TAG"
  exit 0
fi

echo "==> push branch + tag (triggers .github/workflows/release.yml)"
git push -u origin "HEAD:refs/heads/$BRANCH"
git push origin "refs/tags/$TAG"

echo
echo "Pushed tag $TAG. Release workflow will attach main.js + manifest.json + LICENSE."
echo "Watch: https://github.com/vanedb/obsidian-vane-search/actions"
echo "Expected URL: https://github.com/vanedb/obsidian-vane-search/releases/tag/$TAG"
echo "Reply on https://github.com/vanedb/vanedb/issues/242 with that URL when green."
