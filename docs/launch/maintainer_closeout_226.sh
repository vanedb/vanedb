#!/usr/bin/env bash
# Remaining #198 AC3/AC5 closeout driver (host fill + demo cut).
# Residual tracking issue: https://github.com/vanedb/vanedb/issues/242
# (#226 was auto-closed by a merge keyword; do not use close/fix/#N in PRs.)
#
# Reports gaps, then optionally runs the existing one-shots:
#   bash docs/launch/maintainer_closeout_226.sh           # status only
#   bash docs/launch/maintainer_closeout_226.sh --fill    # + Apple/Linux fill
#   bash docs/launch/maintainer_closeout_226.sh --cut     # + demo 0.2.0 cut
#   bash docs/launch/maintainer_closeout_226.sh --all     # fill then cut
#   bash docs/launch/maintainer_closeout_226.sh --cut --dry-run  # validate cut, no push
#
# AC3 Android still needs ANDROID.md (this driver refuses android labels).
# Cloud/CI shells are refused by the fill helper. Demo cut needs write via
# DEMO_REPO_TOKEN, or Cursor GitHub App on vanedb/obsidian-vane-search
# (contents:write) plus a new agent boot after #215 repositoryDependencies
# (cut auto-uses `gh auth token` when /installation/repositories includes the
# demo repo; or --dry-run to validate apply/tag only). Official AC5 URL must be
# obsidian-vane-search 0.2.0 (vanedb demo-0.2.0-staging is not enough).
set -euo pipefail

DO_FILL=0
DO_CUT=0
SKIP_TESTS=0
DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --fill) DO_FILL=1 ;;
    --cut) DO_CUT=1 ;;
    --all) DO_FILL=1; DO_CUT=1 ;;
    --skip-tests) SKIP_TESTS=1 ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help)
      sed -n '1,19p' "$0"
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
cd "$VANEDB_ROOT"

pending=0
if [[ -f bench/COMPARISON.md ]]; then
  pending="$(grep -c '\*Pending\.\*' bench/COMPARISON.md || true)"
fi

# Release list can lag the tag push (demo release workflow is async). Prefer
# an explicit release view, then the git tag ref.
demo_tag=""
if command -v gh >/dev/null 2>&1; then
  if gh release view 0.2.0 -R vanedb/obsidian-vane-search >/dev/null 2>&1; then
    demo_tag="0.2.0"
  elif gh api repos/vanedb/obsidian-vane-search/git/ref/tags/0.2.0 >/dev/null 2>&1; then
    demo_tag="0.2.0"
  fi
fi

# Paginate: default page size can miss the demo repo when many are installed.
app_demo_in_scope() {
  command -v gh >/dev/null 2>&1 || return 1
  gh api --paginate /installation/repositories --jq '.repositories[].full_name' \
    2>/dev/null | grep -Fxq 'vanedb/obsidian-vane-search'
}

echo "==> #242 closeout status (repo=$VANEDB_ROOT)"
echo "    COMPARISON Pending cells: $pending (need 0)"
if [[ -n "$demo_tag" ]]; then
  echo "    Official demo 0.2.0: present"
  echo "    URL: https://github.com/vanedb/obsidian-vane-search/releases/tag/0.2.0"
else
  echo "    Official demo 0.2.0: MISSING (staging on vanedb does not count)"
fi
if [[ -n "${DEMO_REPO_TOKEN:-}" ]]; then
  echo "    DEMO_REPO_TOKEN: set"
else
  echo "    DEMO_REPO_TOKEN: unset"
fi
# App path: cut auto-uses `gh auth token` only when the demo repo is in scope.
if app_demo_in_scope; then
  echo "    Cursor App scope: includes obsidian-vane-search (--cut can use App token)"
elif command -v gh >/dev/null 2>&1 && gh api /installation/repositories >/dev/null 2>&1; then
  echo "    Cursor App scope: missing obsidian-vane-search (install App or set DEMO_REPO_TOKEN)"
fi
if [[ -e /opt/cursor || -e /exec-daemon || -e /opt/hostedtoolcache ]]; then
  echo "    host: shared runner markers present (AC3 --fill will refuse)"
else
  echo "    host: no shared-runner FS markers (fill may proceed if dedicated)"
fi

if [[ "$pending" -eq 0 && -n "$demo_tag" ]]; then
  echo "==> AC3 + AC5 look done. Reply on https://github.com/vanedb/vanedb/issues/242 with evidence."
  exit 0
fi

if [[ "$DO_FILL" -eq 1 ]]; then
  if [[ "$pending" -eq 0 ]]; then
    echo "==> --fill skipped: no Pending cells"
  else
    echo "==> --fill: maintainer_fill_host_comparison.sh (Apple/Linux dedicated only)"
    bash bench/compare/scripts/maintainer_fill_host_comparison.sh
    echo "    Open a PR with COMPARISON.md + runs/ (main is PR-protected)."
  fi
fi

if [[ "$DO_CUT" -eq 1 ]]; then
  if [[ -n "$demo_tag" ]]; then
    echo "==> --cut skipped: 0.2.0 already exists"
  else
    echo "==> --cut: maintainer_cut_demo_0.2.0.sh"
    cut_args=()
    if [[ "$SKIP_TESTS" -eq 1 ]]; then
      cut_args+=(--skip-tests)
    fi
    if [[ "$DRY_RUN" -eq 1 ]]; then
      cut_args+=(--dry-run)
    fi
    bash docs/launch/maintainer_cut_demo_0.2.0.sh "${cut_args[@]}"
    # Re-query after cut so the summary reflects a successful publish (tag
    # may land before the release workflow finishes).
    if command -v gh >/dev/null 2>&1; then
      if gh release view 0.2.0 -R vanedb/obsidian-vane-search >/dev/null 2>&1 \
        || gh api repos/vanedb/obsidian-vane-search/git/ref/tags/0.2.0 >/dev/null 2>&1; then
        demo_tag="0.2.0"
      else
        demo_tag=""
      fi
    fi
    if [[ -z "$demo_tag" && "$DRY_RUN" -eq 1 ]]; then
      echo "    (dry-run: official 0.2.0 still missing until a real cut)"
    fi
  fi
fi

pending_after="$pending"
if [[ -f bench/COMPARISON.md ]]; then
  pending_after="$(grep -c '\*Pending\.\*' bench/COMPARISON.md || true)"
fi
echo "==> remaining: Pending=$pending_after; demo_0.2.0=${demo_tag:-missing}"
if [[ "$pending_after" -ne 0 ]]; then
  echo "    AC3: fill Apple/Linux here or via Actions → Fill COMPARISON (self-hosted);"
  echo "         Android: bench/compare/ANDROID.md"
fi
if [[ -z "$demo_tag" ]]; then
  echo "    AC5: DEMO_REPO_TOKEN=… $0 --cut"
  echo "         or Cursor App on vanedb/obsidian-vane-search (contents:write) + new agent + $0 --cut"
  echo "         or Actions → Cut demo 0.2.0 (repo secret DEMO_REPO_TOKEN)"
fi
