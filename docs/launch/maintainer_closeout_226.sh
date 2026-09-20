#!/usr/bin/env bash
# Remaining #226 / #198 closeout driver (AC3 host fill + AC5 demo cut).
#
# Reports gaps, then optionally runs the existing one-shots:
#   bash docs/launch/maintainer_closeout_226.sh           # status only
#   bash docs/launch/maintainer_closeout_226.sh --fill    # + Apple/Linux fill
#   bash docs/launch/maintainer_closeout_226.sh --cut     # + demo 0.2.0 cut
#   bash docs/launch/maintainer_closeout_226.sh --all     # fill then cut
#   bash docs/launch/maintainer_closeout_226.sh --cut --dry-run  # validate cut, no push
#
# AC3 Android still needs ANDROID.md (this driver refuses android labels).
# Cloud/CI shells are refused by the fill helper. Demo cut needs write or
# DEMO_REPO_TOKEN (or --dry-run to validate apply/tag only). Official AC5 URL
# must be obsidian-vane-search 0.2.0 (vanedb demo-0.2.0-staging is not enough).
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
      sed -n '1,17p' "$0"
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

demo_tag=""
if command -v gh >/dev/null 2>&1; then
  demo_tag="$(gh release list -R vanedb/obsidian-vane-search --limit 20 2>/dev/null \
    | awk -F'\t' '$1 == "0.2.0" || $3 == "0.2.0" {print "0.2.0"; exit}')" || true
fi

echo "==> #226 closeout status (repo=$VANEDB_ROOT)"
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

if [[ "$pending" -eq 0 && -n "$demo_tag" ]]; then
  echo "==> AC3 + AC5 look done. Reply on https://github.com/vanedb/vanedb/issues/226 with evidence."
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
    # Re-query after cut so the summary reflects a successful publish.
    if command -v gh >/dev/null 2>&1; then
      demo_tag="$(gh release list -R vanedb/obsidian-vane-search --limit 20 2>/dev/null \
        | awk -F'\t' '$1 == "0.2.0" || $3 == "0.2.0" {print "0.2.0"; exit}')" || true
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
  echo "    AC5: DEMO_REPO_TOKEN=… $0 --cut   or Actions → Cut demo 0.2.0"
fi
