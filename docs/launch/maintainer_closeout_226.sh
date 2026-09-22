#!/usr/bin/env bash
# Remaining #198 AC3/AC5 closeout driver (host fill + AC5 status).
# Residual tracking issue: https://github.com/vanedb/vanedb/issues/242
# (#226 was auto-closed by a merge keyword; do not use close/fix/#N in PRs.)
#
# Reports gaps, then optionally runs fill / annotated tag helpers:
#   bash docs/launch/maintainer_closeout_226.sh           # status only
#   bash docs/launch/maintainer_closeout_226.sh --fill    # + Apple/Linux fill
#   bash docs/launch/maintainer_closeout_226.sh --tag \
#     --confirm-vault-walkthrough [--dry-run]             # + AC5 annotated tag
#
# AC5 for the current candidate is NOT `--cut`. Historical
# `maintainer_cut_demo_0.2.0.sh` / Actions → Cut demo 0.2.0 apply an older
# patch. Use the reviewed-PR sequence instead:
#   https://github.com/vanedb/obsidian-vane-search/pull/20
#   docs/launch/0003-demo-update-checklist.md
# (independent review + CI → real Obsidian/Ollama vault walkthrough → merge →
# `maintainer_tag_demo_0.2.0.sh` annotated 0.2.0 tag). Tag push still needs
# DEMO_REPO_TOKEN or Cursor GitHub App on vanedb/obsidian-vane-search
# (contents:write) after #215 repositoryDependencies.
#
# Legacy flags (refused for this candidate; exit 2):
#   --cut / --all / --skip-tests (historical cut-only options)
#
# AC3 Android still needs ANDROID.md (this driver refuses android labels).
# Cloud/CI shells are refused by the fill helper. Official AC5 URL must be
# obsidian-vane-search 0.2.0 (vanedb demo-0.2.0-staging is not enough).
set -euo pipefail

DO_FILL=0
DO_CUT=0
DO_TAG=0
CONFIRM_VAULT=0
DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --fill) DO_FILL=1 ;;
    --cut) DO_CUT=1 ;;
    --tag) DO_TAG=1 ;;
    --all) DO_FILL=1; DO_CUT=1 ;;
    --confirm-vault-walkthrough) CONFIRM_VAULT=1 ;;
    --skip-tests)
      echo "unknown arg for current AC5 path: --skip-tests (historical cut only)" >&2
      exit 2
      ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help)
      sed -n '1,28p' "$0"
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
# App path: needed to push the annotated 0.2.0 tag after demo PR #20 + vault.
if app_demo_in_scope; then
  echo "    Cursor App scope: includes obsidian-vane-search (can push annotated 0.2.0 tag)"
elif command -v gh >/dev/null 2>&1 && gh api /installation/repositories >/dev/null 2>&1; then
  echo "    Cursor App scope: missing obsidian-vane-search (install App or set DEMO_REPO_TOKEN)"
  # vanedb org id 272005268 — Configure → add obsidian-vane-search (contents:write)
  echo "    App configure: https://github.com/apps/cursor/installations/new/permissions?target_id=272005268"
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

exit_code=0

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
  # Historical patch cut is not the current AC5 candidate (see #271 /
  # 0003-demo-update-checklist.md). Keep --cut/--all parseable so old
  # muscle-memory fails loudly instead of publishing the wrong tag.
  echo "==> --cut refused: historical patch cut is not the current AC5 candidate" >&2
  echo "    Current candidate: https://github.com/vanedb/obsidian-vane-search/pull/20" >&2
  echo "    Sequence: review+CI → real vault walkthrough → merge → --tag" >&2
  echo "    Checklist: docs/launch/0003-demo-update-checklist.md" >&2
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "    (--dry-run does not re-enable historical cut)" >&2
  fi
  exit_code=2
fi

if [[ "$DO_TAG" -eq 1 ]]; then
  if [[ -n "$demo_tag" ]]; then
    echo "==> --tag skipped: 0.2.0 already exists"
  else
    echo "==> --tag: maintainer_tag_demo_0.2.0.sh"
    tag_args=()
    if [[ "$CONFIRM_VAULT" -eq 1 ]]; then
      tag_args+=(--confirm-vault-walkthrough)
    fi
    if [[ "$DRY_RUN" -eq 1 ]]; then
      tag_args+=(--dry-run)
    fi
    if bash docs/launch/maintainer_tag_demo_0.2.0.sh "${tag_args[@]}"; then
      if command -v gh >/dev/null 2>&1; then
        if gh release view 0.2.0 -R vanedb/obsidian-vane-search >/dev/null 2>&1 \
          || gh api repos/vanedb/obsidian-vane-search/git/ref/tags/0.2.0 >/dev/null 2>&1; then
          demo_tag="0.2.0"
        fi
      fi
      if [[ -z "$demo_tag" && "$DRY_RUN" -eq 1 ]]; then
        echo "    (dry-run: official 0.2.0 still missing until a real tag push)"
      fi
    else
      exit_code=2
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
  echo "         Fill workflow: https://github.com/vanedb/vanedb/actions/workflows/fill-comparison-self-hosted.yml"
fi
if [[ -z "$demo_tag" ]]; then
  echo "    AC5: demo PR https://github.com/vanedb/obsidian-vane-search/pull/20"
  echo "         → vault walkthrough → merge → $0 --tag --confirm-vault-walkthrough"
  echo "         (not historical --cut; helper: docs/launch/maintainer_tag_demo_0.2.0.sh)"
  echo "         Checklist: docs/launch/0003-demo-update-checklist.md"
  echo "         Tag push: DEMO_REPO_TOKEN or Cursor App on vanedb/obsidian-vane-search"
fi

exit "$exit_code"
