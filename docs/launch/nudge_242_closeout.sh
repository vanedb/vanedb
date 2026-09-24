#!/usr/bin/env bash
# Post a maintainer nudge on residual #242 (AC3/AC5 unlocks).
# Cloud agents often lack issues:write; Actions GITHUB_TOKEN can comment.
# Never shuts #242 / #198 / #226.
#
# Usage:
#   bash docs/launch/nudge_242_closeout.sh           # print comment body
#   bash docs/launch/nudge_242_closeout.sh --apply   # gh issue comment 242
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
APPLY=0
for arg in "$@"; do
  case "$arg" in
    --apply) APPLY=1 ;;
    -h|--help)
      sed -n '2,10p' "$0"
      exit 0
      ;;
    *)
      echo "unknown arg: $arg" >&2
      exit 2
      ;;
  esac
done

pending=0
if [[ -f "$ROOT/bench/COMPARISON.md" ]]; then
  pending="$(grep -c '\*Pending\.\*' "$ROOT/bench/COMPARISON.md" || true)"
fi
tip="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"

demo_state="unknown"
demo_mergeable=""
if command -v gh >/dev/null 2>&1; then
  if pr_json="$(gh pr view 20 -R vanedb/obsidian-vane-search --json state,mergeable 2>/dev/null)"; then
    demo_state="$(printf '%s' "$pr_json" | python3 -c 'import json,sys; print(json.load(sys.stdin)["state"])')"
    demo_mergeable="$(printf '%s' "$pr_json" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("mergeable") or "")')"
  fi
fi

token_line="DEMO_REPO_TOKEN: unset in this shell"
if [[ -n "${DEMO_REPO_TOKEN:-}" ]]; then
  token_line="DEMO_REPO_TOKEN: set in this shell"
fi

app_line="Cursor App: demo repo not in /installation/repositories (or unreadable)"
if command -v gh >/dev/null 2>&1; then
  if gh api --paginate /installation/repositories --jq '.repositories[].full_name' 2>/dev/null \
    | grep -Fxq 'vanedb/obsidian-vane-search'; then
    app_line="Cursor App: includes vanedb/obsidian-vane-search"
  elif gh api /installation/repositories >/dev/null 2>&1; then
    app_line="Cursor App: missing vanedb/obsidian-vane-search (only vanedb/vanedb or other)"
  fi
fi

body="$(cat <<EOF
### Closeout nudge (engine tip \`${tip}\`)

Residual acceptance still open:

- **AC3:** \`bench/COMPARISON.md\` has **${pending}×** \`*Pending.*\` (need 0). Fill on dedicated HW only (\`bash docs/launch/maintainer_closeout_226.sh --fill\` or Actions → **Fill COMPARISON (self-hosted)**; Android: \`bench/compare/ANDROID.md\`).
- **AC5:** official \`obsidian-vane-search\` **0.2.0** missing. Demo PR https://github.com/vanedb/obsidian-vane-search/pull/20 is **${demo_state}**${demo_mergeable:+ (mergeable=${demo_mergeable})}.

Maintainer unlocks needed:

1. Merge demo PR #20 with a **merge commit** (not squash).
2. \`bash docs/launch/maintainer_closeout_226.sh --tag --confirm-vault-walkthrough\` (or Actions → **Tag demo 0.2.0**).
3. Set repo secret \`DEMO_REPO_TOKEN\` (contents:write on the demo repo) **or** add \`vanedb/obsidian-vane-search\` to the Cursor GitHub App install.
4. Dedicated HW / self-hosted runners for the six COMPARISON slots.

Status probe: ${token_line}; ${app_line}.

How on this issue should already show \`--tag --confirm-vault-walkthrough\` (not historical \`--cut\`). Refresh via Actions → **Sync #242 How** if needed.

Leave this issue open until Pending=0 **and** the official 0.2.0 release URL is recorded.
EOF
)"

if [[ "$APPLY" -eq 0 ]]; then
  printf '%s\n' "$body"
  exit 0
fi

if ! command -v gh >/dev/null 2>&1; then
  echo "refused: gh not available for --apply" >&2
  exit 1
fi

tmp="$(mktemp)"
printf '%s\n' "$body" >"$tmp"
gh issue comment 242 --repo vanedb/vanedb --body-file "$tmp"
rm -f "$tmp"
echo "==> commented on https://github.com/vanedb/vanedb/issues/242"
