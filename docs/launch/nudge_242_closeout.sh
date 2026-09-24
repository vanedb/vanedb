#!/usr/bin/env bash
# Post a maintainer nudge on residual #242 (AC3/AC5 unlocks).
# Cloud agents often lack issues:write; Actions GITHUB_TOKEN can comment.
# Never shuts #242 / #198 / #226.
# When --apply and residuals remain, reopens #242 and #198 if shut early.
#
# Usage:
#   bash docs/launch/nudge_242_closeout.sh           # print comment body
#   bash docs/launch/nudge_242_closeout.sh --apply   # reopen if needed + comment
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
APPLY=0
for arg in "$@"; do
  case "$arg" in
    --apply) APPLY=1 ;;
    -h|--help)
      sed -n '2,11p' "$0"
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
demo_vault=""
official_020="missing"
if command -v gh >/dev/null 2>&1; then
  if pr_json="$(gh pr view 20 -R vanedb/obsidian-vane-search --json state,mergeable,headRefOid 2>/dev/null)"; then
    demo_state="$(printf '%s' "$pr_json" | python3 -c 'import json,sys; print(json.load(sys.stdin)["state"])')"
    demo_mergeable="$(printf '%s' "$pr_json" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("mergeable") or "")')"
    demo_head="$(printf '%s' "$pr_json" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("headRefOid") or "")')"
    if [[ -n "$demo_head" ]]; then
      if gh api "repos/vanedb/obsidian-vane-search/contents/docs/releases/0.2.0-desktop-acceptance.md?ref=${demo_head}" \
          >/dev/null 2>&1; then
        demo_vault="recorded"
      else
        demo_vault="missing"
      fi
    fi
  fi
  if gh release view 0.2.0 -R vanedb/obsidian-vane-search >/dev/null 2>&1; then
    official_020="present"
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

ac5_line="official \`obsidian-vane-search\` **0.2.0** missing. Demo PR https://github.com/vanedb/obsidian-vane-search/pull/20 is **${demo_state}**${demo_mergeable:+ (mergeable=${demo_mergeable})}."
if [[ "$official_020" == "present" ]]; then
  ac5_line="official \`obsidian-vane-search\` **0.2.0** release is live. Demo PR https://github.com/vanedb/obsidian-vane-search/pull/20 is **${demo_state}**${demo_mergeable:+ (mergeable=${demo_mergeable})}."
fi
if [[ "$demo_vault" == "recorded" ]]; then
  ac5_line="${ac5_line} Vault acceptance **recorded** on PR head."
elif [[ "$demo_vault" == "missing" ]]; then
  ac5_line="${ac5_line} Vault acceptance **missing** on PR head."
fi

residuals=0
if [[ "$pending" -gt 0 || "$official_020" != "present" ]]; then
  residuals=1
fi

# Emit via a function so the heredoc is not nested inside body="$(…)" —
# macOS /bin/bash mis-parses \`…\` + ")" inside that form (bad substitution).
nudge_body() {
  cat <<EOF
### Closeout nudge (engine tip \`${tip}\`)

Residual acceptance still open:

- **AC3:** \`bench/COMPARISON.md\` has **${pending}×** \`*Pending.*\` (need 0). Fill on dedicated HW only (\`bash docs/launch/maintainer_closeout_226.sh --fill\` or Actions → **Fill COMPARISON (self-hosted)**; Android: \`bench/compare/ANDROID.md\`).
- **AC5:** ${ac5_line}

Maintainer unlocks needed:

1. Set repo secret \`DEMO_REPO_TOKEN\` (contents:write **and** pull_requests:write on the demo repo) **or** add \`vanedb/obsidian-vane-search\` to the Cursor GitHub App install.
2. Actions → **Tag demo 0.2.0** with \`confirm_vault_walkthrough=true\` (passes \`--merge-if-open\`: merge commit on #20 if still OPEN, then annotated tag) — or \`bash docs/launch/maintainer_closeout_226.sh --tag --confirm-vault-walkthrough\`.
   If the merge starts fresh CI, rerun after that merged commit is green. A changed bundle requires a new desktop walkthrough before tagging.
3. Dedicated HW / self-hosted runners for the six COMPARISON slots.

Status probe: ${token_line}; ${app_line}.

How on this issue should already show \`--tag --confirm-vault-walkthrough\` (not historical \`--cut\`). Refresh via Actions → **Sync #242 How** if needed.

Leave #242 (and #198) open until Pending=0 **and** the official 0.2.0 release URL is recorded. If either was shut early, \`nudge_242_closeout.sh --apply\` reopens them while residuals remain.
EOF
}

body="$(nudge_body)"

if [[ "$APPLY" -eq 0 ]]; then
  printf '%s\n' "$body"
  exit 0
fi

if ! command -v gh >/dev/null 2>&1; then
  echo "refused: gh not available for --apply" >&2
  exit 1
fi

if [[ "$residuals" -eq 0 ]]; then
  echo "==> residuals cleared (Pending=0 and official 0.2.0 present); skip reopen/nudge"
  exit 0
fi

# Reopen residual trackers while AC3/AC5 evidence is still missing.
# Do not put GitHub merge-closing verbs next to issue numbers in commits.
for issue in 242 198; do
  issue_state="$(gh issue view "$issue" --repo vanedb/vanedb --json state -q .state 2>/dev/null || echo unknown)"
  if [[ "$issue_state" == "CLOSED" ]]; then
    gh issue reopen "$issue" --repo vanedb/vanedb \
      --comment "Auto-reopened: residual AC3/AC5 still open (Pending=${pending}; official 0.2.0 ${official_020})."
    echo "==> reopened https://github.com/vanedb/vanedb/issues/${issue} (was CLOSED with residuals)"
  fi
done

tmp="$(mktemp)"
printf '%s\n' "$body" >"$tmp"
gh issue comment 242 --repo vanedb/vanedb --body-file "$tmp"
rm -f "$tmp"
echo "==> commented on https://github.com/vanedb/vanedb/issues/242"
