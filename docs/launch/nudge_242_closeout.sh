#!/usr/bin/env bash
# Maintain one status comment on residual #242 (AC3/AC5 unlocks).
# Cloud agents often lack issues:write; Actions GITHUB_TOKEN can comment.
# Never shuts #242 / #198 / #226.
# When --apply and residuals remain, reopens #242 and #198 if shut early.
# Manual mode uses gh's authenticated viewer; Actions uses github-actions[bot].
# Each emitter maintains only its own marked comment. Run manual applies serially.
#
# Usage:
#   bash docs/launch/nudge_242_closeout.sh           # print comment body
#   bash docs/launch/nudge_242_closeout.sh --apply   # reopen + create/update if changed
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

# A missing or unreadable comparison is unknown, never zero or a positive count.
pending="$(python3 - "$ROOT/bench/COMPARISON.md" <<'PYPENDING'
from pathlib import Path
import sys
try:
    print(sum("*Pending.*" in line for line in Path(sys.argv[1]).read_text().splitlines()))
except (OSError, UnicodeError):
    print("unknown")
PYPENDING
)"
tip="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"

# The demo is public, but this repository's Actions token may not have access.
# Probe anonymously: authenticated 404s can mean permission failure. Disable
# curlrc so ambient credentials cannot turn this into an authenticated request.
public_api() {
  local response
  public_status=unknown
  public_body=''
  if response="$(curl --disable --silent --show-error --location \
      --connect-timeout 5 --max-time 20 --header 'Accept: application/vnd.github+json' \
      --write-out $'\n%{http_code}' "https://api.github.com/$1" 2>/dev/null)"; then
    public_status="${response##*$'\n'}"
    public_body="${response%$'\n'*}"
  fi
}

probe_resource() {
  public_api "$1"
  if [[ "$public_status" == "200" ]]; then
    echo present
  elif [[ "$public_status" == "404" && "$demo_repo_access" == "present" ]]; then
    echo missing
  else
    echo unknown
  fi
}

demo_state="unknown"
demo_mergeable=""
demo_vault="unknown"
official_020="unknown"
demo_repo_access="unknown"
if command -v curl >/dev/null 2>&1; then
  public_api repos/vanedb/obsidian-vane-search
  if [[ "$public_status" == "200" ]]; then demo_repo_access="present"; fi
  public_api repos/vanedb/obsidian-vane-search/pulls/20
  if [[ "$public_status" == "200" ]]; then
    pr_info="$(printf '%s' "$public_body" | python3 -c '
import json, re, sys
try:
    pr = json.load(sys.stdin)
    assert pr["state"] in ("open", "closed")
    assert re.fullmatch(r"[0-9a-f]{40}", pr["head"]["sha"])
    state = "MERGED" if pr.get("merged_at") else pr["state"].upper()
    mergeable = {True: "MERGEABLE", False: "CONFLICTING", None: "UNKNOWN"}[pr.get("mergeable")]
    print("|".join((state, mergeable, pr["head"]["sha"])))
except (ValueError, TypeError, KeyError, AssertionError):
    print("unknown||")
')"
    IFS='|' read -r demo_state demo_mergeable demo_head <<< "$pr_info"
    if [[ -n "$demo_head" ]]; then
      demo_vault="$(probe_resource "repos/vanedb/obsidian-vane-search/contents/docs/releases/0.2.0-desktop-acceptance.md?ref=${demo_head}")"
      if [[ "$demo_vault" == "present" ]]; then demo_vault="recorded"; fi
    fi
  fi
  official_020="$(probe_resource repos/vanedb/obsidian-vane-search/releases/tags/0.2.0)"
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

ac5_line="official \`obsidian-vane-search\` **0.2.0** missing."
if [[ "$official_020" == "unknown" ]]; then
  ac5_line="official \`obsidian-vane-search\` **0.2.0** status unknown (release API unreadable); retry the probe."
elif [[ "$official_020" == "present" ]]; then
  ac5_line="official \`obsidian-vane-search\` **0.2.0** release is live."
fi
# Keep the candidate link visible even when the independent release probe fails.
ac5_line="${ac5_line} Demo PR https://github.com/vanedb/obsidian-vane-search/pull/20 is **${demo_state}**${demo_mergeable:+ (mergeable=${demo_mergeable})}."
if [[ "$demo_vault" == "recorded" ]]; then
  ac5_line="${ac5_line} Vault acceptance **recorded** on PR head."
elif [[ "$demo_vault" == "missing" ]]; then
  ac5_line="${ac5_line} Vault acceptance **missing** on PR head."
fi

residuals=0
if [[ ( "$pending" != "unknown" && "$pending" -gt 0 ) || "$official_020" == "missing" ]]; then
  residuals=1
fi
probes_known=1
if [[ "$pending" == "unknown" || "$official_020" == "unknown" || "$demo_state" == "unknown" || "$demo_vault" == "unknown" ]]; then
  probes_known=0
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

# Fingerprint acceptance state only. Tip, mergeability and credential probes can
# change every scheduled run without changing the work needed for closeout.
marker='<!-- vanedb-closeout-242:v1 -->'
fingerprint="$(python3 - "$pending" "$official_020" "$demo_state" "$demo_vault" <<'PYHASH'
import hashlib, json, sys
print(hashlib.sha256(json.dumps(sys.argv[1:]).encode()).hexdigest())
PYHASH
)"
body="${marker}
<!-- acceptance-state:${fingerprint} -->
$(nudge_body)"

if [[ "$APPLY" -eq 0 ]]; then
  printf '%s\n' "$body"
  exit 0
fi

if ! command -v gh >/dev/null 2>&1; then
  echo "refused: gh not available for --apply" >&2
  exit 1
fi

if [[ "$residuals" -eq 0 ]]; then
  if [[ "$probes_known" -eq 0 ]]; then
    echo "refused: acceptance probe unknown; cannot establish residuals" >&2
    exit 1
  fi
  echo "==> residuals cleared (Pending=0 and official 0.2.0 present); skip reopen/nudge"
  exit 0
fi

# Reopen residual trackers independently of comment deduplication.
# Do not put GitHub merge-closing verbs next to issue numbers in commits.
for issue in 242 198; do
  issue_state="$(gh issue view "$issue" --repo vanedb/vanedb --json state -q .state)"
  case "$issue_state" in
    CLOSED)
      gh issue reopen "$issue" --repo vanedb/vanedb \
        --comment "Auto-reopened: residual AC3/AC5 still open (Pending=${pending}; official 0.2.0 ${official_020})."
      echo "==> reopened https://github.com/vanedb/vanedb/issues/${issue} (was CLOSED with residuals)"
      ;;
    OPEN) ;;
    *) echo "refused: unknown state for issue ${issue}" >&2; exit 1 ;;
  esac
done

# Positive residuals justify reopening, but transient unknowns must not replace
# a known acceptance state or create a new status notification.
if [[ "$probes_known" -eq 0 ]]; then
  echo "refused: acceptance probe unknown; skip closeout comment" >&2
  exit 1
fi

# Resolve the token's emitter, not the triggering workflow actor. GraphQL viewer
# is checked with the Actions token in CI; manual gh uses its current credential.
# Never infer ownership from the marker, GITHUB_ACTOR or a caller-supplied name.
if ! viewer_json="$(gh api graphql -f 'query=query { viewer { id login } }')"; then
  echo "refused: cannot resolve authenticated comment author" >&2
  exit 1
fi
comment_author_id="$(printf '%s' "$viewer_json" | python3 -c '
import json, re, sys
value = json.load(sys.stdin)
if value.get("errors"):
    raise ValueError("authenticated viewer query failed")
viewer = value["data"]["viewer"]
login = viewer["login"]
node_id = viewer["id"]
if not isinstance(node_id, str) or not node_id.strip():
    raise ValueError("authenticated comment author ID unresolved")
if not isinstance(login, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]*(?:\[bot\])?", login):
    raise ValueError("authenticated comment author unresolved")
print(node_id)
')"

# Read every page before posting or editing. Unreadable/incomplete history is
# not evidence that no status comment exists: never blindly post on read failure.
tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT
if ! gh api --paginate --slurp 'repos/vanedb/vanedb/issues/242/comments?per_page=100' >"$tmpdir/comments.json"; then
  echo "refused: cannot read existing closeout comments" >&2
  exit 1
fi
python3 - "$tmpdir" "$marker" "$fingerprint" "$comment_author_id" <<'PYCOMMENTS'
import json, pathlib, sys
root, marker, fingerprint = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
author_id = sys.argv[4]
pages = json.loads((root / "comments.json").read_text())
if not isinstance(pages, list) or not pages or any(not isinstance(p, list) for p in pages):
    raise ValueError("invalid paginated comment response")
comments = [c for page in pages for c in page]
if any(not isinstance(c, dict) or not isinstance(c.get("body"), str)
       or type(c.get("id")) is not int or c["id"] <= 0 for c in comments):
    raise ValueError("invalid comment response")
marked = [c for c in comments if c["body"].startswith(marker + "\n")
          and isinstance(c.get("user"), dict)
          and c["user"].get("node_id") == author_id]
if len(marked) > 1:
    raise ValueError("multiple marked status comments; reconcile before retrying")
if marked:
    comment = marked[0]
    (root / "comment-id").write_text(str(comment["id"]))
    if comment["body"].splitlines()[1:2] == [f"<!-- acceptance-state:{fingerprint} -->"]:
        (root / "unchanged").touch()
PYCOMMENTS

if [[ -f "$tmpdir/unchanged" ]]; then
  echo "==> acceptance state unchanged; skip closeout comment"
  exit 0
fi
printf '%s\n' "$body" >"$tmpdir/body"
if [[ -f "$tmpdir/comment-id" ]]; then
  comment_id="$(cat "$tmpdir/comment-id")"
  gh api --method PATCH "repos/vanedb/vanedb/issues/comments/${comment_id}" \
    --field "body=@$tmpdir/body" >/dev/null
  echo "==> updated closeout status on https://github.com/vanedb/vanedb/issues/242"
else
  gh issue comment 242 --repo vanedb/vanedb --body-file "$tmpdir/body"
  echo "==> added closeout status on https://github.com/vanedb/vanedb/issues/242"
fi
