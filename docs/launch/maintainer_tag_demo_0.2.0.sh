#!/usr/bin/env bash
# Tag the official obsidian-vane-search 0.2.0 release for vanedb#242 / AC5.
#
# This is the current-candidate path (demo PR #20), not the historical patch
# cut in maintainer_cut_demo_0.2.0.sh. Sequence required by the demo checklist:
#   1. Independent review + CI on the candidate
#   2. Real Obsidian/Ollama vault walkthrough (human; record evidence)
#   3. Merge https://github.com/vanedb/obsidian-vane-search/pull/20
#   4. This script: annotated 0.2.0 tag on the reviewed merged commit + push
#
# Step 3 can be folded into this script with --merge-if-open (merge commit,
# not squash) when DEMO_REPO_TOKEN / App write can merge the PR. Actions →
# Tag demo 0.2.0 passes --merge-if-open after vault confirmation + secret.
# Tagging also requires green merged-commit CI and a rebuilt bundle matching
# desktop acceptance. If CI is pending, rerun after that exact commit passes.
#
# Requires: write access to vanedb/obsidian-vane-search (DEMO_REPO_TOKEN or
# Cursor GitHub App contents:write on that repo after #215
# repositoryDependencies), network, and an explicit vault confirmation.
# Reply on https://github.com/vanedb/vanedb/issues/242 with the release URL.
#
# Usage:
#   bash docs/launch/maintainer_tag_demo_0.2.0.sh \
#     --confirm-vault-walkthrough [--merge-if-open] [--dry-run] [--commit <sha>]
#   DEMO_REPO_TOKEN=… bash docs/launch/maintainer_tag_demo_0.2.0.sh \
#     --confirm-vault-walkthrough --merge-if-open
set -euo pipefail

TAG="0.2.0"
DEMO_PR=20
DEMO_REPO="vanedb/obsidian-vane-search"
CONFIRM_VAULT=0
MERGE_IF_OPEN=0
DRY_RUN=0
COMMIT=""

for arg in "$@"; do
  case "$arg" in
    --confirm-vault-walkthrough) CONFIRM_VAULT=1 ;;
    --merge-if-open) MERGE_IF_OPEN=1 ;;
    --dry-run) DRY_RUN=1 ;;
    --commit)
      echo "error: --commit requires a SHA argument (use --commit=<sha>)" >&2
      exit 2
      ;;
    --commit=*)
      COMMIT="${arg#--commit=}"
      ;;
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

if [[ "$CONFIRM_VAULT" -ne 1 ]]; then
  echo "refused: real vault walkthrough must be recorded before tagging" >&2
  echo "  Pass --confirm-vault-walkthrough after completing docs/releases/0.2.0.md" >&2
  echo "  on $DEMO_REPO (PR #$DEMO_PR). Do not use historical maintainer_cut_demo_0.2.0.sh." >&2
  exit 2
fi

# Prefer explicit DEMO_REPO_TOKEN. Else, when DEMO_URL is unset and the Cursor
# App installation includes the demo repo, use `gh auth token` so clone+push
# (and --merge-if-open) authenticate. Do not fall back to ambient GH_TOKEN when
# the demo repo is missing from /installation/repositories — that vanedb-only
# credential 403s even a public HTTPS clone.
# Paginate: default page size can miss the demo repo when many are installed.
app_demo_in_scope() {
  command -v gh >/dev/null 2>&1 || return 1
  gh api --paginate /installation/repositories --jq '.repositories[].full_name' \
    2>/dev/null | grep -Fxq 'vanedb/obsidian-vane-search'
}

if [[ -z "${DEMO_REPO_TOKEN:-}" && -z "${DEMO_URL:-}" ]] && app_demo_in_scope; then
  tok="$(gh auth token 2>/dev/null || true)"
  if [[ -n "$tok" ]]; then
    DEMO_REPO_TOKEN="$tok"
    echo "==> DEMO_REPO_TOKEN: using Cursor App installation token (demo repo in App scope)"
  fi
fi

# Prefer DEMO_REPO_TOKEN for demo-repo gh calls (Actions GITHUB_TOKEN is
# vanedb-scoped and cannot merge/tag the demo repo).
gh_demo() {
  if [[ -n "${DEMO_REPO_TOKEN:-}" ]]; then
    GH_TOKEN="$DEMO_REPO_TOKEN" gh "$@"
  else
    gh "$@"
  fi
}

# Honor a pre-set DEMO_URL (CI bare-remote tests use file://…). When TOKEN is
# set, embed it into plain https://github.com/… URLs so push cannot fall back
# to ambient Cursor url.*.insteadOf credentials. Leave file:// and already-
# credentialed URLs alone.
if [[ -n "${DEMO_REPO_TOKEN:-}" ]]; then
  if [[ -z "${DEMO_URL:-}" ]]; then
    DEMO_URL="https://x-access-token:${DEMO_REPO_TOKEN}@github.com/vanedb/obsidian-vane-search.git"
  elif [[ "$DEMO_URL" == https://github.com/* ]]; then
    DEMO_URL="https://x-access-token:${DEMO_REPO_TOKEN}@github.com/${DEMO_URL#https://github.com/}"
  fi
elif [[ -z "${DEMO_URL:-}" ]]; then
  DEMO_URL="https://github.com/vanedb/obsidian-vane-search.git"
fi

# Local bare-remote CI only: file:// DEMO_URL with an explicit --commit skips
# the live GitHub PR merge check (no network write; proves tag+push plumbing).
SKIP_PR_GATE=0
if [[ -n "${DEMO_URL:-}" && "$DEMO_URL" == file://* && -n "$COMMIT" ]]; then
  SKIP_PR_GATE=1
  echo "==> file:// DEMO_URL + --commit: skipping live PR #$DEMO_PR merge gate (CI)"
fi

if [[ "$SKIP_PR_GATE" -eq 0 ]]; then
  if ! command -v gh >/dev/null 2>&1; then
    echo "gh CLI required to verify demo PR #$DEMO_PR merge state" >&2
    exit 1
  fi

  pr_json="$(gh_demo pr view "$DEMO_PR" -R "$DEMO_REPO" \
    --json state,mergedAt,mergeCommit,headRefOid,mergeable,mergeStateStatus,statusCheckRollup,url)"
  pr_state="$(printf '%s' "$pr_json" | python3 -c 'import json,sys; print(json.load(sys.stdin)["state"])')"

  if [[ "$pr_state" == "OPEN" && "$MERGE_IF_OPEN" -eq 1 ]]; then
    mergeable="$(printf '%s' "$pr_json" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("mergeable") or "")')"
    merge_status="$(printf '%s' "$pr_json" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("mergeStateStatus") or "")')"
    head_oid="$(printf '%s' "$pr_json" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("headRefOid") or "")')"
    if [[ "$mergeable" != "MERGEABLE" ]]; then
      echo "refused: demo PR #$DEMO_PR is OPEN but mergeable=$mergeable (need MERGEABLE)" >&2
      exit 2
    fi
    # Fail closed: UNKNOWN/UNSTABLE/HAS_HOOKS are not green evidence.
    if [[ "$merge_status" != "CLEAN" ]]; then
      echo "refused: demo PR #$DEMO_PR mergeStateStatus=$merge_status (need CLEAN)" >&2
      exit 2
    fi
    if [[ ! "$head_oid" =~ ^[0-9a-fA-F]{40}$ ]]; then
      echo "refused: demo PR #$DEMO_PR has no valid reviewed head SHA" >&2
      exit 2
    fi
    if ! printf '%s' "$pr_json" | python3 -c '
import json,sys
checks=json.load(sys.stdin).get("statusCheckRollup") or []
def green(check):
    if check.get("__typename") == "StatusContext":
        return check.get("state") == "SUCCESS"
    return check.get("status") == "COMPLETED" and check.get("conclusion") in ("SUCCESS", "SKIPPED")
if not any(check.get("name") == "test" and green(check) and check.get("conclusion") == "SUCCESS" for check in checks) or not all(green(check) for check in checks):
    raise SystemExit("refused: demo PR checks must all finish successfully before merging")
'; then
      exit 2
    fi
    if [[ -n "$COMMIT" ]]; then
      echo "refused: --commit cannot select a commit before PR #$DEMO_PR is merged" >&2
      exit 2
    fi
    if [[ -n "$head_oid" ]]; then
      if ! gh_demo api "repos/${DEMO_REPO}/contents/docs/releases/0.2.0-desktop-acceptance.md?ref=${head_oid}" \
          >/dev/null 2>&1; then
        echo "refused: vault acceptance missing on PR #$DEMO_PR head ${head_oid:0:12}" >&2
        echo "  Need docs/releases/0.2.0-desktop-acceptance.md before --merge-if-open" >&2
        exit 2
      fi
    fi
    if [[ "$DRY_RUN" -eq 1 ]]; then
      echo "==> --dry-run: would merge PR #$DEMO_PR with merge commit, then tag"
      if [[ -z "$COMMIT" && -n "$head_oid" ]]; then
        COMMIT="$head_oid"
        echo "    (dry-run uses head ${COMMIT:0:12} as stand-in; real run tags merge commit)"
      fi
    else
      if [[ -z "${DEMO_REPO_TOKEN:-}" ]]; then
        echo "refused: --merge-if-open needs DEMO_REPO_TOKEN or Cursor App on $DEMO_REPO" >&2
        exit 2
      fi
      echo "==> merging demo PR #$DEMO_PR (merge commit, not squash)"
      gh_demo pr merge "$DEMO_PR" -R "$DEMO_REPO" --merge --match-head-commit "$head_oid"
      pr_json="$(gh_demo pr view "$DEMO_PR" -R "$DEMO_REPO" \
        --json state,mergedAt,mergeCommit,headRefOid,mergeable,mergeStateStatus,statusCheckRollup,url)"
      pr_state="$(printf '%s' "$pr_json" | python3 -c 'import json,sys; print(json.load(sys.stdin)["state"])')"
    fi
  fi

  if [[ "$pr_state" != "MERGED" ]]; then
    if [[ "$DRY_RUN" -eq 1 && "$MERGE_IF_OPEN" -eq 1 && "$pr_state" == "OPEN" && -n "$COMMIT" ]]; then
      echo "==> --dry-run: skipping MERGED gate (would merge first)"
    else
      echo "refused: demo PR #$DEMO_PR is $pr_state (must be MERGED before tagging)" >&2
      echo "  $(printf '%s' "$pr_json" | python3 -c 'import json,sys; print(json.load(sys.stdin)["url"])')" >&2
      if [[ "$pr_state" == "OPEN" && "$MERGE_IF_OPEN" -eq 0 ]]; then
        echo "  Pass --merge-if-open (with DEMO_REPO_TOKEN) to merge then tag in one step." >&2
      fi
      exit 2
    fi
  fi

  if [[ "$pr_state" == "MERGED" ]]; then
    merge_oid="$(printf '%s' "$pr_json" | python3 -c '
import json,sys
mc=json.load(sys.stdin).get("mergeCommit") or {}
print(mc.get("oid") or "")
')"
    if [[ ! "$merge_oid" =~ ^[0-9a-fA-F]{40}$ ]]; then
      echo "refused: mergeCommit.oid missing or invalid" >&2
      exit 2
    fi
    if [[ -n "$COMMIT" && ( ! "$COMMIT" =~ ^[0-9a-fA-F]{7,40}$ || "$merge_oid" != "$COMMIT" ) ]]; then
      echo "refused: --commit must identify the reviewed PR #$DEMO_PR merge commit" >&2
      exit 2
    fi
    COMMIT="$merge_oid"
  fi
fi

if [[ -z "$COMMIT" ]]; then
  echo "refused: need a merge commit (PR #$DEMO_PR MERGED) or --commit=<sha>" >&2
  exit 2
fi
if [[ ! "$COMMIT" =~ ^[0-9a-fA-F]{7,40}$ ]]; then
  echo "invalid --commit SHA: $COMMIT" >&2
  exit 2
fi

redact_url() {
  sed -E 's#://[^/@]+@#://***@#g' <<<"$1"
}

# Cursor cloud injects global url.*.insteadOf rewrites that map every
# https://github.com/… URL to a vanedb-scoped App token. That rewrite survives
# `env -u GH_TOKEN` and 403s demo-repo writes. Disable global/system gitconfig
# for clone/fetch/push so DEMO_REPO_TOKEN (or anonymous HTTPS) is what git uses.
git_no_ambient_url_rewrite() {
  env GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_SYSTEM=/dev/null git "$@"
}

clone_demo() {
  local dest="$1"
  if [[ -n "${DEMO_REPO_TOKEN:-}" ]]; then
    git_no_ambient_url_rewrite clone --depth 50 "$DEMO_URL" "$dest"
  else
    env -u GH_TOKEN -u GITHUB_TOKEN -u GH_ENTERPRISE_TOKEN \
      GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_SYSTEM=/dev/null \
      git clone --depth 50 "$DEMO_URL" "$dest"
  fi
}

WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/vane-demo-tag-0.2.0.XXXXXX")"
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

echo "==> fetch + checkout reviewed merged commit $COMMIT"
# deepen/fetch so the merge commit is present even on a shallow clone
git_no_ambient_url_rewrite fetch --depth 50 origin "$COMMIT" 2>/dev/null \
  || git_no_ambient_url_rewrite fetch origin "$COMMIT"
git checkout --detach "$COMMIT"

ver="$(python3 -c 'import json; print(json.load(open("manifest.json"))["version"])')"
if [[ "$ver" != "$TAG" ]]; then
  echo "refused: manifest.json version is '$ver' (want $TAG) at $COMMIT" >&2
  exit 1
fi

# Only the explicit local file:// bare-remote test skips GitHub CI and bundle
# acceptance. It cannot publish to GitHub; it tests tag/push plumbing alone.
if [[ "$SKIP_PR_GATE" -eq 0 ]]; then
  echo "==> verify CI on final merged commit $COMMIT"
  gh_demo api --paginate --slurp \
    "repos/${DEMO_REPO}/commits/${COMMIT}/check-runs?per_page=100" >"$WORKDIR/check-runs.json"
  gh_demo api --paginate --slurp \
    "repos/${DEMO_REPO}/commits/${COMMIT}/status?per_page=100" >"$WORKDIR/statuses.json"
  python3 - "$WORKDIR/check-runs.json" "$WORKDIR/statuses.json" <<'PYCI'
import json,sys
from pathlib import Path
runs=[run for page in json.loads(Path(sys.argv[1]).read_text()) for run in page.get("check_runs", [])]
statuses=[status for page in json.loads(Path(sys.argv[2]).read_text()) for status in page.get("statuses", [])]
green_runs=all(run.get("status") == "completed" and run.get("conclusion") in ("success", "skipped") for run in runs)
green_statuses=all(status.get("state") == "success" for status in statuses)
has_success=any(run.get("name") == "test" and run.get("conclusion") == "success" for run in runs)
if not (has_success and green_runs and green_statuses):
    raise SystemExit("refused: final merged-commit CI is missing, pending, or unsuccessful; wait for green CI and rerun before tagging")
PYCI

  echo "==> validate and rebuild final merged demo"
  npm ci
  npm run check:release
  npm run typecheck
  npm test
  npm run build
  node scripts/check-size.mjs
  python3 - <<'PYBUNDLE'
import hashlib,re
from pathlib import Path
record=Path("docs/releases/0.2.0-desktop-acceptance.md")
if not record.is_file():
    raise SystemExit("refused: desktop acceptance record missing on final merged commit")
matches=re.findall(r"^\| Installed `main\.js` SHA-256 \| `([0-9a-f]{64})` \|$", record.read_text(), re.MULTILINE)
if len(matches) != 1:
    raise SystemExit("refused: desktop acceptance must contain exactly one valid installed main.js SHA-256")
bundle=Path("main.js")
if not bundle.is_file() or hashlib.sha256(bundle.read_bytes()).hexdigest() != matches[0]:
    raise SystemExit("refused: rebuilt main.js differs from desktop acceptance; repeat the walkthrough and record the new accepted bundle before tagging")
print("ok: rebuilt final bundle matches recorded desktop acceptance")
PYBUNDLE
fi

if git rev-parse "refs/tags/$TAG" >/dev/null 2>&1; then
  echo "tag $TAG already exists locally after clone; aborting" >&2
  exit 1
fi
if git ls-remote --tags origin "refs/tags/$TAG" | grep -q .; then
  echo "tag $TAG already exists on origin; aborting" >&2
  exit 1
fi

echo "==> annotated tag $TAG at $COMMIT (vault walkthrough confirmed)"
git tag -a "$TAG" -m "obsidian-vane-search $TAG (vanedb#242; demo PR #$DEMO_PR)"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "==> --dry-run: skipping push of tag $TAG"
  echo "Local tag ready: $(git rev-parse "$TAG")"
  echo "Expected URL after a real tag push: https://github.com/vanedb/obsidian-vane-search/releases/tag/$TAG"
  exit 0
fi

if [[ -z "${DEMO_REPO_TOKEN:-}" && "$DEMO_URL" != file://* ]]; then
  echo "refused: no DEMO_REPO_TOKEN / App write creds for push" >&2
  echo "  Set DEMO_REPO_TOKEN or install Cursor GitHub App on $DEMO_REPO" >&2
  exit 2
fi

echo "==> push tag (triggers .github/workflows/release.yml)"
git_no_ambient_url_rewrite push origin "refs/tags/$TAG"

echo
echo "Pushed tag $TAG. Release workflow will attach main.js + manifest.json + LICENSE."
echo "Watch: https://github.com/vanedb/obsidian-vane-search/actions"
echo "Expected URL: https://github.com/vanedb/obsidian-vane-search/releases/tag/$TAG"
echo "Reply on https://github.com/vanedb/vanedb/issues/242 with that URL when green."
