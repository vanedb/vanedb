#!/usr/bin/env bash
# Refresh GitHub issue #242 How-section onto the current AC5 tag path.
# Residual tracking: https://github.com/vanedb/vanedb/issues/242
# (#198/#226 may stay closed; this script never shuts issues.)
#
# Usage:
#   bash docs/launch/sync_242_how.sh           # print body only
#   bash docs/launch/sync_242_how.sh --apply   # gh issue edit 242
#
# Prefer Actions → Sync #242 How (GITHUB_TOKEN has issues:write). Cloud agents
# often cannot edit issues with the Cursor App token.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
APPLY=0
for arg in "$@"; do
  case "$arg" in
    --apply) APPLY=1 ;;
    -h|--help)
      sed -n '2,12p' "$0"
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

body="$(cat <<EOF
## Why

[#226](https://github.com/vanedb/vanedb/issues/226) was auto-closed when [#238](https://github.com/vanedb/vanedb/pull/238) merged (\`Fixes\`/\`closes\` keyword). That PR only unblocked the Android on-device \`--out-dir\` path — it did **not** fill COMPARISON or land the official demo release.

#198 remains closed from the [#212](https://github.com/vanedb/vanedb/pull/212) merge while the same two acceptance criteria are still open.

## Remaining acceptance (from #198)

- [ ] \`bench/COMPARISON.md\` filled for Apple Silicon, Linux AVX2, and Android ARM64 (cosine + L2) from **dedicated** hardware only — currently ${pending}× \`*Pending.*\` on tip \`${tip}\`
- [ ] Official [\`obsidian-vane-search\` 0.2.0](https://github.com/vanedb/obsidian-vane-search) release URL (vanedb \`demo-0.2.0-staging\` ≠ this)

## How

\`\`\`bash
bash docs/launch/maintainer_closeout_226.sh          # status
bash docs/launch/maintainer_closeout_226.sh --fill   # Apple/Linux on dedicated HW
# Android: bench/compare/ANDROID.md
# After demo PR #20 merge + vault confirm:
bash docs/launch/maintainer_closeout_226.sh --tag --confirm-vault-walkthrough
\`\`\`

Or Actions → **Fill COMPARISON (self-hosted)** / **Tag demo 0.2.0** (not historical **Cut demo 0.2.0**).

Demo PR: https://github.com/vanedb/obsidian-vane-search/pull/20

**Do not** put \`Fixes\`/\`Closes\` next to \`#198\` / \`#226\` / this issue on harness PRs until both boxes above have evidence URLs. Shut the issue from the GitHub UI after evidence, not via merge keywords.
EOF
)"

# Refuse if the generated How still recommends historical --cut.
if printf '%s\n' "$body" | grep -E '^bash docs/launch/maintainer_closeout_226.sh --cut'; then
  echo "refused: generated body still recommends --cut" >&2
  exit 1
fi

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
gh issue edit 242 --repo vanedb/vanedb --body-file "$tmp"
rm -f "$tmp"
echo "==> updated https://github.com/vanedb/vanedb/issues/242 How (--tag path; Pending=${pending})"
