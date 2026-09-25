#!/usr/bin/env bash
# Refresh GitHub issue #242 How-section onto the current AC5 tag path.
# Residual tracking: https://github.com/vanedb/vanedb/issues/242
# (#198/#226 may stay closed; this script never shuts issues.)
#
# Usage:
#   bash docs/launch/sync_242_how.sh           # print How section only
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

# Preview only the managed section; --apply reads and preserves the live issue.
how="$(cat <<'EOF'
## How

```bash
bash docs/launch/maintainer_closeout_226.sh          # status
bash docs/launch/maintainer_closeout_226.sh --fill   # Apple/Linux on dedicated HW
# Android: bench/compare/ANDROID.md
# After demo PR #20 merge + vault confirm:
bash docs/launch/maintainer_closeout_226.sh --tag --confirm-vault-walkthrough
```

Or Actions → **Fill COMPARISON (self-hosted)** / **Tag demo 0.2.0** (not historical **Cut demo 0.2.0**).

Demo PR: https://github.com/vanedb/obsidian-vane-search/pull/20

EOF
)"

if [[ "$APPLY" -eq 0 ]]; then
  printf '%s\n' "$how"
  exit 0
fi
printf '%s\n' "$how" | python3 "$ROOT/scripts/sync_242_how.py"
