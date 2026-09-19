#!/usr/bin/env bash
# Exit 0 if this host may run publishable compare timings; exit 1 if shared.
# Mirrors bench/compare/src/publish.rs::shared_runner_signals.
#
# Order matches Rust:
#   1) Cursor/Codespaces env + FS markers → shared
#   2) /opt/hostedtoolcache → shared (GitHub-hosted image)
#   3) RUNNER_ENVIRONMENT=self-hosted → dedicated (operator GHA runner)
#   4) CI / GITHUB_ACTIONS / other CI → shared
#
# Usage:
#   bash refuse_shared_runner.sh          # exit status only
#   source refuse_shared_runner.sh && compare_refuse_if_shared_runner
set -euo pipefail

compare_refuse_if_shared_runner() {
  if [[ -n "${CURSOR_AGENT:-}" || -n "${CODESPACES:-}" \
     || -e /opt/cursor || -e /exec-daemon ]]; then
    echo "refusing: Cursor/Codespaces shared runner (env or FS markers)" >&2
    return 1
  fi
  if [[ -e /opt/hostedtoolcache ]]; then
    echo "refusing: GitHub-hosted image marker /opt/hostedtoolcache" >&2
    return 1
  fi
  # Operator-owned GHA self-hosted: CI/GITHUB_ACTIONS are set, but timings are
  # dedicated hardware. Must not override the hostedtoolcache check above.
  if [[ "${RUNNER_ENVIRONMENT:-}" == "self-hosted" ]]; then
    return 0
  fi
  if [[ "${CI:-}" == "true" || "${CI:-}" == "1" \
     || "${GITHUB_ACTIONS:-}" == "true" || "${GITHUB_ACTIONS:-}" == "1" \
     || "${GITLAB_CI:-}" == "true" || "${CIRCLECI:-}" == "true" \
     || "${BUILDKITE:-}" == "true" || "${TF_BUILD:-}" == "True" \
     || "${TF_BUILD:-}" == "true" ]]; then
    echo "refusing: shared CI/cloud env (set RUNNER_ENVIRONMENT=self-hosted on operator-owned GHA runners without /opt/hostedtoolcache)" >&2
    return 1
  fi
  return 0
}

# When executed (not sourced), run the check as the script's exit status.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  compare_refuse_if_shared_runner
fi
