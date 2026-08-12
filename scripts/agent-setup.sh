#!/usr/bin/env bash
# Environment setup for coding agents (Codex, Claude Code web, CI sandboxes).
# Usage: scripts/agent-setup.sh [wasm] [py]
#   no args : core Rust only (fmt/clippy components)
#   wasm    : + wasm32 target and wasm-pack (needs node for wasm-pack test)
#   py      : + maturin and pytest for vanedb-py
set -euo pipefail

rustup component add rustfmt clippy

for extra in "$@"; do
  case "$extra" in
    wasm)
      rustup target add wasm32-unknown-unknown
      command -v wasm-pack >/dev/null 2>&1 ||
        curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
      ;;
    py)
      python3 -m pip install --quiet maturin pytest
      ;;
    *)
      echo "unknown extra: $extra (expected: wasm, py)" >&2
      exit 1
      ;;
  esac
done

echo "agent-setup done: core${*:+ + $*}"
