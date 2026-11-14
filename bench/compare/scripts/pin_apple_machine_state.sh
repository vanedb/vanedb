#!/usr/bin/env bash
# RFC 0003 Apple Silicon machine state for a publish run.
# Source this file; it defines functions and does not run them.
#
# `VANEDB_COMPARE_DEDICATED=1` is the operator attestation. These checks are
# the part that attestation cannot see: the laptop is on AC power and Low
# Power Mode is off, so the clocks stay at the plugged-in baseline. The
# operator still has to stop other work before starting the command.
# shellcheck shell=bash

compare_pin_apple_machine_state() {
  local hw="${1:?hw label}"
  [[ "$hw" == apple-* ]] || return 0
  if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "refusing ${hw} on $(uname -s); need Darwin Apple Silicon" >&2
    return 1
  fi
  if ! command -v pmset >/dev/null 2>&1; then
    echo "refusing ${hw}: pmset is required to confirm the machine is plugged in" >&2
    return 1
  fi
  local batt
  batt="$(pmset -g batt)"
  if ! grep -q "AC Power" <<<"$batt"; then
    echo "refusing ${hw}: pmset -g batt does not show AC Power." >&2
    echo "Plug the machine in (RFC 0003: dedicated, plugged in, other work stopped), then re-run." >&2
    return 1
  fi
  local low
  low="$(pmset -g | awk '/^ lowpowermode / { print $2; exit }')"
  if [[ "$low" == "1" ]]; then
    echo "refusing ${hw}: Low Power Mode is on (pmset lowpowermode 1)." >&2
    echo "Turn Low Power Mode off so the CPU stays at the plugged-in baseline, then re-run." >&2
    return 1
  fi
  echo "machine state: AC Power confirmed; Low Power Mode is off."
  echo "Stop other work before the timed section. cargo runs under caffeinate -i."
}

# Hold an assertion so a Mac does not idle-sleep during compile or measure.
compare_run_awake() {
  if [[ "$(uname -s)" == "Darwin" ]] && command -v caffeinate >/dev/null 2>&1; then
    caffeinate -i "$@"
  else
    "$@"
  fi
}
