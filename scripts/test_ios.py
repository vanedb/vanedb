#!/usr/bin/env python3
"""Build and run the C acceptance test in a disposable iOS simulator."""

import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def run(*args, capture=False, timeout=300):
    return subprocess.run(
        args, cwd=ROOT, check=True, text=True, capture_output=capture, timeout=timeout
    ).stdout


def main():
    devices = json.loads(run("xcrun", "simctl", "list", "devices", "available", "--json", capture=True))
    # Use an installed runtime/device pairing, but never boot or alter a user's device.
    choices = [
        (runtime, device["deviceTypeIdentifier"])
        for runtime, entries in devices["devices"].items()
        if ".iOS-" in runtime
        for device in entries
        if device.get("isAvailable") and "iPhone" in device["name"]
    ]
    if not choices:
        raise SystemExit("An installed iPhone simulator runtime is required")
    runtime, device_type = sorted(choices)[-1]
    with tempfile.TemporaryDirectory(prefix="vanedb-ios-") as temporary:
        binary = str(Path(temporary) / "acceptance")
        run("xcrun", "--sdk", "iphonesimulator", "clang", "-std=c11", "-Wall", "-Wextra", "-Werror",
            "-target", "arm64-apple-ios14.0-simulator", "-I", "vanedb-capi/include",
            "vanedb-capi/tests/acceptance.c",
            "target/aarch64-apple-ios-sim/release/libvanedb_capi.a", "-o", binary)
        device = run("xcrun", "simctl", "create", "VaneDB acceptance", device_type, runtime, capture=True).strip()
        print(f"Running C ABI acceptance on {runtime}: {device}", flush=True)
        try:
            run("xcrun", "simctl", "boot", device)
            run("xcrun", "simctl", "bootstatus", device, "-b")
            run("xcrun", "simctl", "spawn", device, binary, temporary)
        finally:
            subprocess.run(["xcrun", "simctl", "shutdown", device], check=False, timeout=60)
            run("xcrun", "simctl", "delete", device)


if __name__ == "__main__":
    main()
