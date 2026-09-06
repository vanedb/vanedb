#!/usr/bin/env python3
"""Serve an extracted npm artifact and verify it in a real browser."""

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
from threading import Thread

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("browser", choices=["chrome", "firefox", "webkit"])
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="vanedb-web-") as temporary:
        with tarfile.open(args.archive) as archive:
            archive.extractall(temporary, filter="data")
        shutil.copy2(ROOT / "vanedb-wasm/tests/packaged.html", Path(temporary) / "index.html")
        with ThreadingHTTPServer(("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=temporary)) as server:
            worker = Thread(target=server.serve_forever, daemon=True)
            worker.start()
            cli = ["npx", "--yes", "--package", "@playwright/cli@0.1.19", "playwright-cli",
                   "--session", Path(temporary).name]
            try:
                subprocess.run(cli + ["open", f"http://127.0.0.1:{server.server_port}",
                                      "--browser", args.browser], cwd=temporary, check=True, timeout=180)
                subprocess.run(cli + ["run-code", "async (page) => { await page.locator('#result')"
                                      ".filter({ hasText: /^PASS:/ }).waitFor({ timeout: 30000 }); }"],
                               cwd=temporary, check=True, timeout=60)
            finally:
                try:
                    subprocess.run(cli + ["close"], cwd=temporary, check=False, timeout=60)
                finally:
                    server.shutdown()
                    worker.join()
    print(f"Packaged browser acceptance passed: {args.browser}")


if __name__ == "__main__":
    main()
