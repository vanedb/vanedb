#!/usr/bin/env python3
"""Exercise the actual shell entrypoint with an isolated, no-network gh stub."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docs/launch/sync_242_how.sh"
spec = importlib.util.spec_from_file_location("sync_how", ROOT / "scripts/sync_242_how.py")
sync = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sync)
PREFIX = "## Why\n\n#198 is OPEN.\n\n## Acceptance\n\n- [x] Done: https://example.com/proof\n- [ ] Pending Ω\n\n"
SUFFIX = "## Evidence\n\n[Verified build](https://example.com/build)\n\n- [x] Accepted\n"
BODY = PREFIX + "## How\n\nold instructions\n\n" + SUFFIX


class HowSyncTests(unittest.TestCase):
    def run_sync(self, body=BODY, *, second=None, error=None, raw=None, args=("--apply",), script=SCRIPT):
        with tempfile.TemporaryDirectory(prefix="sync-how-test-") as directory:
            tmp = Path(directory)
            config = {"body": body, "second": second, "error": error, "raw": raw}
            (tmp / "config.json").write_text(json.dumps(config))
            gh = tmp / "gh"
            gh.write_text('''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
p = Path(os.environ["SYNC_TEST_DIR"])
c = json.loads((p / "config.json").read_text())
args = sys.argv[1:]
with (p / "calls").open("a") as out: out.write(json.dumps(args) + "\\n")
if args[:2] == ["issue", "view"]:
    if c["error"] == "read": raise SystemExit(2)
    count = len((p / "calls").read_text().splitlines())
    body = c["second"] if count > 1 and c["second"] is not None else c["body"]
    print(c["raw"] if c["raw"] is not None else json.dumps({"body": body}))
elif args[:2] == ["issue", "edit"]:
    if c["error"] == "write": raise SystemExit(3)
    source = args[args.index("--body-file") + 1]
    value = sys.stdin.buffer.read() if source == "-" else Path(source).read_bytes()
    (p / "written").write_bytes(value)
else: raise SystemExit("unexpected gh call")
''')
            gh.chmod(0o755)
            env = dict(os.environ, PATH=str(tmp) + os.pathsep + os.environ["PATH"], SYNC_TEST_DIR=str(tmp))
            result = subprocess.run(["bash", str(script), *args], env=env, capture_output=True)
            calls = [json.loads(s) for s in (tmp / "calls").read_text().splitlines()] if (tmp / "calls").exists() else []
            written = (tmp / "written").read_bytes().decode() if (tmp / "written").exists() else None
            return result, calls, written

    def assert_refused(self, **kwargs):
        result, calls, written = self.run_sync(**kwargs)
        self.assertNotEqual(result.returncode, 0)
        self.assertIsNone(written)
        self.assertFalse(any(c[:2] == ["issue", "edit"] for c in calls))

    def test_preserves_checkboxes_evidence_and_issue_state(self):
        result, calls, written = self.run_sync()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(written.startswith(PREFIX + "## How\n"))
        self.assertTrue(written.endswith(SUFFIX))
        self.assertIn("--tag --confirm-vault-walkthrough", written)
        self.assertEqual([c[:2] for c in calls], [["issue", "view"], ["issue", "view"], ["issue", "edit"]])

    def test_legacy_footer_preserved_byte_for_byte(self):
        suffix = sync.FOOTER + "\n\nMaintainer evidence: ✅\n"
        result, _, written = self.run_sync(body=PREFIX + "## How\nold\n\n" + suffix)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(written.endswith(suffix))

    def test_unchanged_has_no_write(self):
        _, _, updated = self.run_sync()
        result, calls, written = self.run_sync(body=updated)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(calls), 1)
        self.assertIsNone(written)

    def test_preview_is_offline_how_only(self):
        result, calls, written = self.run_sync(args=())
        self.assertEqual(result.returncode, 0)
        self.assertTrue(result.stdout.startswith(b"## How\n"))
        self.assertNotIn(b"## Why", result.stdout)
        self.assertEqual(calls, [])
        self.assertIsNone(written)

    def test_unreadable_body(self):
        self.assert_refused(error="read")

    def test_invalid_json(self):
        self.assert_refused(raw="not json")

    def test_non_object_json(self):
        self.assert_refused(raw="[]")

    def test_missing_body(self):
        self.assert_refused(raw="{}")

    def test_null_body(self):
        self.assert_refused(body=None)

    def test_empty_body(self):
        self.assert_refused(body="")

    def test_missing_heading(self):
        self.assert_refused(body=BODY.replace("## How", "## Instructions"))

    def test_duplicate_heading(self):
        self.assert_refused(body=BODY + "## How\nother\n## End\n")

    def test_missing_end_boundary(self):
        self.assert_refused(body=PREFIX + "## How\nold\n")

    def test_edited_legacy_boundary_refused(self):
        self.assert_refused(body=PREFIX + "## How\nold\n" + sync.FOOTER.replace("both boxes", "all boxes"))

    def test_unclosed_fence_refused(self):
        self.assert_refused(body=BODY + "```\n")

    def test_setext_boundary_refused(self):
        self.assert_refused(body=PREFIX + "## How\nold\n\nEvidence\n--------\nproof\n" + sync.FOOTER)

    def test_fenced_heading_not_boundary(self):
        body = PREFIX + "## How\n```md\n## How\n```\nold\n" + SUFFIX
        result, _, written = self.run_sync(body=body)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(written.endswith(SUFFIX))

    def test_concurrent_edit_refused(self):
        self.assert_refused(second=BODY.replace("- [ ] Pending", "- [x] Pending"))

    def test_edit_failure_propagates(self):
        result, _, written = self.run_sync(error="write")
        self.assertNotEqual(result.returncode, 0)
        self.assertIsNone(written)

    def test_crlf_and_no_final_newline_preserved(self):
        body = BODY.replace("\n", "\r\n").rstrip("\r\n")
        result, _, written = self.run_sync(body=body)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(written.startswith(PREFIX.replace("\n", "\r\n")))
        self.assertTrue(written.endswith(SUFFIX.replace("\n", "\r\n").rstrip("\r\n")))
        self.assertNotIn("\n", written.replace("\r\n", ""))

    def test_unknown_argument(self):
        self.assert_refused(args=("--bogus",))


if __name__ == "__main__":
    unittest.main()
