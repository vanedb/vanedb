#!/usr/bin/env python3
"""Tests for check_rfcs.py, run by CI's workflow-lint job."""

import shutil
import tempfile
import unittest
from pathlib import Path

import check_rfcs

ROOT = Path(__file__).resolve().parents[1]


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="rfcs-"))
        (self.tmp / "docs/rfcs").mkdir(parents=True)
        self.rfc("0001-alpha.md", "accepted")
        (self.tmp / "docs/rfcs/README.md").write_text("| [0001](0001-alpha.md) |\n")
        (self.tmp / "docs/ROADMAP.md").write_text("[0001](rfcs/0001-alpha.md)\n")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def rfc(self, name, status, title=None):
        number = name[:4]
        (self.tmp / "docs/rfcs" / name).write_text(
            f"# RFC {title or number}: Alpha\n\n- Status: {status}\n\n"
            "## Problem\n\n## Decision\n\n## Acceptance criteria\n"
        )

    def test_clean_tree_passes(self):
        self.assertEqual(check_rfcs.check(self.tmp), [])

    def test_repository_passes(self):
        self.assertEqual(check_rfcs.check(ROOT), [])

    def test_bad_status(self):
        self.rfc("0001-alpha.md", "pending")
        self.assertTrue(any("status" in p for p in check_rfcs.check(self.tmp)))

    def test_compound_status_accepted(self):
        for value in ("accepted; the amendment below is draft",
                      "accepted (decision of 2026-09-07)",
                      "superseded by 0009", "parked"):
            self.rfc("0001-alpha.md", value)
            self.assertEqual(check_rfcs.check(self.tmp), [], value)

    def test_unindexed_rfc(self):
        self.rfc("0002-beta.md", "draft")
        problems = check_rfcs.check(self.tmp)
        self.assertTrue(any("README.md" in p for p in problems))
        self.assertTrue(any("ROADMAP.md" in p for p in problems))

    def test_wrong_title_number(self):
        self.rfc("0001-alpha.md", "draft", title="0002")
        self.assertTrue(any("title" in p for p in check_rfcs.check(self.tmp)))

    def test_template_is_ignored(self):
        (self.tmp / "docs/rfcs/0000-template.md").write_text("# RFC NNNN: Title\n- Status: draft\n")
        self.assertEqual(check_rfcs.check(self.tmp), [])


if __name__ == "__main__":
    unittest.main()
