#!/usr/bin/env python3
"""Check CI routing and execute the real gate against simulated job outcomes."""
import itertools
import os
from pathlib import Path
import subprocess
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = yaml.load((ROOT / ".github/workflows/ci.yml").read_text(), Loader=yaml.BaseLoader)
COMPONENTS = ("rust", "cpp", "integration")
JOBS = WORKFLOW["jobs"]
GATE = JOBS["gate"]["steps"][0]


class CiDispatchTests(unittest.TestCase):
    def gate(self, selected, results=None, **overrides):
        env = dict(os.environ, CHANGES_RESULT="success", WORKFLOW_LINT_RESULT="success")
        for component, changed in zip(COMPONENTS, selected):
            env[component.upper() + "_CHANGED"] = "true" if changed else "false"
            env[component.upper() + "_RESULT"] = (
                (results or {}).get(component, "success" if changed else "skipped")
            )
        env.update(overrides)
        return subprocess.run(["bash", "-c", GATE["run"]], env=env,
                              capture_output=True, text=True)

    def test_dispatch_has_no_bypass_inputs_and_preserves_normal_triggers(self):
        self.assertEqual(WORKFLOW["on"], {
            "push": {"branches": ["main"]},
            "pull_request": {"branches": ["main"]},
            "workflow_dispatch": "",
        })

    def test_outputs_select_all_on_dispatch_and_use_filters_otherwise(self):
        # Pin the routing contract at its source: no second override on jobs or gate.
        for component in COMPONENTS:
            self.assertEqual(JOBS["changes"]["outputs"][component],
                             "${{ github.event_name == 'workflow_dispatch' && 'true' || "
                             f"steps.filter.outputs.{component} }}}}")
            selector = f"${{{{ needs.changes.outputs.{component} == 'true' }}}}"
            self.assertEqual(JOBS[component]["if"], selector)
            self.assertEqual(JOBS[component]["needs"], "changes")
            self.assertEqual(GATE["env"][component.upper() + "_CHANGED"],
                             f"${{{{ needs.changes.outputs.{component} }}}}")
            self.assertEqual(GATE["env"][component.upper() + "_RESULT"],
                             f"${{{{ needs.{component}.result }}}}")
        path_filter = next(s for s in JOBS["changes"]["steps"] if s.get("id") == "filter")
        self.assertEqual(path_filter["if"], "${{ github.event_name != 'workflow_dispatch' }}")
        self.assertIn("dorny/paths-filter@", path_filter["uses"])
        self.assertEqual(JOBS["gate"]["if"], "${{ always() }}")
        self.assertEqual(set(JOBS["gate"]["needs"]), {"changes", "workflow-lint", *COMPONENTS})

    def test_normal_path_selection_accepts_only_expected_job_outcomes(self):
        # All eight real selection masks, including docs-only (all skipped).
        for selected in itertools.product((False, True), repeat=3):
            with self.subTest(selected=selected):
                result = self.gate(selected)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            for component, changed in zip(COMPONENTS, selected):
                for outcome in ("success", "failure", "cancelled", "skipped"):
                    if outcome == ("success" if changed else "skipped"):
                        continue
                    with self.subTest(selected=selected, component=component, outcome=outcome):
                        self.assertNotEqual(self.gate(selected, {component: outcome}).returncode, 0)

    def test_dispatch_requires_every_component_success(self):
        self.assertEqual(self.gate((True, True, True)).returncode, 0)
        for component in COMPONENTS:
            for outcome in ("failure", "cancelled", "skipped", ""):
                with self.subTest(component=component, outcome=outcome):
                    self.assertNotEqual(self.gate((True, True, True),
                                                 {component: outcome}).returncode, 0)

    def test_detection_or_lint_failure_cannot_be_hidden_by_component_results(self):
        for selected in ((False, False, False), (True, True, True)):
            for upstream in ("CHANGES_RESULT", "WORKFLOW_LINT_RESULT"):
                for outcome in ("failure", "cancelled", "skipped", ""):
                    with self.subTest(selected=selected, upstream=upstream, outcome=outcome):
                        self.assertNotEqual(self.gate(selected, **{upstream: outcome}).returncode, 0)


if __name__ == "__main__":
    unittest.main()
