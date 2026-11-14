#!/usr/bin/env python3
"""Check CI routing and execute the real gate against simulated job outcomes."""
import itertools
import os
from pathlib import Path
import re
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

    def test_unselected_components_must_be_skipped_for_the_gate_to_pass(self):
        # Docs-only and other empty selections: every component job is skipped
        # and the gate passes. A selected component that did not run, or an
        # unselected one that did, is a hole.
        self.assertEqual(self.gate((False, False, False)).returncode, 0)
        self.assertNotEqual(
            self.gate((True, False, False), {"rust": "skipped"}).returncode, 0)
        self.assertNotEqual(
            self.gate((False, False, False), {"rust": "success"}).returncode, 0)
        self.assertNotEqual(
            self.gate((False, True, False), {"cpp": "skipped"}).returncode, 0)
        self.assertNotEqual(
            self.gate((False, False, True), {"integration": "failure"}).returncode, 0)

    def test_path_filters_name_the_files_each_component_actually_reads(self):
        # A blanket `.github/workflows/**` made a one-file action pin rebuild
        # every component. workflow-lint has no path filter, so actionlint and
        # the release-script unit tests still run for every pull request.
        self.assertNotIn("paths", WORKFLOW["on"]["pull_request"])
        self.assertNotIn("if", JOBS["workflow-lint"])

        path_filter = next(s for s in JOBS["changes"]["steps"] if s.get("id") == "filter")
        filters = yaml.load(path_filter["with"]["filters"], Loader=yaml.BaseLoader)
        self.assertEqual(set(filters), set(COMPONENTS))

        for patterns in filters.values():
            self.assertNotIn(".github/workflows/**", patterns)

        # Sources and release workflows for the artifacts that ship. The PR
        # jobs that build them are the Rust component. C++ does not publish
        # them, so a release-workflow edit must not drag that matrix in, and
        # must not skip Rust.
        release_inputs = [
            "vanedb/**",
            "vanedb-py/**",
            "vanedb-wasm/**",
            "vanedb-capi/**",
            "Cargo.toml",
            "Cargo.lock",
            ".github/workflows/publish-crate.yml",
            ".github/workflows/publish-rust.yml",
            ".github/workflows/publish-wasm.yml",
            ".github/workflows/publish-capi.yml",
            ".github/requirements-release.txt",
            ".github/scripts/validate_python_release.py",
            ".github/scripts/check_wheel_matrix.py",
            "scripts/check_crate_package.py",
            "scripts/capi_release.py",
            "scripts/check_release_readmes.py",
            "scripts/check_wheel_install.py",
            "scripts/package_capi.py",
            "scripts/package_wasm.py",
            "scripts/build_npm_package.py",
            "scripts/check_npm_package.py",
        ]
        for path in release_inputs:
            self.assertTrue(covers(path, filters["rust"]), path)
            self.assertFalse(covers(path, filters["cpp"]), path)

        self.assertIn(".github/workflows/ci.yml", filters["rust"])
        self.assertIn(".github/workflows/rust-ci.yml", filters["rust"])
        self.assertIn(".github/workflows/cpp-ci.yml", filters["cpp"])
        self.assertIn("cpp/**", filters["cpp"])
        self.assertNotIn(".github/workflows/rust-ci.yml", filters["cpp"])
        self.assertNotIn(".github/requirements-release.txt", filters["cpp"])

        self.assertIn("docs/launch/**", filters["integration"])
        self.assertIn(".github/workflows/integration-ci.yml", filters["integration"])
        self.assertIn("bench/**", filters["integration"])

        untouched = [
            ".github/workflows/claude.yml",
            ".github/workflows/claude-code-review.yml",
            ".github/workflows/bench.yml",
            ".github/workflows/cut-demo-0.2.0.yml",
        ]
        for path in untouched:
            for component, patterns in filters.items():
                self.assertFalse(covers(path, patterns), f"{component} matches {path}")

        rust_refs = script_refs(ROOT / ".github/workflows/rust-ci.yml")
        self.assertTrue(rust_refs, "rust-ci.yml should reference scripts/")
        for ref in rust_refs:
            self.assertTrue(covers(ref, filters["rust"]), ref)
        for name in (
            "publish-crate.yml",
            "publish-rust.yml",
            "publish-wasm.yml",
            "publish-capi.yml",
        ):
            for ref in script_refs(ROOT / ".github/workflows" / name):
                self.assertTrue(covers(ref, filters["rust"]), f"{name} {ref}")

        integration_text = (ROOT / ".github/workflows/integration-ci.yml").read_text()
        for ref in script_refs(ROOT / ".github/workflows/integration-ci.yml"):
            self.assertTrue(covers(ref, filters["integration"]), ref)
        for ref in re.findall(r"\.github/workflows/[A-Za-z0-9_.-]+\.yml", integration_text):
            self.assertTrue(
                covers(ref, filters["integration"]) or ref.endswith("/ci.yml"),
                ref,
            )
            self.assertTrue((ROOT / ref).is_file(), ref)


def covers(path, patterns):
    for pattern in patterns:
        if pattern.endswith("/**"):
            root = pattern[:-3]
            if path == root or path.startswith(root + "/"):
                return True
        elif path == pattern:
            return True
    return False


def script_refs(workflow):
    # Root scripts/ only. bench/compare/scripts/ is covered by bench/**.
    return set(re.findall(
        r"(?<![A-Za-z0-9_./])scripts/[A-Za-z0-9_./-]+\.(?:py|sh)",
        workflow.read_text(),
    ))


if __name__ == "__main__":
    unittest.main()
