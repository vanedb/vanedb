"""Reconcile retained workflow and regression outcomes; this is not release sign-off."""
from pathlib import Path
import json

root = Path(__file__).parent
original = "e1761dcf4258011b446a361dc529da082ea2fc94"
fixed = "6e4fc2d88f2f248cb06becb24bcc0ef228323d22"
runs = {41602796987: original, 41602801159: original, 41602805577: original,
        41604609578: fixed, 41604725593: fixed}
publication_jobs = []
for run, source in runs.items():
    record = json.loads((root / f"run-{run}.json").read_text())
    assert record["headSha"] == source and record["event"] == "workflow_dispatch"
    assert record["status"] == "completed" and record["conclusion"] == "success"
    assert all(j["status"] == "completed" and j["conclusion"] in ("success", "skipped")
               for j in record["jobs"])
    for job in record["jobs"]:
        if job["name"].startswith("Publish ") or "Attach signed C assets" in job["name"]:
            assert job["conclusion"] == "skipped"
            publication_jobs.append((run, job["name"]))
    if run == 41604725593:
        gate = [j for j in record["jobs"] if j["name"] == "Required CI Gate"]
        assert len(gate) == 1 and gate[0]["conclusion"] == "success"
assert len(publication_jobs) == 6
inventory = json.loads((root / "artifact-inventory-validation.json").read_text())
assert inventory["candidate"] == original
assert inventory["python_wheel_count"] == 28 and inventory["python_sdist_count"] == 1
assert len(inventory["wheels"]) == 28 and len({w["file"] for w in inventory["wheels"]}) == 28
assert inventory["capi_signature_verifier_exit"] == 0
assert json.loads((root / "capi-verification.json").read_text())["exit_code"] == 0
assert json.loads((root / "cross-realm-red.json").read_text())["exit_code"] != 0
assert "foreign metadata" in (root / "cross-realm-red.stderr.log").read_text()
for step in json.loads((root / "fix-validation.json").read_text()):
    assert step["exit_code"] == 0
for step in json.loads((root / "fix-final-validation.json").read_text())["records"]:
    assert step["exit_code"] == 0
remote = json.loads((root / "fixed-remote-npm-verification.json").read_text())
assert remote["candidate"] == fixed and remote["exit_code"] == 0
repro = json.loads((root / "review-reproducer/results.json").read_text())
assert repro["candidate"] == original and len(repro["cases"]) == 4
for case in repro["cases"]:
    assert case["exit_code"] == (1 if case["case"].endswith("foreign") else 0)
assert all(case["exit_code"] == 0 for case in
           json.loads((root / "review-reproducer/fixed-results.json").read_text()))
print("PASS: exact-source run outcomes, six skipped publication jobs, artifact counts and RED/GREEN regression records; release acceptance remains outstanding")
