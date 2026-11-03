"""Recompute the interrupted attempt's quality counts from retained raw samples."""
from pathlib import Path
import hashlib
import json
import math

root = Path(__file__).parent
summary = json.loads((root / "summary.json").read_text())
manifest = json.loads((root / "criterion-campaign.json").read_text())
samples = [json.loads(line) for line in (root / "criterion-host.jsonl").read_text().splitlines()]
for sample in samples:
    background = [p["cpu"] for p in sample["processes"]
                  if p["pid"] not in sample["excluded_pids"] and p["executable"] != "ps"]
    maximum, total = max(background, default=0), sum(background)
    assert math.isclose(maximum, sample["background_max"], abs_tol=1e-9)
    assert math.isclose(total, sample["background_sum"], abs_tol=1e-9)
    assert sample["quiet"] == (maximum <= 50 and total <= 150)
    assert sample["ac_power"] == ("'AC Power'" in sample["power"])
measured = [s for s in samples if not s["stage"].endswith("preflight")]
preflight = [s for s in samples if s["stage"].endswith("preflight")]
assert len(measured) == summary["process_phase_samples"]
assert len(preflight) == summary["preflight_samples"]
assert sum(not s["quiet"] for s in measured) == summary["process_phase_load_flags"]
assert sum(not s["quiet"] for s in preflight) == summary["preflight_load_flags"]
assert sum(not s["ac_power"] for s in samples) == summary["power_flags_all_samples"]
for event in manifest["events"]:
    block = [s for s in measured if s["stage"] == event["name"]]
    expected = {"samples": len(block), "load_flags": sum(not s["quiet"] for s in block),
                "power_flags": sum(not s["ac_power"] for s in block)}
    assert summary["stages"][event["name"]] == expected
    assert event["load_flags"] == expected["load_flags"]
    assert event["power_flags"] == expected["power_flags"]
    before = [s for s in preflight if s["stage"] == event["name"] + "-preflight"]
    assert len(before) >= 3 and all(s["quiet"] and s["ac_power"] for s in before[-3:])
peaks = {}
for s in measured:
    for p in s["processes"]:
        if p["pid"] not in s["excluded_pids"] and p["executable"] != "ps" and p["cpu"] > 50:
            peaks[p["executable"]] = max(peaks.get(p["executable"], 0), p["cpu"])
assert peaks == summary["background_process_peak_cpu_over_50"]
assert manifest["status"] == "failed" and manifest["error"] == "KeyboardInterrupt()"
assert summary["status"] == "blocked" and summary["process_phase_load_flags"] > 0
assert summary["completed_targets"] == [e["name"] for e in manifest["events"] if e.get("exit_code") == 0]
assert summary["interrupted_target"] == manifest["events"][-1]["name"]
assert all("-A-" in e["name"] for e in manifest["events"]), "No candidate comparison was completed"
for row in summary["criterion_rows_with_artifacts"]:
    path = root / row["path"]
    assert json.loads(path.read_text())["full_id"] == row["benchmark"]
    values = json.loads(path.with_name("sample.json").read_text())
    assert len(values["times"]) == row["sample_count"] == len(values["iters"])
    assert all(math.isfinite(t) and t >= 0 for t in values["times"])
    assert all(math.isfinite(n) and n > 0 for n in values["iters"])
    assert path.with_name("estimates.json").exists() == row["estimates_present"]
provenance = json.loads((root / "controller-provenance.json").read_text())
for change in provenance["changes"]:
    assert hashlib.sha256((root / change["file"]).read_bytes()).hexdigest() == change["sha256"]
records = json.loads((root / "post-stop-validation.json").read_text())["records"]
live_pids = {int(line.split()[0]) for line in records[0]["stdout"].splitlines()}
assert not live_pids.intersection(summary["benchmark_and_controller_pids_absent"])
print(f"PASS: {len(samples)} load/power samples, {len(summary['criterion_rows_with_artifacts'])} retained Criterion rows, controller provenance and recorded cleanup; acceptance remains blocked")
