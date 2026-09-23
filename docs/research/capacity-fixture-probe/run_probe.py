"""Run each case as a fresh process and retain all outputs; never record timings."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import tomllib
from verify_files import verify

ROOT = Path(__file__).resolve().parent
CANDIDATE = ROOT.parent / "vanedb-020-integration"
FIXTURE = ROOT.parent / "vanedb-pr-214/bench/compare/fixtures/embeddings.vnef"
SHA = "35758c6ccac0b9e657be8d80e84d1744378df120"
FIXTURE_SHA = "4676911521f13dc6592d102cf3b4fcace85c6b62b3011e9bd57557dfc9c01f2d"


def capture(cmd, cwd=ROOT):
    return subprocess.check_output(cmd, cwd=cwd, text=True).strip()


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def source_check():
    require(capture(["git", "rev-parse", "HEAD"], CANDIDATE) == SHA, "source HEAD mismatch")
    require(not capture(["git", "status", "--porcelain", "--untracked-files=no"], CANDIDATE), "source has tracked changes")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts", nargs="+", type=int, default=[8192, 10000, 100000])
    parser.add_argument("--kinds", nargs="+", choices=["flat", "disk", "approx"], default=["flat", "disk", "approx"])
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    source_check()
    with FIXTURE.open("rb") as f:
        require(hashlib.file_digest(f, "sha256").hexdigest() == FIXTURE_SHA, "fixture checksum mismatch")
    out = ROOT / args.out
    out.mkdir(exist_ok=True)
    binary = ROOT / "target/release/capacity-fixture-probe"
    require(binary.exists(), "build probe first")
    require(binary.stat().st_mtime >= max((ROOT / p).stat().st_mtime for p in ["src/main.rs", "Cargo.toml", "Cargo.lock"]), "binary predates probe source/lock")
    candidate_lock = tomllib.loads((CANDIDATE / "Cargo.lock").read_text())
    probe_lock = tomllib.loads((ROOT / "Cargo.lock").read_text())
    cargo_metadata = json.loads(capture(["cargo", "metadata", "--locked", "--offline", "--format-version", "1"]))
    nodes = {node['id']: node for node in cargo_metadata['resolve']['nodes']}
    engine = next(p for p in cargo_metadata['packages'] if p['name'] == 'vanedb')
    reachable, pending = set(), [engine['id']]
    while pending:
        ident = pending.pop()
        if ident not in reachable:
            reachable.add(ident)
            pending.extend(nodes[ident]['dependencies'])
    engine_packages = {(p['name'], p['version'], p.get('source')) for p in cargo_metadata['packages'] if p['id'] in reachable}
    candidate_packages = {(p['name'], p['version'], p.get('source'), p.get('checksum')) for p in candidate_lock['package']}
    differences = [p for p in probe_lock['package'] if (p['name'], p['version'], p.get('source')) in engine_packages and
                   (p['name'], p['version'], p.get('source'), p.get('checksum')) not in candidate_packages]
    require(not differences, f"candidate dependency lock drift: {differences}")
    metadata = dict(source_sha=SHA, fixture_sha256=FIXTURE_SHA, fixture=str(FIXTURE),
                    fixture_metadata=json.loads((CANDIDATE / "bench/compare/fixtures/metadata.json").read_text()),
                    platform=platform.platform(), machine=platform.machine(),
                    rustc=capture(["rustc", "--version", "--verbose"]), cargo=capture(["cargo", "--version"]),
                    rustc_cfg=capture(["rustc", "--print", "cfg"]), binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                    engine_dependency_lock_differences=differences,
                    scope="capacity study Q2 requested live heap and graph geometry only; no timings/RSS/device limits",
                    environment={name: __import__('os').environ.get(name) for name in ["RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CARGO_BUILD_TARGET"]},
                    source_files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT / "src/main.rs", ROOT / "Cargo.toml", ROOT / "Cargo.lock", ROOT / "verify_files.py", ROOT / "run_probe.py"]})
    metadata_path = out / "metadata.json"
    require(not metadata_path.exists(), "use a fresh output directory")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    (out / "candidate-Cargo.lock").write_bytes((CANDIDATE / "Cargo.lock").read_bytes())
    (out / "probe-Cargo.lock").write_bytes((ROOT / "Cargo.lock").read_bytes())
    (out / "cargo-metadata.json").write_text(json.dumps(cargo_metadata, indent=2) + "\n")
    for n in args.counts:
        for kind in args.kinds:
            stem = f"{kind}-{n}"
            file = out / f"{stem}.vndb"
            cmd = [str(binary), kind, str(n), str(FIXTURE), str(file)]
            (out / f"{stem}.command.json").write_text(json.dumps(cmd, indent=2) + "\n")
            print(f"Running {kind} n={n} in fresh process", flush=True)
            with (out / f"{stem}.json").open("w") as stdout, (out / f"{stem}.stderr.log").open("w") as stderr:
                subprocess.run(cmd, check=True, stdout=stdout, stderr=stderr, cwd=ROOT)
            result = json.loads((out / f"{stem}.json").read_text())
            if kind in ("flat", "disk"):
                buckets = 1 << ((n * 8 + 6) // 7 - 1).bit_length()
                map_bytes = result['calibration_map_requested_bytes']
                require(map_bytes - max(4, buckets) * 17 in (8, 16), "unexpected hash table control-group tail")
                expected = map_bytes + (n * (8 + 768 * 4) if kind == "flat" else 0)
                require(result['built_delta_bytes'] == expected, "flat/disk measured heap differs from independent formula")
                require(result['retained_after_drop_delta_bytes'] == 0, "unexpected flat/disk retained allocations")
            if kind != "flat":
                check = verify(file, FIXTURE, kind, n)
                (out / f"{stem}.file-verification.json").write_text(json.dumps(check, indent=2) + "\n")
            print(f"PASS {kind} n={n}: built delta={result['built_delta_bytes']}; after-drop residual={result['retained_after_drop_delta_bytes']}", flush=True)
    source_check()
    (out / "complete.json").write_text(json.dumps(dict(source_still_clean=True, source_sha=SHA, counts=args.counts, kinds=args.kinds), indent=2) + "\n")


if __name__ == "__main__":
    main()
