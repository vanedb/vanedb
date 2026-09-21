#!/usr/bin/env python3
"""Assemble, sign, verify and attach the complete native C ABI release.

No command implicitly publishes. ``publish`` additionally requires GitHub's
push/tag context; workflow_dispatch (even on a tag) cannot satisfy it.
All subprocess arguments are arrays; downloaded filenames never become code.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tempfile
import tomllib
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "vanedb/vanedb"
WORKFLOW = ".github/workflows/publish-capi.yml"
ISSUER = "https://token.actions.githubusercontent.com"
TARGETS = {
    "linux-x86_64": "x86_64-unknown-linux-gnu",
    "linux-aarch64": "aarch64-unknown-linux-gnu",
    "macos-aarch64": "aarch64-apple-darwin",
    "macos-x86_64": "x86_64-apple-darwin",
    "windows-x86_64": "x86_64-pc-windows-msvc",
}
MANIFEST = "CAPI-RELEASE.json"
NOTES = "CAPI-VERIFYING.md"
SUMS = "SHA256SUMS"
BUNDLE = ".sigstore.json"


def run(*args, **kwargs):
    return subprocess.run([str(arg) for arg in args], check=True, text=True, **kwargs)


def output(*args):
    return run(*args, capture_output=True).stdout.strip()


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def identity(ref):
    if not re.fullmatch(r"refs/(?:heads|tags)/[A-Za-z0-9._/-]+", ref):
        raise ValueError("signing requires a full branch or tag ref")
    return f"https://github.com/{REPOSITORY}/{WORKFLOW}@{ref}"


def version_at(root=ROOT):
    versions = {tomllib.loads((root / name / "Cargo.toml").read_text(encoding="utf-8"))["package"]["version"]
                for name in ("vanedb", "vanedb-capi")}
    if len(versions) != 1:
        raise ValueError("core and C ABI versions disagree")
    version = versions.pop()
    if not re.fullmatch(r"\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?", version):
        raise ValueError(f"invalid release version: {version}")
    return version


def source_commit():
    return output("git", "-C", ROOT, "rev-parse", "HEAD")


def check_context(version, event, ref, repository):
    if repository != REPOSITORY:
        raise ValueError("release workflow is restricted to vanedb/vanedb")
    if event == "workflow_dispatch" and ref.startswith("refs/heads/"):
        return False
    if event == "push" and ref == f"refs/tags/vanedb-crate-v{version}":
        return True
    raise ValueError("expected a branch rehearsal or a matching pushed crate tag")


def names(version):
    return sorted(f"vanedb-capi-{version}-{platform}{suffix}"
                  for platform in TARGETS for suffix in (".zip", ".cdx.json"))


def require_files(directory, expected):
    actual = set()
    for path in directory.iterdir():
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"not a regular release asset: {path.name}")
        actual.add(path.name)
    if actual != set(expected):
        raise ValueError(f"asset set mismatch: missing={sorted(set(expected) - actual)}, "
                         f"unexpected={sorted(actual - set(expected))}")


def check_sbom(path, version, platform, commit):
    bom = json.loads(path.read_text(encoding="utf-8"))
    root = bom.get("metadata", {}).get("component", {})
    if (bom.get("bomFormat") != "CycloneDX" or bom.get("specVersion") != "1.5"
            or root.get("name") != "vanedb-capi" or root.get("version") != version):
        raise ValueError(f"wrong CycloneDX package/version in {path.name}")
    if not any(c.get("name") == "vanedb" and c.get("version") == version
               for c in bom.get("components", [])):
        raise ValueError(f"SBOM omits the matching core dependency: {path.name}")
    props = {p["name"]: p["value"] for p in bom["metadata"].get("properties", [])}
    expected = {"vanedb:source-commit": commit, "vanedb:target": TARGETS[platform],
                "vanedb:profile": "capi"}
    if any(props.get(k) != v for k, v in expected.items()):
        raise ValueError(f"SBOM provenance mismatch: {path.name}")
    return bom


def stage_sbom(source, directory, platform, version, commit):
    bom = json.loads(source.read_text(encoding="utf-8"))
    metadata = bom.setdefault("metadata", {})
    props = metadata.setdefault("properties", [])
    for key, value in {"vanedb:source-commit": commit, "vanedb:target": TARGETS[platform],
                       "vanedb:profile": "capi"}.items():
        if any(p.get("name") == key for p in props):
            raise ValueError(f"SBOM already has provenance property {key}")
        props.append({"name": key, "value": value})
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"vanedb-capi-{version}-{platform}.cdx.json"
    path.write_text(json.dumps(bom, indent=2) + "\n", encoding="utf-8")
    check_sbom(path, version, platform, commit)


def check_archive(path, version, platform):
    prefix = f"vanedb-capi-{version}-{platform}/"
    with zipfile.ZipFile(path) as archive:
        entries = archive.namelist()
        if len(entries) != len(set(entries)):
            raise ValueError(f"duplicate zip entries: {path.name}")
        if any(not n.startswith(prefix) or ".." in PurePosixPath(n).parts or "\\" in n
               for n in entries):
            raise ValueError(f"unsafe or mismatched archive paths: {path.name}")
        required = ["include/vanedb_rs_capi.h", "compatibility.json", "LICENSE",
                    "lib/cmake/vanedb/vanedbConfig.cmake", "lib/pkgconfig/vanedb.pc"]
        libs = (["vanedb_capi.dll", "vanedb_capi.lib", "vanedb_capi.dll.lib"]
                if platform.startswith("windows") else
                ["libvanedb_capi.a", "libvanedb_capi.dylib" if platform.startswith("macos")
                 else "libvanedb_capi.so"])
        for name in required + ["lib/" + lib for lib in libs]:
            if prefix + name not in entries or archive.getinfo(prefix + name).file_size == 0:
                raise ValueError(f"missing/empty {name} in {path.name}")
        metadata = json.loads(archive.read(prefix + "compatibility.json"))
        if metadata.get("platform") != platform or not metadata.get("requirements"):
            raise ValueError(f"archive compatibility target mismatch: {path.name}")
        header = archive.read(prefix + "include/vanedb_rs_capi.h").decode()
        if f'#define VANEDB_RS_VERSION "{version}"' not in header:
            raise ValueError(f"archive header version mismatch: {path.name}")
        return metadata


def verification_notes(version, ref, commit):
    subject = identity(ref)
    return f"""## Signed C ABI assets — {version}

Source commit: `{commit}`. The five desktop archives and their target-specific
CycloneDX 1.5 SBOMs, `CAPI-RELEASE.json`, this file and `SHA256SUMS` each have a
`.sigstore.json` keyless signature bundle. The SBOM describes Cargo dependencies
for the C ABI's default features on its recorded target; OS-provided libraries
are listed separately in each archive's `compatibility.json`.

Download the assets into an empty directory. Install cosign v3.1.3 from
https://github.com/sigstore/cosign/releases/tag/v3.1.3 and authenticate the
checksums before using any archive:

```sh
cosign verify-blob SHA256SUMS --bundle SHA256SUMS.sigstore.json \\
  --certificate-identity '{subject}' \\
  --certificate-oidc-issuer '{ISSUER}' \\
  --certificate-github-workflow-sha '{commit}'
sha256sum --check SHA256SUMS  # macOS: shasum -a 256 --check SHA256SUMS
```

For a single archive, use the same `cosign verify-blob` command with its `.zip`
and matching `.zip.sigstore.json`. Verify every payload/bundle and enforce the
complete five-platform set with the script from the approved source checkout:

```sh
python3 scripts/capi_release.py verify --directory /path/to/downloads \\
  --ref '{ref}' --version '{version}' --commit '{commit}'
```

The expected identity above is exact. A branch-rehearsal signature is not a
release-tag signature. The certificate must also bind the expected source SHA.
Issuer, certificate and transparency-log verification
must all succeed; do not disable them. Neither checksums alone nor an arbitrary
GitHub Actions identity authenticates a release.
"""


def assemble(source, directory, version, commit, ref):
    expected = names(version)
    require_files(source, expected + [n + ".sha256" for n in expected if n.endswith(".zip")])
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("assembly output must be empty")
    directory.mkdir(parents=True, exist_ok=True)
    platforms = {}
    for platform in TARGETS:
        name = f"vanedb-capi-{version}-{platform}.zip"
        if (source / (name + ".sha256")).read_text(encoding="utf-8") != f"{digest(source / name)}  {name}\n":
            raise ValueError(f"native artifact checksum mismatch: {name}")
        platforms[platform] = check_archive(source / name, version, platform)
        check_sbom(source / name.replace(".zip", ".cdx.json"), version, platform, commit)
    for name in expected:
        shutil.copyfile(source / name, directory / name)
    manifest = {"schema": 1, "version": version, "source_commit": commit,
                "signing_identity": identity(ref), "platforms": platforms,
                "sha256": {name: digest(directory / name) for name in expected}}
    (directory / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (directory / NOTES).write_text(verification_notes(version, ref, commit), encoding="utf-8")
    payloads = expected + [MANIFEST, NOTES]
    (directory / SUMS).write_text("".join(f"{digest(directory / name)}  {name}\n"
                                         for name in sorted(payloads)), encoding="utf-8")


def verify_contents(directory, version, commit, ref, signed):
    payloads = names(version) + [MANIFEST, NOTES, SUMS]
    require_files(directory, payloads + ([n + BUNDLE for n in payloads] if signed else []))
    expected_sums = "".join(f"{digest(directory / name)}  {name}\n"
                            for name in sorted(n for n in payloads if n != SUMS))
    if (directory / SUMS).read_text(encoding="utf-8") != expected_sums:
        raise ValueError("SHA256SUMS does not match the complete release payload")
    manifest = json.loads((directory / MANIFEST).read_text(encoding="utf-8"))
    if (manifest.get("schema") != 1 or manifest.get("version") != version
            or manifest.get("source_commit") != commit
            or manifest.get("signing_identity") != identity(ref)
            or set(manifest.get("platforms", {})) != set(TARGETS)
            or manifest.get("sha256") != {n: digest(directory / n) for n in names(version)}):
        raise ValueError("release manifest identity or inventory mismatch")
    if (directory / NOTES).read_text(encoding="utf-8") != verification_notes(version, ref, commit):
        raise ValueError("verification notes do not match the release identity")
    for platform in TARGETS:
        base = f"vanedb-capi-{version}-{platform}"
        if check_archive(directory / (base + ".zip"), version, platform) != manifest["platforms"][platform]:
            raise ValueError("archive metadata differs from release manifest")
        check_sbom(directory / (base + ".cdx.json"), version, platform, commit)
    return payloads


def sign(directory, version, commit, ref):
    payloads = verify_contents(directory, version, commit, ref, signed=False)
    for name in payloads:
        run("cosign", "sign-blob", "--yes", "--bundle", directory / (name + BUNDLE), directory / name)
    verify(directory, version, commit, ref)


def verify(directory, version, commit, ref):
    payloads = verify_contents(directory, version, commit, ref, signed=True)
    for name in payloads:
        run("cosign", "verify-blob", "--bundle", directory / (name + BUNDLE),
            "--certificate-identity", identity(ref), "--certificate-oidc-issuer", ISSUER,
            "--certificate-github-workflow-sha", commit,
            directory / name)
    return payloads


def publish(directory, version, commit, ref):
    if not check_context(version, os.environ.get("GITHUB_EVENT_NAME", ""), ref,
                         os.environ.get("GITHUB_REPOSITORY", "")):
        raise ValueError("rehearsals cannot publish")
    if os.environ.get("GITHUB_REF") != ref:
        raise ValueError("requested ref differs from GitHub event ref")
    if source_commit() != commit:
        raise ValueError("checkout differs from approved source")
    run("git", "fetch", "--no-tags", "origin", "refs/heads/main:refs/remotes/origin/main", cwd=ROOT)
    run("git", "merge-base", "--is-ancestor", commit, "origin/main", cwd=ROOT)
    run("git", "fetch", "--no-tags", "origin", ref, cwd=ROOT)
    tag_commit = output("git", "-C", ROOT, "rev-parse", "FETCH_HEAD^{commit}")
    if tag_commit != commit:
        raise ValueError("release tag differs from approved source")
    payloads = verify(directory, version, commit, ref)
    assets = sorted(payloads + [n + BUNDLE for n in payloads])
    tag = ref.removeprefix("refs/tags/")
    prerelease = "-" in version
    # Only a 404 proves absence; permission/network errors are not permission
    # to create another release or conceal a partial previous publication.
    response = subprocess.run(["gh", "api", f"repos/{REPOSITORY}/releases/tags/{tag}"],
                              text=True, capture_output=True)
    if response.returncode:
        if "HTTP 404" not in response.stderr:
            raise ValueError(f"cannot inspect release: {response.stderr}")
        run("gh", "release", "create", tag, "--repo", REPOSITORY, "--verify-tag", "--draft",
            "--title", f"VaneDB {version}", "--notes-file", directory / NOTES,
            *(["--prerelease"] if prerelease else []))
        release = {"draft": True, "prerelease": prerelease,
                   "body": (directory / NOTES).read_text(encoding="utf-8"), "assets": []}
    else:
        release = json.loads(response.stdout)
    if bool(release.get("prerelease")) != prerelease:
        raise ValueError("existing release prerelease status disagrees with the version")
    existing = {asset["name"] for asset in release["assets"]}
    unexpected = {n for n in existing if n.startswith("vanedb-capi-")} - set(assets)
    if unexpected:
        raise ValueError(f"unexpected C ABI release assets: {sorted(unexpected)}")
    with tempfile.TemporaryDirectory(prefix="vanedb-release-verify-") as temporary:
        downloaded = Path(temporary)
        for name in assets:
            if name in existing:
                run("gh", "release", "download", tag, "--repo", REPOSITORY,
                    "--pattern", name, "--dir", downloaded)
                if digest(downloaded / name) != digest(directory / name):
                    raise ValueError(f"refusing to overwrite existing release asset: {name}")
            else:
                run("gh", "release", "upload", tag, directory / name, "--repo", REPOSITORY)
                run("gh", "release", "download", tag, "--repo", REPOSITORY,
                    "--pattern", name, "--dir", downloaded)
        verify(downloaded, version, commit, ref)
    # Preserve human-authored release notes and append exact verification
    # instructions once. No --clobber and no moving/recreating published tags.
    body = release.get("body") or ""
    notes = (directory / NOTES).read_text(encoding="utf-8")
    if notes not in body:
        with tempfile.TemporaryDirectory() as temporary:
            combined = Path(temporary) / "notes.md"
            combined.write_text(body + "\n\n" + notes, encoding="utf-8")
            run("gh", "release", "edit", tag, "--repo", REPOSITORY, "--notes-file", combined)
    if release.get("draft"):
        run("gh", "release", "edit", tag, "--repo", REPOSITORY, "--draft=false")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["identity", "stage-sbom", "assemble", "sign", "verify", "publish"])
    parser.add_argument("--directory", type=Path, default=ROOT / "target/c-release")
    parser.add_argument("--source", type=Path, default=ROOT / "target/c-artifacts")
    parser.add_argument("--platform", choices=TARGETS)
    parser.add_argument("--version")
    parser.add_argument("--commit")
    parser.add_argument("--ref", default=os.environ.get("GITHUB_REF", ""))
    args = parser.parse_args()
    version = args.version or version_at()
    commit = args.commit or source_commit()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("expected a full source commit")
    if args.command == "identity":
        check_context(version, os.environ.get("GITHUB_EVENT_NAME", ""), args.ref,
                      os.environ.get("GITHUB_REPOSITORY", ""))
        if os.environ.get("GITHUB_OUTPUT"):
            with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as handle:
                handle.write(f"version={version}\ncommit={commit}\n")
        print(f"validated C ABI {version}, source {commit}")
    elif args.command == "stage-sbom":
        if not args.platform:
            parser.error("stage-sbom requires --platform")
        stage_sbom(args.source, args.directory, args.platform, version, commit)
    elif args.command == "assemble":
        assemble(args.source, args.directory, version, commit, args.ref)
    else:
        {"sign": sign, "verify": verify, "publish": publish}[args.command](
            args.directory, version, commit, args.ref)


if __name__ == "__main__":
    main()
