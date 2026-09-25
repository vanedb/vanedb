#!/usr/bin/env python3
"""Adversarial release-inventory, signing-boundary and publication-guard tests."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import capi_release as release

VERSION = "0.2.0"
COMMIT = "a" * 40
REF = "refs/tags/vanedb-crate-v0.2.0"


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "native"
        self.source.mkdir()
        self.directory = self.root / "release"
        for platform in release.TARGETS:
            base = f"vanedb-capi-{VERSION}-{platform}"
            filename = base + ".zip"
            libs = (["vanedb_capi.dll", "vanedb_capi.lib", "vanedb_capi.dll.lib"]
                    if platform.startswith("windows") else
                    ["libvanedb_capi.a", "libvanedb_capi.dylib" if platform.startswith("macos")
                     else "libvanedb_capi.so"])
            with zipfile.ZipFile(self.source / filename, "w") as archive:
                for name in ["LICENSE", "lib/cmake/vanedb/vanedbConfig.cmake",
                             "lib/pkgconfig/vanedb.pc"] + ["lib/" + lib for lib in libs]:
                    archive.writestr(base + "/" + name, "nonempty fixture")
                archive.writestr(base + "/include/vanedb_rs_capi.h",
                                 '#define VANEDB_RS_VERSION "0.2.0"\n')
                archive.writestr(base + "/compatibility.json", json.dumps(
                    {"platform": platform, "requirements": {"fixture": True}}))
            (self.source / (filename + ".sha256")).write_text(
                f"{release.digest(self.source / filename)}  {filename}\n")
            bom = {"bomFormat": "CycloneDX", "specVersion": "1.5",
                   "metadata": {"component": {"name": "vanedb-capi", "version": VERSION}},
                   "components": [{"name": "vanedb", "version": VERSION}]}
            path = self.root / "generator-output.json"
            path.write_text(json.dumps(bom))
            release.stage_sbom(path, self.source, platform, VERSION, COMMIT)

    def assemble(self):
        release.assemble(self.source, self.directory, VERSION, COMMIT, REF)

    def unsigned(self):
        return release.verify_contents(self.directory, VERSION, COMMIT, REF, signed=False)

    def dummy_bundles(self):
        for name in release.names(VERSION) + [release.MANIFEST, release.NOTES, release.SUMS]:
            (self.directory / (name + release.BUNDLE)).write_text("fixture signature")

    def test_sbom_uses_utf8_on_a_cp1252_host(self):
        # cargo-cyclonedx emits UTF-8 names. U+0101's UTF-8 contains 0x81,
        # undefined in CP1252: exactly the Windows rehearsal failure.
        source = self.root / "unicode-sbom.json"
        bom = {"bomFormat": "CycloneDX", "specVersion": "1.5",
               "metadata": {"component": {"name": "vanedb-capi", "version": VERSION},
                            "authors": [{"name": "Māris"}]},
               "components": [{"name": "vanedb", "version": VERSION}]}
        source.write_text(json.dumps(bom, ensure_ascii=False), encoding="utf-8")
        read_text, write_text = Path.read_text, Path.write_text
        def read_in_legacy_locale(path, encoding=None, **kwargs):
            return read_text(path, encoding=encoding or "cp1252", **kwargs)
        def write_in_legacy_locale(path, text, encoding=None, **kwargs):
            return write_text(path, text, encoding=encoding or "cp1252", **kwargs)
        with patch.object(Path, "read_text", read_in_legacy_locale), \
                patch.object(Path, "write_text", write_in_legacy_locale):
            release.stage_sbom(source, self.source, "windows-x86_64", VERSION, COMMIT)
            self.assemble()
            self.unsigned()
        path = self.directory / "vanedb-capi-0.2.0-windows-x86_64.cdx.json"
        self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["metadata"]["authors"][0]["name"], "Māris")
        self.assertIn("—", (self.directory / release.NOTES).read_text(encoding="utf-8"))

    def test_complete_inventory_has_all_five_archives_and_target_sboms(self):
        self.assemble()
        self.assertEqual(len(self.unsigned()), 13)
        manifest = json.loads((self.directory / release.MANIFEST).read_text())
        self.assertEqual(set(manifest["platforms"]), set(release.TARGETS))

    def test_missing_platform_fails_before_signing(self):
        (self.source / "vanedb-capi-0.2.0-windows-x86_64.zip").unlink()
        with self.assertRaisesRegex(ValueError, "missing"):
            self.assemble()

    def test_extra_old_version_fails(self):
        (self.source / "vanedb-capi-0.1.1-linux-x86_64.zip").write_bytes(b"old")
        with self.assertRaisesRegex(ValueError, "unexpected"):
            self.assemble()

    def test_symlink_cannot_supply_an_asset(self):
        path = self.source / "vanedb-capi-0.2.0-linux-x86_64.zip"
        original = self.root / "archive"
        path.rename(original)
        path.symlink_to(original)
        with self.assertRaisesRegex(ValueError, "regular"):
            self.assemble()

    def test_transport_corruption_rejects_original_native_checksum(self):
        with (self.source / "vanedb-capi-0.2.0-linux-x86_64.zip").open("ab") as handle:
            handle.write(b"changed")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.assemble()

    def test_wrong_target_and_source_sbom_fail(self):
        path = self.source / "vanedb-capi-0.2.0-linux-x86_64.cdx.json"
        for key in ("vanedb:target", "vanedb:source-commit", "vanedb:profile"):
            original = path.read_text()
            bom = json.loads(original)
            next(p for p in bom["metadata"]["properties"] if p["name"] == key)["value"] = "wrong"
            path.write_text(json.dumps(bom))
            with self.assertRaisesRegex(ValueError, "provenance"):
                self.assemble()
            path.write_text(original)

    def test_missing_core_dependency_fails(self):
        path = self.source / "vanedb-capi-0.2.0-linux-x86_64.cdx.json"
        bom = json.loads(path.read_text())
        bom["components"] = []
        path.write_text(json.dumps(bom))
        with self.assertRaisesRegex(ValueError, "core dependency"):
            self.assemble()

    def test_bad_zip_header_version_fails_even_if_checksum_updated(self):
        path = self.source / "vanedb-capi-0.2.0-linux-x86_64.zip"
        with zipfile.ZipFile(path) as archive:
            contents = {n: archive.read(n) for n in archive.namelist()}
        key = "vanedb-capi-0.2.0-linux-x86_64/include/vanedb_rs_capi.h"
        contents[key] = b'#define VANEDB_RS_VERSION "0.1.1"\n'
        with zipfile.ZipFile(path, "w") as archive:
            for name, content in contents.items():
                archive.writestr(name, content)
        path.with_suffix(".zip.sha256").write_text(f"{release.digest(path)}  {path.name}\n")
        with self.assertRaisesRegex(ValueError, "header version"):
            self.assemble()

    def test_modified_payload_and_incomplete_checksum_list_fail(self):
        self.assemble()
        sums = self.directory / release.SUMS
        original = sums.read_text()
        sums.write_text("\n".join(original.splitlines()[1:]) + "\n")
        with self.assertRaisesRegex(ValueError, "SHA256SUMS"):
            self.unsigned()
        sums.write_text(original)
        (self.directory / release.NOTES).write_text("altered instructions")
        with self.assertRaisesRegex(ValueError, "SHA256SUMS"):
            self.unsigned()

    def test_exact_source_and_signer_identity_required(self):
        self.assemble()
        for commit, ref in [("b" * 40, REF), (COMMIT, "refs/heads/main")]:
            with self.assertRaisesRegex(ValueError, "manifest"):
                release.verify_contents(self.directory, VERSION, commit, ref, signed=False)

    def test_empty_or_missing_signature_never_skipped(self):
        self.assemble()
        self.dummy_bundles()
        (self.directory / (release.SUMS + release.BUNDLE)).unlink()
        with patch.object(release, "run") as run:
            with self.assertRaisesRegex(ValueError, "missing"):
                release.verify(self.directory, VERSION, COMMIT, REF)
            run.assert_not_called()

    def test_every_payload_uses_exact_identity_and_issuer(self):
        self.assemble()
        self.dummy_bundles()
        with patch.object(release, "run") as run:
            release.verify(self.directory, VERSION, COMMIT, REF)
        self.assertEqual(run.call_count, 13)
        verified = set()
        for call in run.call_args_list:
            args = call.args
            self.assertEqual(args[0:2], ("cosign", "verify-blob"))
            self.assertEqual(args[args.index("--certificate-identity") + 1], release.identity(REF))
            self.assertEqual(args[args.index("--certificate-oidc-issuer") + 1], release.ISSUER)
            self.assertEqual(args[args.index("--certificate-github-workflow-sha") + 1], COMMIT)
            self.assertFalse(any("ignore" in str(arg) or "insecure" in str(arg) for arg in args))
            verified.add(args[-1].name)
        self.assertEqual(verified, set(release.names(VERSION) + [release.MANIFEST, release.NOTES, release.SUMS]))

    def test_failed_cosign_verification_aborts(self):
        self.assemble()
        self.dummy_bundles()
        with patch.object(release, "run", side_effect=RuntimeError("invalid signature")):
            with self.assertRaisesRegex(RuntimeError, "invalid signature"):
                release.verify(self.directory, VERSION, COMMIT, REF)

    def test_dispatch_tag_or_branch_cannot_publish(self):
        for ref in (REF, "refs/heads/main"):
            with patch.dict(os.environ, {"GITHUB_EVENT_NAME": "workflow_dispatch",
                                       "GITHUB_REPOSITORY": release.REPOSITORY,
                                       "GITHUB_REF": ref}), patch.object(release, "run") as run:
                with self.assertRaises(ValueError):
                    release.publish(self.directory, VERSION, COMMIT, ref)
                run.assert_not_called()

    def test_wrong_repo_version_and_event_rejected(self):
        cases = [("push", REF, "fork/vanedb"), ("push", "refs/tags/vanedb-crate-v0.1.1", release.REPOSITORY),
                 ("pull_request", REF, release.REPOSITORY), ("push", "refs/heads/main", release.REPOSITORY)]
        for event, ref, repo in cases:
            with self.assertRaises(ValueError):
                release.check_context(VERSION, event, ref, repo)
        self.assertFalse(release.check_context(VERSION, "workflow_dispatch", "refs/heads/rehearsal", release.REPOSITORY))
        self.assertTrue(release.check_context(VERSION, "push", REF, release.REPOSITORY))

    def test_release_publication_never_overwrites_different_remote_asset(self):
        self.assemble()
        self.dummy_bundles()
        asset = "CAPI-RELEASE.json"
        response = release.subprocess.CompletedProcess([], 0, json.dumps({"draft": False, "body": "human notes",
                                                     "assets": [{"name": asset}]}), "")
        def execute(*args, **kwargs):
            if args[:3] == ("gh", "release", "download"):
                directory = Path(args[args.index("--dir") + 1])
                name = args[args.index("--pattern") + 1]
                (directory / name).write_text("different remote content")
        with patch.dict(os.environ, {"GITHUB_EVENT_NAME": "push", "GITHUB_REF": REF,
                                     "GITHUB_REPOSITORY": release.REPOSITORY}), \
                patch.object(release, "source_commit", return_value=COMMIT), \
                patch.object(release, "output", return_value=COMMIT), \
                patch.object(release, "verify", return_value=release.names(VERSION) + [release.MANIFEST, release.NOTES, release.SUMS]), \
                patch.object(release.subprocess, "run", return_value=response), \
                patch.object(release, "run", side_effect=execute) as run:
            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                release.publish(self.directory, VERSION, COMMIT, REF)
        self.assertFalse(any(c.args[:3] == ("gh", "release", "upload") for c in run.call_args_list))
        self.assertFalse(any("--clobber" in c.args for c in run.call_args_list))


    def test_publish_new_draft_verifies_remote_bytes_before_making_it_public(self):
        self.assemble()
        self.dummy_bundles()
        payloads = release.names(VERSION) + [release.MANIFEST, release.NOTES, release.SUMS]
        response = release.subprocess.CompletedProcess([], 1, "", "gh: Not Found (HTTP 404)")
        remote = {}
        def execute(*args, **kwargs):
            if args[:3] == ("gh", "release", "upload"):
                remote[Path(args[4]).name] = Path(args[4]).read_bytes()
            elif args[:3] == ("gh", "release", "download"):
                name = args[args.index("--pattern") + 1]
                (Path(args[args.index("--dir") + 1]) / name).write_bytes(remote[name])
        checked = []
        def verify(directory, *args):
            checked.append(directory)
            release.verify_contents(directory, VERSION, COMMIT, REF, signed=True)
            return payloads
        with patch.dict(os.environ, {"GITHUB_EVENT_NAME": "push", "GITHUB_REF": REF,
                                     "GITHUB_REPOSITORY": release.REPOSITORY}), \
                patch.object(release, "source_commit", return_value=COMMIT), \
                patch.object(release, "output", return_value=COMMIT), \
                patch.object(release, "verify", side_effect=verify), \
                patch.object(release.subprocess, "run", return_value=response), \
                patch.object(release, "run", side_effect=execute) as run:
            release.publish(self.directory, VERSION, COMMIT, REF)
        self.assertEqual(len(remote), 26)
        self.assertEqual(len(checked), 2)
        self.assertEqual(run.call_args_list[-1].args,
                         ("gh", "release", "edit", "vanedb-crate-v0.2.0", "--repo", release.REPOSITORY, "--draft=false"))
        create = next(c for c in run.call_args_list if c.args[:3] == ("gh", "release", "create"))
        self.assertIn("--draft", create.args)
        self.assertIn("--verify-tag", create.args)
        self.assertNotIn("--prerelease", create.args)

    def test_prerelease_tag_creates_a_prerelease_and_refuses_stable_mismatch(self):
        self.assemble()
        version = "0.2.0-rc.1"
        ref = "refs/tags/vanedb-crate-v" + version
        for existing in (False, True):
            response = (release.subprocess.CompletedProcess([], 0, json.dumps(
                {"draft": False, "prerelease": False, "assets": []}), "") if existing else
                release.subprocess.CompletedProcess([], 1, "", "gh: Not Found (HTTP 404)"))
            with self.subTest(existing=existing), patch.dict(os.environ,
                    {"GITHUB_EVENT_NAME": "push", "GITHUB_REF": ref,
                     "GITHUB_REPOSITORY": release.REPOSITORY}), \
                    patch.object(release, "source_commit", return_value=COMMIT), \
                    patch.object(release, "output", return_value=COMMIT), \
                    patch.object(release, "verify", return_value=[]), \
                    patch.object(release.subprocess, "run", return_value=response), \
                    patch.object(release, "run") as run:
                if existing:
                    with self.assertRaisesRegex(ValueError, "prerelease status"):
                        release.publish(self.directory, version, COMMIT, ref)
                    self.assertFalse(any(c.args[:2] == ("gh", "release") for c in run.call_args_list))
                else:
                    release.publish(self.directory, version, COMMIT, ref)
                    create = next(c for c in run.call_args_list if c.args[:3] == ("gh", "release", "create"))
                    self.assertIn("--prerelease", create.args)

    def test_api_failure_is_not_treated_as_missing_release(self):
        self.assemble()
        self.dummy_bundles()
        response = release.subprocess.CompletedProcess([], 1, "", "gh: forbidden (HTTP 403)")
        with patch.dict(os.environ, {"GITHUB_EVENT_NAME": "push", "GITHUB_REF": REF,
                                     "GITHUB_REPOSITORY": release.REPOSITORY}), \
                patch.object(release, "source_commit", return_value=COMMIT), \
                patch.object(release, "output", return_value=COMMIT), \
                patch.object(release, "verify", return_value=[]), \
                patch.object(release.subprocess, "run", return_value=response), \
                patch.object(release, "run") as run:
            with self.assertRaisesRegex(ValueError, "cannot inspect release"):
                release.publish(self.directory, VERSION, COMMIT, REF)
        self.assertFalse(any(c.args[:3] == ("gh", "release", "create") for c in run.call_args_list))

    def test_moved_remote_tag_fails_before_signatures_or_release_write(self):
        with patch.dict(os.environ, {"GITHUB_EVENT_NAME": "push", "GITHUB_REF": REF,
                                     "GITHUB_REPOSITORY": release.REPOSITORY}), \
                patch.object(release, "source_commit", return_value=COMMIT), \
                patch.object(release, "output", return_value="b" * 40), \
                patch.object(release, "verify") as verify, patch.object(release, "run") as run:
            with self.assertRaisesRegex(ValueError, "tag differs"):
                release.publish(self.directory, VERSION, COMMIT, REF)
            verify.assert_not_called()
        self.assertTrue(all(c.args[0] == "git" for c in run.call_args_list))


class WorkflowTests(unittest.TestCase):
    def test_matrix_guards_permissions_and_pins(self):
        import yaml
        workflow = yaml.safe_load((release.ROOT / release.WORKFLOW).read_text())
        jobs = workflow["jobs"]
        matrix = jobs["native"]["strategy"]["matrix"]["include"]
        self.assertEqual({m["platform"]: m["target"] for m in matrix}, release.TARGETS)
        self.assertEqual(workflow["permissions"], {"contents": "read"})
        self.assertEqual(jobs["sign"]["permissions"], {"contents": "read", "id-token": "write"})
        self.assertEqual(jobs["publish"]["permissions"], {"contents": "write"})
        self.assertEqual(jobs["publish"]["if"],
                         "github.event_name == 'push' && startsWith(github.ref, 'refs/tags/vanedb-crate-v')")
        self.assertEqual(set(jobs["sign"]["needs"]), {"identity", "native"})
        self.assertEqual(set(jobs["publish"]["needs"]), {"identity", "sign"})
        self.assertEqual(jobs["publish"]["environment"], "crates-io")
        for job in jobs.values():
            for step in job["steps"]:
                if "uses" in step:
                    self.assertRegex(step["uses"], r"@[0-9a-f]{40}$")
        events = workflow.get("on", workflow.get(True))
        self.assertEqual(events, {"workflow_call": None})
        caller = yaml.safe_load((release.ROOT / ".github/workflows/publish-crate.yml").read_text())
        self.assertEqual(caller["jobs"]["capi"]["uses"], "./" + release.WORKFLOW)
        self.assertEqual(set(caller["jobs"]["capi"]["needs"]), {"validate-version", "verify"})
        self.assertIn("capi", caller["jobs"]["publish"]["needs"])
        sign_steps = "\n".join(s.get("run", "") for s in jobs["sign"]["steps"])
        self.assertIn("capi_release.py assemble", sign_steps)
        self.assertIn("capi_release.py sign", sign_steps)
        publish_steps = "\n".join(s.get("run", "") for s in jobs["publish"]["steps"])
        self.assertNotIn("cargo build", publish_steps)
        self.assertIn("capi_release.py publish", publish_steps)


if __name__ == "__main__":
    unittest.main()
