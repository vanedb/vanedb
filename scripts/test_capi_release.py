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

    def publication(self, existing=None, version=VERSION, corrupt=False):
        if not (self.directory / release.NOTES).exists():
            self.assemble()
        self.dummy_bundles()
        payloads = release.names(VERSION) + [release.MANIFEST, release.NOTES, release.SUMS]
        remote = {}
        calls = []
        if existing:
            for asset in existing["assets"]:
                remote[asset["id"]] = (self.directory / asset["name"]).read_bytes()
        def api(method, endpoint, data=None):
            calls.append((method, endpoint, data))
            if method == "POST":
                return {"id": 71, "assets": [], **data}
            if method == "PATCH":
                self.assertEqual(endpoint, f"repos/{release.REPOSITORY}/releases/71")
                self.assertEqual(len(remote), 26)
                return {}
            raise AssertionError((method, endpoint))
        def upload(release_id, path):
            self.assertEqual(release_id, 71)
            asset_id = len(remote) + 100
            remote[asset_id] = path.read_bytes()
            return {"id": asset_id}
        def download(asset_id, path):
            path.write_bytes(b"tampered" if corrupt else remote[asset_id])
        checked = []
        def verify(directory, *args):
            checked.append(directory)
            release.verify_contents(directory, VERSION, COMMIT, REF, signed=True)
            return payloads
        ref = "refs/tags/vanedb-crate-v" + version
        with patch.dict(os.environ, {"GITHUB_EVENT_NAME": "push", "GITHUB_REF": ref,
                                     "GITHUB_REPOSITORY": release.REPOSITORY}), \
                patch.object(release, "source_commit", return_value=COMMIT), \
                patch.object(release, "output", return_value=COMMIT), \
                patch.object(release, "verify", side_effect=verify), \
                patch.object(release, "find_release", return_value=existing), \
                patch.object(release, "release_api", side_effect=api), \
                patch.object(release, "upload_asset", side_effect=upload), \
                patch.object(release, "download_asset", side_effect=download), \
                patch.object(release, "run"):
            release.publish(self.directory, version, COMMIT, ref)
        return calls, checked

    def test_existing_draft_is_reused_and_human_notes_preserved(self):
        calls, checked = self.publication({"id": 71, "draft": True,
            "prerelease": False, "body": "human notes", "assets": []})
        self.assertEqual([c[0] for c in calls], ["PATCH"])
        self.assertTrue(calls[0][2]["body"].startswith("human notes\n\n"))
        self.assertFalse(calls[0][2]["draft"])
        self.assertEqual(len(checked), 2)

    def test_new_release_stays_draft_until_remote_bytes_verified(self):
        calls, checked = self.publication()
        self.assertEqual([c[0] for c in calls], ["POST", "PATCH"])
        self.assertTrue(calls[0][2]["draft"])
        self.assertFalse(calls[0][2]["prerelease"])
        self.assertEqual(calls[0][2]["target_commitish"], COMMIT)
        self.assertEqual(calls[1][2], {"draft": False})
        self.assertEqual(len(checked), 2)

    def test_prerelease_creation_and_stable_mismatch(self):
        calls, _ = self.publication(version="0.2.0-rc.1")
        self.assertTrue(calls[0][2]["prerelease"])
        with self.assertRaisesRegex(ValueError, "prerelease status"):
            self.publication({"id": 71, "draft": False, "prerelease": False,
                              "body": "", "assets": []}, version="0.2.0-rc.1")

    def test_existing_asset_is_never_overwritten(self):
        with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
            self.publication({"id": 71, "draft": False, "prerelease": False,
                "body": "human notes", "assets": [{"name": "CAPI-RELEASE.json", "id": 42}]},
                corrupt=True)

    def test_corrupt_uploaded_bytes_cannot_publish(self):
        with self.assertRaises(ValueError):
            self.publication(corrupt=True)

    def test_published_release_retry_preserves_notes_and_assets(self):
        self.assemble()
        assets = release.names(VERSION) + [release.MANIFEST, release.NOTES, release.SUMS]
        assets += [name + release.BUNDLE for name in assets]
        calls, checked = self.publication({"id": 71, "draft": False, "prerelease": False,
            "body": "human notes\n\n" + (self.directory / release.NOTES).read_text(),
            "assets": [{"id": i, "name": name} for i, name in enumerate(assets)]})
        self.assertEqual(calls, [])
        self.assertEqual(len(checked), 2)

    def test_release_lookup_includes_drafts_on_later_pages(self):
        first = [{"tag_name": f"other-{i}"} for i in range(100)]
        draft = {"id": 71, "tag_name": "target", "draft": True}
        with patch.object(release, "release_api", side_effect=[first, [draft]]) as api:
            self.assertEqual(release.find_release("target"), draft)
        self.assertIn("page=2", api.call_args.args[1])

    def test_duplicate_tag_release_is_ambiguous_even_if_one_is_published(self):
        with patch.object(release, "release_api", return_value=[
            {"tag_name": "target", "draft": True}, {"tag_name": "target", "draft": False}]):
            with self.assertRaisesRegex(ValueError, "multiple releases"):
                release.find_release("target")

    def test_release_lookup_errors_do_not_mean_absence(self):
        for error in ("HTTP 403", "HTTP 404", "network failure"):
            response = release.subprocess.CompletedProcess([], 1, "", error)
            with self.subTest(error=error), patch.object(release.subprocess, "run", return_value=response):
                with self.assertRaisesRegex(ValueError, "cannot inspect"):
                    release.find_release("target")

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
