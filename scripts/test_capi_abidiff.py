#!/usr/bin/env python3
"""Tests for the C ABI gate's pure parts: prototype extraction, the diff, the
baseline choice and the ABI-version verdict. The first version of the gate
compared stripped libraries with abidiff and reported "0 Changed" across the
pointer-to-integer handle change; these cases make that blindness a test
failure rather than a review finding.
"""

import sys
import contextlib
import io
import tempfile
import zipfile
from unittest.mock import patch
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import capi_abidiff  # noqa: E402
import capi_exports  # noqa: E402

OLD_HEADER = """
/* comment with vanedb_rs_fake(int) inside */
#define VANEDB_RS_VERSION "0.1.1"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct vanedb_rs_store vanedb_rs_store;
typedef bool (*vanedb_rs_filter_fn)(uint64_t id, void *user_data);
vanedb_rs_store *vanedb_rs_store_new(size_t dim, uint32_t metric);
int32_t vanedb_rs_store_add(vanedb_rs_store *s, uint64_t id, const float *v);
size_t vanedb_rs_store_search(const vanedb_rs_store *s,
                              const float *q,
                              size_t k,
                              uint64_t *out_ids,
                              float *out_dists);
const char *vanedb_rs_version(void);
void vanedb_rs_store_free(vanedb_rs_store *s);
#ifdef __cplusplus
}
#endif
"""

NEW_HEADER = """
#define VANEDB_RS_VERSION "0.2.0"
#define VANEDB_RS_ABI_VERSION 1
typedef uint64_t vanedb_rs_store;
typedef bool (*vanedb_rs_filter_fn)(uint64_t id, void *user_data);
vanedb_rs_store vanedb_rs_store_new(size_t dimension, uint32_t metric);
int32_t vanedb_rs_store_add(vanedb_rs_store s, uint64_t id, const float *vector);
size_t vanedb_rs_store_search(vanedb_rs_store s, const float *q, size_t k,
                              uint64_t *out_ids, float *out_dists);
const char *vanedb_rs_version(void);
void vanedb_rs_store_free(vanedb_rs_store s);
uint32_t vanedb_rs_abi_version(void);
"""


class Prototypes(unittest.TestCase):
    def test_normalises_types_and_drops_parameter_names(self):
        protos = capi_exports.prototypes(OLD_HEADER)
        self.assertEqual(sorted(protos), ["vanedb_rs_store_add", "vanedb_rs_store_free",
                                          "vanedb_rs_store_new", "vanedb_rs_store_search",
                                          "vanedb_rs_version"])
        self.assertEqual(protos["vanedb_rs_store_new"], "struct vanedb_rs_store *vanedb_rs_store_new(size_t, uint32_t)")
        self.assertEqual(protos["vanedb_rs_store_search"],
                         "size_t vanedb_rs_store_search(const struct vanedb_rs_store *, const float *, size_t, uint64_t *, float *)")
        self.assertEqual(protos["vanedb_rs_version"], "const char *vanedb_rs_version(void)")

    def test_skips_typedefs_comments_and_preprocessor_lines(self):
        protos = capi_exports.prototypes(NEW_HEADER)
        self.assertNotIn("vanedb_rs_filter_fn", protos)
        self.assertNotIn("vanedb_rs_fake", protos)
        self.assertIn("vanedb_rs_abi_version", protos)

    def test_unsupported_public_syntax_is_rejected(self):
        header = capi_exports.HEADER.read_text(encoding="utf-8")
        for declaration in (
            "typedef struct { uint32_t field; } vanedb_rs_record;",
            "typedef uint32_t vanedb_rs_array[4];",
            "typedef missing_type vanedb_rs_unknown;",
            "typedef vanedb_rs_b vanedb_rs_a; typedef vanedb_rs_a vanedb_rs_b;",
            "uint32_t vanedb_rs_array_arg(uint32_t values[4]);",
            "uint32_t vanedb_rs_inline_callback(bool (*callback)(uint64_t));",
            "uint32_t vanedb_rs_variadic(uint32_t count, ...);",
            "uint32_t vanedb_rs_unspecified();",
            "uint32_t vanedb_rs_trailing_qualifier(uint32_t const);",
            "typedef uint32_t * vanedb_rs_pointer; void vanedb_rs_pointer_arg(const vanedb_rs_pointer p);",
        ):
            with self.subTest(declaration=declaration):
                with self.assertRaisesRegex(SystemExit, "unsupported"):
                    capi_exports.prototypes(header + "\n" + declaration)

    def test_named_and_unnamed_multiword_types_preserve_the_type(self):
        self.assertEqual(
            capi_exports.prototypes("unsigned long vanedb_rs_probe(unsigned int count, unsigned int);")["vanedb_rs_probe"],
            "unsigned long vanedb_rs_probe(unsigned int, unsigned int)",
        )

    def test_abi_version(self):
        self.assertIsNone(capi_exports.abi_version(OLD_HEADER))
        self.assertEqual(capi_exports.abi_version(NEW_HEADER), 1)

    def test_the_committed_list_matches_the_generated_header(self):
        header = capi_exports.HEADER.read_text(encoding="utf-8")
        expected = "".join(f"{p}\n" for _, p in sorted(capi_exports.prototypes(header).items()))
        committed = (capi_exports.EXPORTS / "vanedb_capi.sigs").read_text(encoding="utf-8")
        self.assertEqual(committed, expected, "run scripts/capi_exports.py generate")


class Diff(unittest.TestCase):
    def test_a_retyped_parameter_is_a_change_a_renamed_one_is_not(self):
        diff = capi_abidiff.compare_prototypes(OLD_HEADER, NEW_HEADER)
        changed = {new.split("(")[0].split()[-1].lstrip("*") for _, new in diff["changed"]}
        # every handle-taking or handle-returning function changed...
        self.assertEqual(changed, {"vanedb_rs_store_new", "vanedb_rs_store_add",
                                   "vanedb_rs_store_search", "vanedb_rs_store_free"})
        # ...while vanedb_rs_version, whose only difference is nothing, did not,
        # and a renamed parameter (dim -> dimension) is not a change by itself.
        self.assertEqual(diff["removed"], [])
        self.assertEqual(diff["added"], ["uint32_t vanedb_rs_abi_version(void)"])

    def test_a_removed_function_is_reported(self):
        without = NEW_HEADER.replace("void vanedb_rs_store_free(vanedb_rs_store s);\n", "")
        diff = capi_abidiff.compare_prototypes(NEW_HEADER, without)
        self.assertEqual(diff["removed"], ["void vanedb_rs_store_free(uint64_t)"])

    def test_identical_headers_diff_empty(self):
        diff = capi_abidiff.compare_prototypes(NEW_HEADER, NEW_HEADER)
        self.assertEqual(diff, {"removed": [], "changed": [], "added": []})


class Verdict(unittest.TestCase):
    CLEAN = {"removed": [], "changed": [], "added": []}
    CHANGED = {"removed": [], "changed": [("a", "b")], "added": []}

    def test_same_abi_version_must_be_compatible(self):
        self.assertEqual(capi_abidiff.verdict(1, 1, self.CLEAN, 0)[0], 0)
        self.assertEqual(capi_abidiff.verdict(1, 1, self.CHANGED, 0)[0], 1)
        self.assertEqual(capi_abidiff.verdict(1, 1, self.CLEAN, 4)[0], 1)
        self.assertEqual(capi_abidiff.verdict(1, 1, {"removed": ["x"], "changed": [], "added": []}, 0)[0], 1)

    def test_a_different_abi_version_is_an_intentional_break(self):
        code, message = capi_abidiff.verdict(0, 1, self.CHANGED, 4)
        self.assertEqual(code, 0)
        self.assertIn("intentional ABI break", message)

    def test_tool_errors_and_signals_fail_even_across_a_version_bump(self):
        # 1/2 are libabigail error/usage bits, independently of ABI bits 4/8.
        for status in (1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, -9, -15, 16, 255):
            for current in (0, 1):
                with self.subTest(status=status, current=current):
                    code, message = capi_abidiff.verdict(0, current, self.CLEAN, status)
                    self.assertEqual(code, 1)
                    self.assertIn("abidiff failed", message)

    def test_only_comparison_results_can_be_accepted_at_a_version_bump(self):
        for status in (0, 4, 8, 12):
            with self.subTest(status=status):
                self.assertEqual(capi_abidiff.verdict(0, 1, self.CHANGED, status)[0], 0)

    def test_abi_version_downgrade_fails_even_with_clean_comparisons(self):
        for status in (0, 4, None):
            with self.subTest(status=status):
                code, message = capi_abidiff.verdict(2, 1, self.CLEAN, status)
                self.assertEqual(code, 1)
                self.assertIn("must not decrease", message)

    def test_abidiff_not_run_is_not_a_failure_by_itself(self):
        self.assertEqual(capi_abidiff.verdict(1, 1, self.CLEAN, None)[0], 0)


class GeneratedHeaderGate(unittest.TestCase):
    """Use real generated declarations and the archive/main verdict path.

    ELF contents are placeholders: --skip-abidiff isolates the header layer
    that must catch these changes because stripped binaries cannot.
    """

    def run_gate(self, baseline_text, current_text):
        with tempfile.TemporaryDirectory(prefix="capi-gate-test-") as directory:
            root = Path(directory)
            archive = root / "baseline.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("include/vanedb_rs_capi.h", baseline_text)
                output.writestr("lib/libvanedb_capi.so", b"baseline placeholder")
            current = root / "libvanedb_capi.so"
            current.write_bytes(b"current placeholder")
            header = root / "current.h"
            header.write_text(current_text)
            with patch.object(capi_abidiff, "CURRENT_HEADER", header), patch.object(
                sys, "argv", ["capi_abidiff.py", str(current), "--baseline", str(archive), "--skip-abidiff"]
            ), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                return capi_abidiff.main()

    def test_underlying_alias_and_callback_changes_fail_under_same_abi(self):
        header = capi_exports.HEADER.read_text(encoding="utf-8")
        for old, new in (
            ("typedef uint64_t vanedb_rs_handle;", "typedef uint32_t vanedb_rs_handle;"),
            ("typedef bool (*vanedb_rs_filter_fn)(uint64_t id, void *user_data);",
             "typedef uint32_t (*vanedb_rs_filter_fn)(uint64_t id, void *user_data);"),
            ("typedef bool (*vanedb_rs_filter_fn)(uint64_t id, void *user_data);",
             "typedef bool (*vanedb_rs_filter_fn)(uint32_t id, void *user_data);"),
            ("typedef bool (*vanedb_rs_filter_fn)(uint64_t id, void *user_data);",
             "typedef bool (*vanedb_rs_filter_fn)(uint64_t id, uint32_t user_data);"),
        ):
            with self.subTest(new=new):
                self.assertIn(old, header)
                changed = header.replace(old, new)
                self.assertEqual(self.run_gate(header, changed), 1)
                self.assertTrue(capi_abidiff.compare_prototypes(header, changed)["changed"])

    def test_transitive_alias_resolution_reaches_function_arguments_and_returns(self):
        header = capi_exports.HEADER.read_text(encoding="utf-8")
        prototypes = capi_exports.prototypes(header)
        self.assertTrue(prototypes["vanedb_rs_store_new"].startswith("uint64_t "))
        self.assertEqual(prototypes["vanedb_rs_store_free"], "void vanedb_rs_store_free(uint64_t)")
        self.assertIn("bool (*)(uint64_t, void *)", prototypes["vanedb_rs_store_search_filtered"])

    def test_parameter_renames_in_functions_and_callbacks_stay_compatible(self):
        header = capi_exports.HEADER.read_text(encoding="utf-8")
        renamed = header.replace("uint64_t id", "uint64_t renamed_id").replace("void *user_data", "void *context")
        self.assertNotEqual(header, renamed)
        self.assertEqual(self.run_gate(header, renamed), 0)
        self.assertEqual(capi_exports.prototypes(header), capi_exports.prototypes(renamed))

    def test_equivalent_typedef_chain_stays_compatible(self):
        header = capi_exports.HEADER.read_text(encoding="utf-8")
        renamed = header.replace("typedef uint64_t vanedb_rs_handle;",
                                 "typedef uint64_t vanedb_rs_id; typedef vanedb_rs_id vanedb_rs_handle;")
        self.assertEqual(self.run_gate(header, renamed), 0)


class Baseline(unittest.TestCase):
    RELEASES = [
        {"tagName": "vanedb-crate-v0.1.1"},
        {"tagName": "vanedb-v0.1.1"},
        {"tagName": "vanedb-wasm-v0.1.1"},
        {"tagName": "vanedb-v0.1.0-rc.2"},
        {"tagName": "vanedb-crate-v0.2.0", "isDraft": True},
        {"tagName": "vanedb-crate-v0.3.0"},
    ]

    def test_candidates_are_at_most_current_newest_first_crate_tag_before_python_tag(self):
        current = capi_abidiff.parse_version("0.2.0")
        self.assertEqual(capi_abidiff.baseline_candidates(current, self.RELEASES),
                         ["vanedb-crate-v0.1.1", "vanedb-v0.1.1", "vanedb-v0.1.0-rc.2"])

    def test_a_release_compares_against_itself_after_the_bump(self):
        current = capi_abidiff.parse_version("0.3.0")
        self.assertEqual(capi_abidiff.baseline_candidates(current, self.RELEASES)[0], "vanedb-crate-v0.3.0")

    def test_the_first_candidate_listing_the_asset_wins(self):
        current = capi_abidiff.parse_version("0.2.0")
        assets = {"vanedb-crate-v0.1.1": ["vanedb-capi-0.1.1-macos-aarch64.zip"],
                  "vanedb-v0.1.1": ["vanedb-capi-0.1.1-linux-x86_64.zip", "x.sha256"]}
        self.assertEqual(capi_abidiff.select_baseline(current, self.RELEASES, lambda t: assets.get(t, [])),
                         ("vanedb-v0.1.1", "vanedb-capi-0.1.1-linux-x86_64.zip"))
        self.assertIsNone(capi_abidiff.select_baseline(current, self.RELEASES, lambda t: []))

    def test_newest_prerelease_baseline_uses_numeric_identifier_order(self):
        published = [{"tagName": f"vanedb-crate-v0.2.0-rc.{n}"} for n in (1, 2, 9, 10)]
        def assets(tag):
            return [f"vanedb-capi-{tag.removeprefix('vanedb-crate-v')}-linux-x86_64.zip"]
        for current, expected in (("0.2.0", "0.2.0-rc.10"), ("0.2.0-rc.2", "0.2.0-rc.2")):
            with self.subTest(current=current):
                tag = f"vanedb-crate-v{expected}"
                self.assertEqual(
                    capi_abidiff.select_baseline(capi_abidiff.parse_version(current), published, assets),
                    (tag, assets(tag)[0]),
                )
        self.assertEqual(
            capi_abidiff.baseline_candidates(capi_abidiff.parse_version("0.2.0-rc.2"), published),
            ["vanedb-crate-v0.2.0-rc.2", "vanedb-crate-v0.2.0-rc.1"],
        )

    def test_semver_prerelease_precedence(self):
        ordered = ["0.2.0-alpha", "0.2.0-alpha.1", "0.2.0-alpha.beta", "0.2.0-beta",
                   "0.2.0-beta.2", "0.2.0-beta.11", "0.2.0-rc.1", "0.2.0"]
        versions = [capi_abidiff.parse_version(version) for version in ordered]
        for before, after in zip(versions, versions[1:]):
            self.assertLess(before, after)
        self.assertEqual(capi_abidiff.parse_version("0.2.0-rc.1+build.01"), versions[-2])
        self.assertEqual(capi_abidiff.parse_version("0.2.0+build.99"), versions[-1])

    def test_invalid_semver_is_not_a_baseline(self):
        for version in ("01.2.0", "0.02.0", "0.2.00", "0.2.0-rc.01", "0.2.0-",
                        "0.2.0-rc..1", "0.2.0+", "0.2.0+build..1"):
            with self.subTest(version=version):
                self.assertIsNone(capi_abidiff.parse_version(version))
                self.assertEqual(capi_abidiff.baseline_candidates(
                    capi_abidiff.parse_version("1.0.0"), [{"tagName": f"vanedb-crate-v{version}"}]), [])

    def test_prerelease_sorts_below_its_release(self):
        self.assertLess(capi_abidiff.parse_version("0.2.0-rc.1"), capi_abidiff.parse_version("0.2.0"))


if __name__ == "__main__":
    unittest.main()
