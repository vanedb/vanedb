#!/usr/bin/env python3
"""Tests for the C ABI gate's pure parts: prototype extraction, the diff, the
baseline choice and the ABI-version verdict. The first version of the gate
compared stripped libraries with abidiff and reported "0 Changed" across the
pointer-to-integer handle change; these cases make that blindness a test
failure rather than a review finding.
"""

import sys
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
        self.assertEqual(protos["vanedb_rs_store_new"], "vanedb_rs_store *vanedb_rs_store_new(size_t, uint32_t)")
        self.assertEqual(protos["vanedb_rs_store_search"],
                         "size_t vanedb_rs_store_search(const vanedb_rs_store *, const float *, size_t, uint64_t *, float *)")
        self.assertEqual(protos["vanedb_rs_version"], "const char *vanedb_rs_version(void)")

    def test_skips_typedefs_comments_and_preprocessor_lines(self):
        protos = capi_exports.prototypes(NEW_HEADER)
        self.assertNotIn("vanedb_rs_filter_fn", protos)
        self.assertNotIn("vanedb_rs_fake", protos)
        self.assertIn("vanedb_rs_abi_version", protos)

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
        self.assertEqual(diff["removed"], ["void vanedb_rs_store_free(vanedb_rs_store)"])

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

    def test_abidiff_not_run_is_not_a_failure_by_itself(self):
        self.assertEqual(capi_abidiff.verdict(1, 1, self.CLEAN, None)[0], 0)


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

    def test_prerelease_sorts_below_its_release(self):
        self.assertLess(capi_abidiff.parse_version("0.2.0-rc.1"), capi_abidiff.parse_version("0.2.0"))


if __name__ == "__main__":
    unittest.main()
