# Preparing and publishing VaneDB

The [readiness record](1.0.0-readiness.md) is the release checklist. A version
bump prepares artifacts; it does not satisfy the remaining requirements or
authorize publication. [Draft notes](1.0.0-notes.md) describe the candidate.

## Verify the candidate

1. Refresh main and all contributor PR heads. Integrate changes through the
   release PR, resolve the remaining findings, and identify the approved commit.
2. Confirm the core, Python, C ABI and WebAssembly Cargo package versions and
   `vanedb-py/pyproject.toml` agree. Both Cargo lockfiles must resolve that version.
   The frozen C++ reference and benchmark harness have independent versions.
3. Require the complete platform CI matrix and installed/extracted artifact
   checks for that candidate. Complete the physical-device, GPU, performance,
   persistence-contract and role-review requirements in the readiness record.
4. Dispatch publication-disabled rehearsals on the candidate branch:

   ```sh
   gh workflow run publish-crate.yml --ref release/1.0.0-readiness
   gh workflow run publish-rust.yml --ref release/1.0.0-readiness -f publish_testpypi=false
   ```

   Require successful crate, wheel and source-distribution verification, with
   all publication jobs skipped. Download the artifacts; check their versions,
   file lists and checksums. The crate verifier tests the extracted archive in
   its own build directory. Inspect the actual `.crate`, not only the listing.
5. Record the candidate commit, run URLs, artifact checksums, resolved findings
   and final role verdicts. Finalize release notes and compatibility language.

## Registry setup

The Python publisher is repository `vanedb/vanedb`, workflow `publish-rust.yml`,
with environment `pypi` (or `testpypi` for an explicitly requested rehearsal).
PyPI supports a [pending publisher for a new project](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).
The registry owner must verify this configuration; the GitHub environment's
existence does not prove the registry trusts it.

The Rust publisher uses `publish-crate.yml` and environment `crates-io`.
The [crates.io prerequisites](https://crates.io/docs/trusted-publishing)
require an initial manual publication before configuring trusted publishing.
Confirm the current registry setup with the owner before the first release.
If bootstrap publication is needed, use the approved version and commit with a
scoped token through Cargo's credential mechanism; never put credentials in
release notes, shell history or logs. Configure the trusted publisher afterward.

## Publish the approved release

Publication requires explicit maintainer authorization after verification.
Merge the approved candidate through the protected-main PR process. Both tag
workflows reject release commits that are not on main.

- Python uses tag `vanedb-v1.0.0`; its workflow builds and validates the wheels
  and source distribution before the protected `pypi` publication job.
- If the crate's trusted publisher is already configured, use tag
  `vanedb-crate-v1.0.0`. The crate workflow validates the package before the
  protected `crates-io` job. If 1.0.0 is the manual bootstrap publication,
  do not also trigger this tag's upload of the same version; use the crate-tag
  workflow for later versions after configuring its publisher.
- C library archives and the separate Node/browser npm tarballs come from the
  approved commit's CI artifacts. Attach the C archives with their runtime
  compatibility metadata, and `vanedb-wasm-1.0.0-nodejs.tgz` and
  `vanedb-wasm-1.0.0-web.tgz`, each with its matching checksum. There is no
  automatic npm-registry or C++ package publication.

After publication, verify registry version metadata and install the published
packages in clean environments. Run the documented quickstarts and check that
the downloadable C and WebAssembly assets match the approved checksums. Only
then add the verified registry commands and release links to the user guides.

If publication fails, inspect which versions and files were actually accepted
before retrying. Preserve published versions and tags; do not delete or move
them to conceal a partial release. Record the failure and choose the next action
with the maintainer.
