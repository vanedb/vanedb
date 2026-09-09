# Preparing and publishing VaneDB

The [readiness record](0.1.0-readiness.md) is the release checklist. A version
bump prepares artifacts; it does not satisfy the remaining requirements or
authorize publication. [Draft notes](0.1.0-notes.md) describe the candidate.

## Verify the candidate

1. Refresh main and all contributor PR heads. Integrate changes through the
   release PR, resolve the remaining findings, and identify the approved commit.
2. Confirm the core, Python, C ABI and WebAssembly Cargo package versions and
   `vanedb-py/pyproject.toml` agree. Both Cargo lockfiles must resolve that version.
   The C++ manifest, header and CMake version must match the core, as checked by
   `bench/tests/release_identity.rs`; the benchmark package version is independent.
3. Require the complete platform CI matrix and installed/extracted artifact
   checks for that candidate. Complete the accepted simulator/emulator mobile,
   Metal, performance, persistence-contract and role-review requirements in the
   readiness record. CUDA is required after the initial release; physical-device
   checks are a follow-up. Neither may be advertised as verified in this release.
   Check that Rust unsafe disk-open call sites establish file immutability from
   before opening through the mapping's lifetime, and that Python/C guidance
   explains the same obligation. Reconcile the draft notes with the final
   per-query search API, result/lookup types and stricter input validation.
4. Dispatch publication-disabled rehearsals **from a branch, never a tag**:

   ```sh
   gh workflow run publish-crate.yml --ref main
   gh workflow run publish-rust.yml --ref main -f publish_testpypi=false
   ```

   The publish jobs test `github.event_name == 'push'`, so a dispatch cannot
   satisfy them whatever ref it names. That was not always true: they
   previously tested `ref_type == 'tag'`, which a `--ref <tag>` dispatch does
   satisfy, so this step could publish. Keep the `event_name` form.

   Require successful crate, wheel and source-distribution verification, with
   all publication jobs skipped. Download the artifacts; check their versions,
   file lists and checksums. The crate verifier tests the extracted archive in
   its own build directory. Inspect the actual `.crate`, not only the listing.
   **Publication is irreversible.** Neither registry lets a version be
   re-uploaded: if `0.1.0` goes out wrong, that number is burned and the fix
   ships as `0.1.1`. `cargo yank` and a PyPI yank stop new resolution but
   delete nothing and do not break existing lockfiles; deleting a PyPI file is
   permanent and does not free the filename. The Python job uploads 29 files in
   one action, so a partial upload leaves `0.1.0` permanently half-populated.
   This is a property of the registries, not a project policy.

5. **Before designating the final candidate**, put the published-install form
   into `vanedb/README.md` and `vanedb-py/README.md`. `vanedb/Cargo.toml`
   declares `readme = "README.md"`, so that file is inside the `.crate` and is
   what crates.io renders; `vanedb-py/pyproject.toml` does the same, making its
   README the core-metadata description of all 28 wheels and the sdist, and the
   PyPI project page. Neither registry permits a re-upload, so a checkout-only
   install line in either file is permanent for that version: for the whole
   time 0.1.0 is the newest release, both registry pages would tell a visitor
   nothing is published while they are standing on the published package.

   Every other guide is git-only and is corrected after publication (#122).
   Make this edit in the release commit itself, not earlier — until the tag,
   the checkout instructions are the true ones.

6. Record the candidate commit, run URLs, artifact checksums, resolved findings
   and final role verdicts. Finalize release notes and compatibility language.

## Registry setup

The Python publisher is repository `vanedb/vanedb`, workflow `publish-rust.yml`,
with environment `pypi` (or `testpypi` for an explicitly requested rehearsal).
PyPI supports a [pending publisher for a new project](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).
On September 7, 2026, signed-in inspection confirmed an existing pending
publisher for project `vanedb`, repository `vanedb/vanedb` and workflow
`publish-rust.yml`. It currently permits any GitHub environment. Restrict it
to `pypi` and verify the saved configuration before publication; do not create
a duplicate publisher. The GitHub environment's existence alone does not prove
that PyPI trusts it.

The Rust publisher uses `publish-crate.yml` and environment `crates-io`.
The [crates.io prerequisites](https://crates.io/docs/trusted-publishing)
require an initial manual publication before configuring trusted publishing.
On September 7, inspection confirmed the owner signed in as `tsvet01`, with
no email address configured. Add an owner-approved email and complete its
verification before arranging initial publication and the trusted publisher.
Do not create a duplicate account or assume sign-in alone enables publishing.
If bootstrap publication is needed, use the approved version and commit with a
scoped token through Cargo's credential mechanism; never put credentials in
release notes, shell history or logs. Configure the trusted publisher
afterward, then **revoke the bootstrap token** — it is a long-lived credential
whose only purpose was the one publish that trusted publishing could not do,
and leaving it live defeats the reason for using OIDC everywhere else.

Do not also push `vanedb-crate-v<version>` for a version published by
bootstrap: the tagged workflow would attempt the same version and fail against
a registry that never allows a re-upload.

## Publish the approved release

Publication requires explicit maintainer authorization after verification.
Merge the approved candidate through the protected-main PR process. Both tag
workflows reject release commits that are not on main.

The two registries can be published in either order. `vanedb-py` depends on the
core by path (`vanedb = { path = "../vanedb" }`) and maturin vendors that source
into the sdist — `vanedb-<version>/vanedb/src/lib.rs` is inside the tarball — so
installing from source never resolves `vanedb` from crates.io. Verify with
`maturin sdist -m vanedb-py/Cargo.toml` and list the archive if this ever
changes to a version dependency, which would make crates.io a hard prerequisite.

- Python uses tag `vanedb-v0.1.0`; its workflow builds and validates the wheels
  and source distribution before the protected `pypi` publication job.
- If the crate's trusted publisher is already configured, use tag
  `vanedb-crate-v0.1.0`. The crate workflow validates the package before the
  protected `crates-io` job. If 0.1.0 is the manual bootstrap publication,
  do not also trigger this tag's upload of the same version; use the crate-tag
  workflow for later versions after configuring its publisher.
- WebAssembly uses tag `vanedb-wasm-v0.1.0` — the tag names the crate; the
  published package is **`@vanedb/wasm`**. `scripts/build_npm_package.py`
  merges the two wasm-pack targets into that one package with conditional
  `exports`, and `scripts/check_npm_package.py` installs the packed
  tarball into a throwaway project and imports it through the public specifier
  in both ESM and CommonJS before the protected `npm` publication job. npm uses
  OIDC trusted publishing with `--provenance`, so no long-lived token is stored
  and the published package carries an attestation naming this workflow and
  commit. A trusted publisher must be configured on npmjs.com for
  `@vanedb/wasm` first, which requires owning the `vanedb` npm organisation or
  user scope. Unlike crates.io, npm needs no bootstrap publish. Scoped
  packages default to private, which is why the publish step passes
  `--access public`.
- C library archives come from the
  approved commit's CI artifacts. Attach the C archives with their runtime
  compatibility metadata, and `vanedb-wasm-0.1.0-nodejs.tgz` and
  `vanedb-wasm-0.1.0-web.tgz`, each with its matching checksum. There is no
  automatic C++ package publication.

After publication, verify registry version metadata and install the published
packages in clean environments. Run the documented quickstarts and check that
the downloadable C and WebAssembly assets match the approved checksums. Only
then add the verified registry commands and release links to the **git-only**
user guides: the root `README.md`, `cpp/`, and the organisation profile. That
step cannot reach `vanedb/README.md` or `vanedb-py/README.md` — those are
already inside the published artifacts by this point. See step 5.

If publication fails, inspect which versions and files were actually accepted
before retrying. Preserve published versions and tags; do not delete or move
them to conceal a partial release. Record the failure and choose the next action
with the maintainer.
