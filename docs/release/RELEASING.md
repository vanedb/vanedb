# Preparing and publishing VaneDB

The [readiness record](0.1.1-readiness.md) is the release checklist. A version
bump prepares artifacts; it does not satisfy the remaining requirements or
authorize publication. [Draft notes](0.1.1-notes.md) describe the candidate.

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
   candidate_branch=review/0.1.1-npm-signoff # branch containing the reviewed commit
   gh workflow run publish-crate.yml --ref "$candidate_branch"
   gh workflow run publish-rust.yml --ref "$candidate_branch" -f publish_testpypi=false
   gh workflow run publish-wasm.yml --ref "$candidate_branch"
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
   re-uploaded: if `0.1.1` goes out wrong, that number is burned and the fix
   ships as `0.1.2`. `cargo yank` and a PyPI yank stop new resolution but
   delete nothing and do not break existing lockfiles; deleting a PyPI file is
   permanent and does not free the filename. The Python job uploads 29 files in
   one action, so a partial upload leaves that version permanently half-populated.
   This is a property of the registries, not a project policy.

5. **Before designating the final candidate**, put the published-install form
   into `vanedb/README.md`, `vanedb-py/README.md` and `vanedb-wasm/README.md`.
   `vanedb/Cargo.toml`
   declares `readme = "README.md"`, so that file is inside the `.crate` and is
   what crates.io renders; `vanedb-py/pyproject.toml` does the same, making its
   README the core-metadata description of all 28 wheels and the sdist, and the
   PyPI project page. Neither registry permits a re-upload, so a checkout-only
   install line in either file is permanent for that version: for the whole
   time that version is the newest release, both registry pages would tell a visitor
   nothing is published while they are standing on the published package.

   Align git-only guides in the release PR as well. Keep the readiness record
   explicit about verification and publication status.

6. Record the candidate commit, run URLs, artifact checksums, resolved findings
   and final role verdicts. Finalize release notes and compatibility language.

## Registry setup

The Python publisher is repository `vanedb/vanedb`, workflow `publish-rust.yml`,
with environment `pypi` (or `testpypi` for an explicitly requested rehearsal).
PyPI supports a [pending publisher for a new project](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).
Restrict the publisher to the `pypi` environment and verify the saved
configuration; do not create a duplicate. PyPI has no edit — remove the loose
entry before adding the strict one, because it accepts a token if *any*
configured publisher matches, so leaving both changes nothing. The GitHub
environment's existence alone does not prove that PyPI trusts it.

The Rust publisher uses `publish-crate.yml` and environment `crates-io`.
The [crates.io prerequisites](https://crates.io/docs/trusted-publishing)
require an initial manual publication before configuring trusted publishing.
An owner-approved, verified email is required before publication is possible
at all. Do not create a duplicate account or assume sign-in alone enables
publishing.
If bootstrap publication is needed, use the approved version and commit with a
scoped token through Cargo's credential mechanism; never put credentials in
release notes, shell history or logs. Configure the trusted publisher
afterward, then **revoke the bootstrap token** — it is a long-lived credential
whose only purpose was the one publish that trusted publishing could not do,
and leaving it live defeats the reason for using OIDC everywhere else.

Do not also push `vanedb-crate-v<version>` for a version published by
bootstrap: the tagged workflow would attempt the same version and fail against
a registry that never allows a re-upload.

Run `python3 scripts/check_release_readmes.py vanedb-crate-v<version>` by hand
before a bootstrap publish. The tagged workflows run it for you; a bootstrap
does not go through one, and a hand publish outside every workflow is exactly
the act that put "Nothing is published yet" on the npm page.

**Bootstrap a prerelease, not the release.** The version you bootstrap is spent
by hand — no reviewer gate, no OIDC, no provenance — so it should not be the
version people install. `0.1.0-rc.1` reserved the name, and `0.1.1` goes
out through the tagged workflow like every release after it. Prereleases are
excluded from Cargo's default requirements, so `vanedb = "0.1"` never resolves
to the bootstrap. `cargo add vanedb` can select an available prerelease and
write its version requirement. Leave the bootstrap version published rather than yanking it: it
is the provenance record of how the name was claimed, and yanking it would
imply it was defective.

npm is the counter-example already on the shelf. `@vanedb/wasm@0.1.0` was
bootstrapped by hand, so it carries no attestation, and npm never frees a used
version. The first release is therefore `0.1.1` across all three registries,
from one approved source commit.

## Publish the approved release

Publication requires explicit maintainer authorization after verification.
Merge the approved candidate through the protected-main PR process. All three
tag workflows reject release commits that are not on main.

The three registries can be published in any order. `vanedb-py` depends on the
core by path (`vanedb = { path = "../vanedb" }`) and maturin vendors that source
into the sdist — `vanedb-<version>/vanedb/src/lib.rs` is inside the tarball — so
installing from source never resolves `vanedb` from crates.io. Verify with
`maturin sdist -m vanedb-py/Cargo.toml` and list the archive if this ever
changes to a version dependency, which would make crates.io a hard prerequisite.

Every tag below is written `v<version>`; substitute the version being
released. All three publish workflows first run
`scripts/check_release_readmes.py`, which refuses a tag whose packaged README
still describes the checkout — that file is inside the artifact and is what the
registry renders, and no registry here permits a re-upload. It matches prose,
so it is a tripwire rather than a proof: it catches the pre-publication README
being left in place, which is the mistake that has actually happened, and it
cannot certify that a rewritten one is honest. Run it locally before pushing a
tag, so a false positive costs a minute rather than a deleted tag and a PR
through protected main.

- Python uses tag `vanedb-v<version>`; its workflow builds and validates the
  wheels and source distribution before the protected `pypi` publication job.
- The crate uses tag `vanedb-crate-v<version>`, once its trusted publisher is
  configured. The workflow validates the package before the protected
  `crates-io` job. Never point this tag at a version already published by
  bootstrap.
- WebAssembly uses tag `vanedb-wasm-v<version>` — the tag names the crate; the
  published package is **`@vanedb/wasm`**. `scripts/build_npm_package.py`
  merges the two wasm-pack targets into that one package with conditional
  `exports`, and `scripts/check_npm_package.py` installs the packed
  tarball into a throwaway project and imports it through the public specifier
  in both ESM and CommonJS before the protected `npm` publication job. npm uses
  OIDC trusted publishing with `--provenance`, so no long-lived token is stored
  and the published package carries an attestation naming this workflow and
  commit. A trusted publisher must be configured on npmjs.com for
  `@vanedb/wasm` first, which requires owning the `vanedb` npm organisation or
  user scope.

  The verifier packs and tests one tarball. The protected publisher downloads
  that tarball, verifies its checksum, reruns the installed consumer and
  publishes the same bytes. It does not rebuild the package.

  **npm needs a bootstrap publish, exactly like crates.io.** Trusted
  publishing is configured per package, at
  `npmjs.com → Packages → @vanedb/wasm → Settings → Trusted publishing`, and
  that page cannot exist until the package does. npm has no equivalent of
  PyPI's pending publisher. So the first version is published manually with a
  granular access token, the trusted publisher is configured afterwards, and
  the token is revoked. As with crates.io, the bootstrap version carries no
  provenance attestation; every later version does.

  Do not also push `vanedb-wasm-v<version>` for a bootstrapped version — the
  tagged workflow would attempt the same version against a registry that
  rejects re-uploads.

  Scoped packages default to private, which is why the publish step passes
  `--access public`.
- C library archives use the same `vanedb-crate-v<version>` tag through
  `publish-capi.yml`. It builds and tests all five native distributions,
  creates target-specific CycloneDX SBOMs and signs every release payload
  before the C asset job, protected by the existing `crates-io` environment,
  attaches the retained bytes. See
  the signed C distribution procedure below. Do not manually replace those
  assets. Raw `vanedb-wasm-<version>-nodejs.tgz` and
  `vanedb-wasm-<version>-web.tgz` assets, when attached, retain their matching
  checksums. There is no automatic C++ package publication.

After publication, verify registry version metadata and install the published
packages in clean environments. Run the documented quickstarts and check that
the downloadable C and WebAssembly assets match the approved checksums. Update
git-only guides and the organisation profile with verified release links and
publication status. Packaged READMEs must already be correct before tagging;
they cannot be changed inside an existing release. See step 5.

If publication fails, inspect which versions and files were actually accepted
before retrying. Preserve published versions and tags; do not delete or move
them to conceal a partial release. Record the failure and choose the next action
with the maintainer.


## Signed C distribution (RFC 0002 stage 5)

C asset publication reuses the existing `crates-io` GitHub environment, the
same approval boundary as the crate published from `vanedb-crate-v<version>`.
A read-only check on 2026-09-22 confirmed required reviewer `tsvet01` and a
custom **tag** policy matching `vanedb-crate-v*`. Both publication jobs must
pass this environment's approval; the C job does not exchange its token for a
crates.io credential. Re-check the environment's reviewer and tag policy
before releasing; naming an environment alone does not protect it. Do not
introduce a new environment name without first configuring and verifying its
protection: GitHub otherwise creates it without required reviewers. No new
registry credentials or long-lived signing key is used.
Only the signing job receives `id-token: write`; only the tag publication job
receives `contents: write`. Native build jobs have read-only repository access.

Rehearse **from the candidate branch** with
`gh workflow run publish-crate.yml --ref "$candidate_branch"`. Its reusable
`publish-capi.yml` runs from that same commit, so this also works before the
new C workflow exists on the default branch. Dispatches naming
a tag are rejected, and the upload job additionally requires a tag **push**.
A rehearsal generates real keyless signatures with its branch identity and
retains the verified set in the `capi-signed-release` workflow artifact; it
creates no GitHub Release and publishes no package. Record its run URL and
source commit, download that artifact into an empty directory, and run:

```sh
python3 scripts/capi_release.py verify --directory /path/to/capi-signed-release \
  --ref "refs/heads/$candidate_branch" --version 0.2.0 --commit <approved-full-sha>
```

Use the actual candidate version until the version-bump PR lands. Install the
pinned cosign version listed in `CAPI-VERIFYING.md`. The verifier checks every
signature's exact workflow identity, OIDC issuer and source-SHA certificate
claim, all checksums, versions, source provenance, archive contents and the
complete five-platform inventory.
A branch signature cannot satisfy the tag-identity check after release.

The workflow builds natively on Linux x86-64/ARM64, macOS ARM64/x86-64 and
Windows x64 with the `capi` profile and Rust 1.94.0. Each job runs C ABI tests,
header checks, and `package_capi.py`'s extracted static/shared CMake and
pkg-config acceptance before upload. Cargo CycloneDX 0.5.9 generates a separate
CycloneDX 1.5 JSON file for each target using the same lockfile and default C
ABI features. The lockfile must remain unchanged. SBOMs include build-time
Cargo dependencies, not the OS libraries recorded in `compatibility.json`.
The five native platforms are an explicit release tripwire; future mobile or
additional platform assets require updating the matrix, inventory and tests
together. This release does not claim stages 2–4.

The assembled set contains 13 payloads and 13 Sigstore bundles: five archives,
five SBOMs, `CAPI-RELEASE.json`, `CAPI-VERIFYING.md`, and `SHA256SUMS`.
`SHA256SUMS` lists the other 12 payloads; it is separately signed, as is each
payload. The manifest records the exact source SHA, target compatibility
metadata and archive/SBOM hashes. Signing uses pinned cosign 3.1.3 via an
immutable installer action, and verifies the signatures before retaining them.

After explicit release authorization, a matching crate-tag push must resolve
to the approved commit on main. The protected publication job downloads the
retained signed artifact, verifies it again, and attaches it to that tag's
release without rebuilding. If the release is absent it creates a draft,
uploads the complete set, downloads and verifies the actual remote bytes, then
publishes the draft. Prerelease versions (for example `0.2.0-rc.1`) create
GitHub prereleases; an existing release with a contradictory prerelease flag
fails without changing that flag. If a release already exists its human-written
notes are preserved and the exact verification instructions appended. Existing assets
are accepted only when byte-identical; no `--clobber`, tag deletion or tag
movement is used. The published notes contain copyable verification commands
with the exact tag identity and approved source SHA.

If upload is interrupted, preserve the draft and any uploaded files. Re-run
only the failed publication job so it retrieves the same signed artifact;
regenerating signatures or rebuilding produces a different set and is rejected
when existing remote assets differ. Inspect any disagreement before deciding
how to recover; never overwrite an already published artifact to make a retry
pass. A release or registry publication is not complete until its actual
remote bytes and installed consumers have been verified.
