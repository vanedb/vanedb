# Official demo release evidence for RFC 0003

[Obsidian-vane-search PR #20](https://github.com/vanedb/obsidian-vane-search/pull/20)
merged as `5f98bfec` after the provider, centroid, endpoint and tooling fixes
in #23/#21/#22/#24. It supersedes the historical patch in this directory.
Version files and release metadata agree, CI runs the full suite before
publication, and the walkthrough commands are corrected.

The candidate passes 140 automated tests, typecheck, release metadata checks,
production build and size checks; all ten independent reviews and four PR CI
checks passed. The updated real Obsidian/Ollama walkthrough also passed,
including failed rebuild recovery, full restart, successful retry, edits and
deletion. Its rebuilt merged bundle matches the accepted `main.js` SHA-256
`db6f18a706e3650470adcd4961dc1ba62d7011775bd4e60e4438f1b1dfd78775`.
The demo remains pinned to published `@vanedb/wasm` 0.1.1 and discloses that
fact; demo version 0.2.0 does not imply engine 0.2.0 integration. No official
tag or release has been published. [Merged-main CI](https://github.com/vanedb/obsidian-vane-search/actions/runs/36064325791)
and [CodeQL](https://github.com/vanedb/obsidian-vane-search/actions/runs/36064325382)
passed on `5f98bfec`. The reviewed tag-helper dry run passed in an independent
clean clone, repeating 140 tests and reproducing the same bundle checksum.
Publication approval remains pending.

## Required sequence

1. Preserve the completed independent review, PR CI and merged-main CI evidence for `5f98bfec`, including the real test job. A later commit requires renewed checks.
2. Follow the candidate README in a separate test vault using real Obsidian and an embedding model. Record versions, source commit, build checksum, note/query, observed results, note opening, restart, update and deletion checks in the demo release checklist. Do not modify a personal vault for release testing.
   - Desktop acceptance and a synthetic-vault screenshot are recorded at
     merged demo commit `5f98bfec` in
     `docs/releases/0.2.0-desktop-acceptance.md`. This accepts the recorded
     bundle only; later bundle changes require
     renewed validation before release.
3. PR #20 is merged and its rebuilt bundle matches the acceptance record. Merged-main CI and the tag-helper dry run passed. Create an annotated `0.2.0` tag on the reviewed merged commit and push that tag explicitly only when publication is approved. Use `bash docs/launch/maintainer_closeout_226.sh --tag --confirm-vault-walkthrough`
   or Actions → **Tag demo 0.2.0** with `DEMO_REPO_TOKEN` after those gates.
   The demo README contains the exact commands; pushing the tag publishes the release.
4. Verify the official release contains `main.js`, `manifest.json`, and `LICENSE`, and that its README includes the observed walkthrough.
5. Record the official release URL and walkthrough evidence on [vanedb#242](https://github.com/vanedb/vanedb/issues/242).

The historical `maintainer_cut_demo_0.2.0.sh`, `maintainer_closeout_226.sh --cut` and **Cut demo 0.2.0** workflow apply the older patch and publish a tag. They are not the reviewed-PR route above; do not run them for this candidate. A staging artifact on the engine repository is not the official demo release.

If the engine dependency is upgraded before release, use the demo checklist's disposable candidate-package test first, then pin the actually published package and repeat validation and the real walkthrough. Do not commit an unpublished dependency or local tarball path.

The walkthrough evidence is recorded. RFC 0003 demo acceptance remains open until the official release and its assets are verified and linked.
