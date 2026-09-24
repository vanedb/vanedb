# Official demo release evidence for RFC 0003

The current candidate is [obsidian-vane-search PR #20](https://github.com/vanedb/obsidian-vane-search/pull/20), based on official main `e79bef9`. It supersedes the historical patch in this directory. Candidate `ab598838` synchronizes all version files, checks metadata in CI, runs the full suite before publication, and corrects the walkthrough commands.

The candidate passes 107 automated tests, typecheck, production build, size checks and actionlint. Automated tests use real WASM with fake embedding providers and an Obsidian stub: they do not prove the desktop walkthrough or semantic relevance. The demo remains pinned to published `@vanedb/wasm` 0.1.1 and discloses that fact; demo version 0.2.0 does not imply engine 0.2.0 integration.

## Required sequence

1. Complete independent review and required CI on the final demo commit.
2. Follow the candidate README in a separate test vault using real Obsidian and an embedding model. Record versions, source commit, build checksum, note/query, observed results, note opening, restart, update and deletion checks in the demo release checklist. Do not modify a personal vault for release testing.
   - Desktop acceptance and a synthetic-vault screenshot are recorded at
     demo documentation commit `94ebb158` in
     `docs/releases/0.2.0-desktop-acceptance.md`. This accepts the recorded
     bundle only; provider-transition fixes or other bundle changes require
     renewed validation before release.
3. Merge the reviewed release PR. Create an annotated `0.2.0` tag on the reviewed merged commit and push that tag explicitly only when publication is approved. Use `bash docs/launch/maintainer_closeout_226.sh --tag --confirm-vault-walkthrough`
   or Actions → **Tag demo 0.2.0** with `DEMO_REPO_TOKEN` after those gates.
   The demo README contains the exact commands; pushing the tag publishes the release.
4. Verify the official release contains `main.js`, `manifest.json`, and `LICENSE`, and that its README includes the observed walkthrough.
5. Record the official release URL and walkthrough evidence on [vanedb#242](https://github.com/vanedb/vanedb/issues/242).

The historical `maintainer_cut_demo_0.2.0.sh`, `maintainer_closeout_226.sh --cut` and **Cut demo 0.2.0** workflow apply the older patch and publish a tag. They are not the reviewed-PR route above; do not run them for this candidate. A staging artifact on the engine repository is not the official demo release.

If the engine dependency is upgraded before release, use the demo checklist's disposable candidate-package test first, then pin the actually published package and repeat validation and the real walkthrough. Do not commit an unpublished dependency or local tarball path.

Until the official release and walkthrough evidence exist, RFC 0003 demo acceptance remains open.
