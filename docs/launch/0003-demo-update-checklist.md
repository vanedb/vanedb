# Demo update patch notes for `obsidian-vane-search` (issue #198)

This agent cannot push to https://github.com/vanedb/obsidian-vane-search
(no write permission). Maintainer request:
https://github.com/vanedb/obsidian-vane-search/issues/18 — apply the ready
patch below, cut a release, and reply on vanedb#226 with the release
URL.

## Verified on tip (agent)

Against `obsidian-vane-search` `main` (`e79bef9`), the patch:

1. `git apply` — clean
2. `npm ci && npm test` — **98/98** vitest passed
3. `npm run build` — produced `main.js` (215.9 KiB)

Staging installables (prerelease on **vanedb**, not the demo repo):
https://github.com/vanedb/vanedb/releases/tag/demo-0.2.0-staging
(`main.js` + `manifest.json` + `LICENSE`, version `0.2.0`). Manual install into
`<vault>/.obsidian/plugins/vane-search/` works for walkthrough verification;
AC5 still needs the **official** `obsidian-vane-search` `0.2.0` tag/release.

## Ready-to-apply patch

[`0003-obsidian-vane-search-0.2.0.patch`](0003-obsidian-vane-search-0.2.0.patch)
bumps `package.json` / `manifest.json` / `versions.json` to `0.2.0` and adds
README section **"Try it on a real vault"** (BRAT → Ollama nomic → index →
one semantic query).

**One-shot** (on **main** via [#219](https://github.com/vanedb/vanedb/pull/219);
needs push access to the demo repo; cloud agents get 403):

```bash
bash docs/launch/maintainer_cut_demo_0.2.0.sh
# optional: --skip-tests  (release workflow still builds + publishes assets)
# or: DEMO_REPO_TOKEN=... bash docs/launch/maintainer_cut_demo_0.2.0.sh --skip-tests
```

**Actions alternative** (`cut-demo-0.2.0.yml` on **main** via
[#220](https://github.com/vanedb/vanedb/pull/220)): add repo secret
`DEMO_REPO_TOKEN` (contents:write on `vanedb/obsidian-vane-search`), then
Actions → **Cut demo 0.2.0** → Run workflow. Or use the local one-shot above
with demo-repo write.

Manual equivalent:

```bash
git clone https://github.com/vanedb/obsidian-vane-search.git
cd obsidian-vane-search
git apply /path/to/vanedb/docs/launch/0003-obsidian-vane-search-0.2.0.patch
# or: curl -fsSL https://raw.githubusercontent.com/vanedb/vanedb/<tip>/docs/launch/0003-obsidian-vane-search-0.2.0.patch | git apply
npm test && npm run build
git commit -am "chore(release): 0.2.0 demo slice for vanedb#198"
git tag 0.2.0 && git push --follow-tags
```

The demo repo's `release` workflow publishes `main.js` + `manifest.json` +
`LICENSE` when the tag matches `manifest.json` version.

## Evidence required on #226 (residual of #198)

1. Release URL for `0.2.0` (or successor).
2. Confirmation the "Try it on a real vault" walkthrough is in that release's
   README.

Until then, vanedb README links the demo repo but does **not** claim a shipped
0.2.0 walkthrough.
