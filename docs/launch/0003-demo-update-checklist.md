# Demo update patch notes for `obsidian-vane-search` (issue #198)

This agent cannot push to https://github.com/vanedb/obsidian-vane-search
(no write permission). Maintainer apply the ready patch below, cut a release,
and reply on vanedb#198 / PR #212 with the release URL.

## Ready-to-apply patch

[`0003-obsidian-vane-search-0.2.0.patch`](0003-obsidian-vane-search-0.2.0.patch)
bumps `package.json` / `manifest.json` / `versions.json` to `0.2.0` and adds
README section **"Try it on a real vault"** (BRAT → Ollama nomic → index →
one semantic query).

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

## Evidence required on #198 / #212

1. Release URL for `0.2.0` (or successor).
2. Confirmation the "Try it on a real vault" walkthrough is in that release's
   README.

Until then, vanedb README links the demo repo but does **not** claim a shipped
0.2.0 walkthrough.
