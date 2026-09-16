# Demo update patch notes for `obsidian-vane-search` (issue #198)

This agent cannot push to https://github.com/vanedb/obsidian-vane-search
(no write permission). Maintainer apply:

1. Bump `package.json`, `manifest.json`, and `versions.json` to `0.2.0` when
   shipping against the VaneDB 0.2.0 milestone (or keep `0.1.x` until wasm
   persistence lands — do not claim unfinished RFCs).
2. Add a README section **"Try it on a real vault"**:
   - Install via BRAT `vanedb/obsidian-vane-search`
   - `ollama pull nomic-embed-text`
   - Settings → Vane Search → Ollama preset
   - Index vault → run one semantic query → show expected hit
3. Cut a GitHub Release matching `manifest.json` version.
4. Reply on vanedb#198 / PR #212 with the release URL so the vanedb README
   demo link is backed by a runnable build.
