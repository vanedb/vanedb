// Consumes the assembled npm package the way a user would: from an installed
// tarball, through the public specifier, in both module systems.
//
// wasm-pack emits two targets with different contracts — Node is CommonJS and
// loads synchronously, the web build is ESM whose default export is an async
// loader. Merging them under one name is where the failures live, and all
// three found while building this were invisible to any Rust test:
//
//   * ESM `import { ApproxIndex }` from a CommonJS module fails Node's
//     named-export detection.
//   * The root "type": "module" makes Node parse wasm-pack's CommonJS output
//     as ESM and die on its first `exports.` assignment.
//   * A missing export in one target is only visible when both are compared.
//
// Run from inside a directory that has `npm install`ed the tarball, so the
// bare specifier resolves the way a consumer's would — through the package's
// `exports` conditions rather than a path. `scripts/check_npm_package.py`
// sets that up and invokes this.
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const esm = await import('@vanedb/wasm');
await esm.default();                       // no-op on Node; real loader in a browser

assert.equal(typeof esm.version(), 'string', 'version() must be callable');
for (const name of ['FlatIndex', 'ApproxIndex', 'SearchResults']) {
  assert.equal(typeof esm[name], 'function', `${name} must be a named ESM export`);
}

const flat = new esm.FlatIndex(3, 'l2');
flat.add(1n, Float32Array.from([1, 0, 0]));
flat.add(2n, Float32Array.from([0, 1, 0]));
const hits = flat.search(Float32Array.from([1, 0, 0]), 2);
assert.deepEqual([...hits.ids], [1n, 2n], 'nearest first');
assert.ok(Math.abs(hits.distances[0]) < 1e-6);
assert.ok(Math.abs(hits.distances[1] - 2.0) < 1e-5, 'L2 is squared');
hits.free();
flat.free();

// The delete lifecycle, which only reached wasm in #154.
const index = new esm.ApproxIndex(3, 'cosine', 16, 4, 16, 7);
index.add(10n, Float32Array.from([1, 0, 0]));
index.add(11n, Float32Array.from([0, 1, 0]));
assert.equal(index.tombstones(), 0);
index.remove(10n);
assert.equal(index.tombstones(), 1);
index.compact();
assert.equal(index.tombstones(), 0);
assert.equal(index.size(), 1);
assert.equal(index.seed(), 7n, 'the construction seed must round-trip');
index.free();

// The same package through require(), which resolves a different entry point
// under the `node` + `require` condition.
const require = createRequire(import.meta.url);
const cjs = require('@vanedb/wasm');
for (const name of ['FlatIndex', 'ApproxIndex', 'SearchResults', 'version']) {
  assert.ok(cjs[name], `${name} must also be reachable via require()`);
}
const viaRequire = new cjs.FlatIndex(2, 'dot');
viaRequire.add(5n, Float32Array.from([1, 1]));
assert.equal(viaRequire.size(), 1);
viaRequire.free();

console.log('npm package: ESM and CommonJS consumers both OK');
