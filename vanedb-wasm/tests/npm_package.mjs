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

function checkUpsertAndSearch({ ApproxIndex }) {
  const index = new ApproxIndex(1, 'l2', 4, 4, 16, 7);
  const query = Float32Array.of(0);
  const added = 2n ** 64n - 1n;
  try {
    index.add(10n, query);
    index.add(11n, Float32Array.of(10));
    index.upsert(10n, Float32Array.of(20));
    assert.equal(index.size(), 2, 'replacement keeps the live count');
    assert.equal(index.tombstones(), 1);
    index.upsert(added, query);
    assert.equal(index.size(), 3, 'upsert inserts an absent uint64 ID');
    for (const vector of [Float32Array.of(1, 2), Float32Array.of(NaN)]) {
      assert.throws(() => index.upsert(10n, vector));
      assert.deepEqual([...index.get(10n)], [20], 'rejected upsert preserves the vector');
      assert.equal(index.size(), 3);
      assert.equal(index.tombstones(), 1);
    }
    index.ef_search = 1;
    for (const ef of [undefined, null, 0, 64, 2 ** 32 - 1]) {
      const hits = index.search(query, 2, ef);
      try { assert.deepEqual([...hits.ids], [added, 11n]); }
      finally { hits.free(); }
      assert.equal(index.ef_search, 1, 'per-query beam leaves the default alone');
    }
    for (const ef of [-1, 1.5, NaN, Infinity, -Infinity, 2 ** 32]) {
      assert.throws(() => index.search(query, 2, ef), /ef_search must be an integer/);
      assert.equal(index.ef_search, 1);
    }
    index.compact();
    assert.equal(index.tombstones(), 0);
    assert.deepEqual([...index.get(10n)], [20]);
    assert.deepEqual([...index.get(added)], [0]);
  } finally { index.free(); }
}
checkUpsertAndSearch(esm);

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
checkUpsertAndSearch(cjs);

function checkFilteredSearch(bindings) {
  for (const index of [new bindings.FlatIndex(1, 'l2'), new bindings.ApproxIndex(1, 'l2', 16, 4, 16)]) {
    const query = Float32Array.of(0);
    const max = 2n ** 64n - 1n;
    try {
      index.add(1n, query);
      index.add(max, Float32Array.of(1));
      const hits = index.search(query, 2, { allow: BigUint64Array.of(max) });
      try { assert.deepEqual([...hits.ids], [max]); } finally { hits.free(); }
      assert.throws(() => index.search(query, 2, { allow: [], deny: [] }), /at most one/);
      const error = new Error('predicate failed');
      assert.throws(() => index.search(query, 2, { predicate() { throw error; } }), e => e === error);
      const after = index.search(query, 2);
      try { assert.deepEqual([...after.ids], [1n, max]); } finally { after.free(); }
    } finally { index.free(); }
  }
}
checkFilteredSearch(esm);
checkFilteredSearch(cjs);

const { readFileSync, mkdtempSync, writeFileSync } = await import('node:fs');
const { spawnSync } = await import('node:child_process');
const os = await import('node:os');
const path = await import('node:path');

assert.equal(typeof esm.fileStorage, 'function');
assert.equal(typeof esm.indexedDbStorage, 'function');
assert.equal(typeof cjs.fileStorage, 'function');
assert.throws(
  () => esm.indexedDbStorage(),
  /not available in Node/,
  'the Node build must not pretend IndexedDB exists',
);
assert.throws(
  () => cjs.indexedDbStorage(),
  /not available in Node/,
);

const golden = new Uint8Array(readFileSync('l2_rng1.vndb'));
const fromFixture = esm.ApproxIndex.fromBytes(golden);
try {
  assert.equal(fromFixture.size(), 3);
  assert.deepEqual([...fromFixture.get(101n)], [1, 0]);
  const hits = fromFixture.search(Float32Array.from([1, 0]), 1);
  try { assert.equal(hits.ids[0], 101n); }
  finally { hits.free(); }
  const saved = fromFixture.toBytes();
  assert.deepEqual([...saved], [...golden], 'wasm toBytes must reproduce the VNDB fixture');
  const owned = golden.buffer.slice(golden.byteOffset, golden.byteOffset + golden.byteLength);
  const fromAb = esm.ApproxIndex.fromBytes(owned);
  try { assert.equal(fromAb.size(), 3); }
  finally { fromAb.free(); }
  assert.throws(
    () => esm.ApproxIndex.fromBytes({}),
    /Uint8Array or ArrayBuffer/,
    'a non-buffer must not be reported as a corrupt file',
  );
} finally { fromFixture.free(); }

const persistDir = mkdtempSync(path.join(os.tmpdir(), 'vanedb-persist-'));
const storage = esm.fileStorage(persistDir);
const toSave = new esm.ApproxIndex(3, 'cosine', 16, 4, 16, 7);
toSave.add(101n, Float32Array.from([1, 0, 0]));
toSave.add(202n, Float32Array.from([0, 1, 0]));
await toSave.save('restart.vndb', storage);
toSave.free();
assert.equal(await esm.ApproxIndex.load('missing.vndb', storage), null);

const childScript = path.join(process.cwd(), 'restart-child.mjs');
writeFileSync(childScript, `
import { ApproxIndex, fileStorage } from '@vanedb/wasm';
const storage = fileStorage(${JSON.stringify(persistDir)});
const loaded = await ApproxIndex.load('restart.vndb', storage);
if (!loaded) throw new Error('load after process restart returned null');
if (loaded.size() !== 2) throw new Error('wrong size after restart: ' + loaded.size());
if ([...loaded.get(101n)].join(',') !== '1,0,0') throw new Error('wrong vector after restart');
loaded.free();
`);
const child = spawnSync(process.execPath, [childScript], {
  cwd: process.cwd(),
  encoding: 'utf8',
  env: process.env,
});
assert.equal(child.status, 0, child.stderr || child.stdout || 'child failed');

// A fresh process for each import order: loading the other entry point must
// not rebind the shared class's default storage after a cwd change.
const mixedScript = path.join(process.cwd(), 'mixed-modules-child.mjs');
writeFileSync(mixedScript, `
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
const firstIsEsm = process.argv[2] === 'esm';
const first = firstIsEsm ? await import('@vanedb/wasm') : require('@vanedb/wasm');
const index = new first.ApproxIndex(2, 'l2', 8, 4, 16, 7);
try {
  index.add(42n, Float32Array.of(1, 0));
  await index.save('shared-default');
} finally { index.free(); }
process.chdir(process.argv[3]);
const second = firstIsEsm ? require('@vanedb/wasm') : await import('@vanedb/wasm');
assert.equal(first.ApproxIndex, second.ApproxIndex);
const loaded = await second.ApproxIndex.load('shared-default');
assert.ok(loaded, 'loading the second entry point changed the default storage directory');
try { assert.deepEqual([...loaded.get(42n)], [1, 0]); }
finally { loaded.free(); }
`);
for (const first of ['esm', 'cjs']) {
  const originalCwd = mkdtempSync(path.join(process.cwd(), 'mixed-original-'));
  const changedCwd = mkdtempSync(path.join(process.cwd(), 'mixed-changed-'));
  const mixed = spawnSync(process.execPath, [mixedScript, first, changedCwd], {
    cwd: originalCwd,
    encoding: 'utf8',
    env: process.env,
  });
  assert.equal(mixed.status, 0, mixed.stderr || mixed.stdout || first + ' first failed');
}

await assert.rejects(() => storage.put('../escape', new Uint8Array([1])), /file name/);
const nameless = new esm.ApproxIndex(2, 'l2', 8, 4, 16);
try {
  await assert.rejects(() => nameless.save(''), /non-empty string/);
  await assert.rejects(() => nameless.save('../escape', storage), /file name/);
} finally { nameless.free(); }
await assert.rejects(() => esm.ApproxIndex.load(''), /non-empty string/);

{
  const mem = new Map();
  const custom = {
    async put(name, bytes) { mem.set(name, bytes); },
    async get(name) { return mem.has(name) ? mem.get(name) : null; },
    async delete(name) { mem.delete(name); },
  };
  const idx = new esm.ApproxIndex(2, 'l2', 8, 4, 16, 7);
  idx.add(7n, Float32Array.from([0, 1]));
  await idx.save('mem', custom);
  idx.free();
  const loaded = await esm.ApproxIndex.load('mem', custom);
  try {
    assert.equal(loaded.size(), 1);
    assert.deepEqual([...loaded.get(7n)], [0, 1]);
  } finally { loaded.free(); }
  await custom.delete('mem');
  assert.equal(await esm.ApproxIndex.load('mem', custom), null);
}

assert.equal(typeof cjs.ApproxIndex.load, 'function', 'CJS must expose ApproxIndex.load');
assert.equal(typeof cjs.ApproxIndex.prototype.save, 'function', 'CJS must expose save');
{
  const cjsDir = mkdtempSync(path.join(os.tmpdir(), 'vanedb-cjs-'));
  const cjsStore = cjs.fileStorage(cjsDir);
  const idx = new cjs.ApproxIndex(2, 'l2', 8, 4, 16, 7);
  idx.add(9n, Float32Array.from([1, 0]));
  await idx.save('cjs.vndb', cjsStore);
  idx.free();
  const loaded = await cjs.ApproxIndex.load('cjs.vndb', cjsStore);
  try {
    assert.equal(loaded.size(), 1);
    assert.deepEqual([...loaded.get(9n)], [1, 0]);
  } finally { loaded.free(); }
}

{
  const idx = new esm.ApproxIndex(2, 'l2', 8, 4, 16, 7);
  idx.add(3n, Float32Array.from([0, 1]));
  await idx.save('corpus');
  idx.free();
  const loaded = await esm.ApproxIndex.load('corpus');
  try {
    assert.equal(loaded.size(), 1, 'default fileStorage must persist without an explicit adapter');
    assert.deepEqual([...loaded.get(3n)], [0, 1]);
  } finally { loaded.free(); }
  assert.equal(await esm.ApproxIndex.load('no-such-default'), null);
}

console.log('npm package: ESM and CommonJS consumers both OK');
