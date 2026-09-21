// Run against wasm-pack's generated Node package, including its JS conversions.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { FlatIndex, ApproxIndex } = require(path.resolve(process.argv[2]));
// `unchecked_param_type` can accidentally make an Option<JsValue> required
// in the shipped declarations even though two-argument JS calls still work.
const declarations = fs.readFileSync(path.join(path.resolve(process.argv[2]), 'vanedb_wasm.d.ts'), 'utf8');
const searches = declarations.split('\n').filter(line => /^\s+search\(/.test(line));
assert.equal(searches.length, 2);
for (const signature of searches) {
    assert.match(signature, /k: number, \w+\?:/, 'search options must stay optional in TypeScript');
}
// The vocabulary of RFC 0011, as the shipped declarations state it: a lookup
// miss is `undefined`, both read spellings say so on both classes, and the
// beam default is the `efSearch` property rather than a method pair or a
// snake_case name.
const reads = declarations.split('\n').filter(line => /^\s+get(_vector)?\(/.test(line));
assert.equal(reads.length, 4, 'get and get_vector on both classes');
for (const signature of reads) {
    assert.match(signature, /\): Float32Array \| undefined;/, 'a lookup miss must be typed as undefined');
}
assert.match(declarations, /^\s+efSearch: number;/m, 'efSearch must be a property');
assert.doesNotMatch(declarations, /ef_search|setEfSearch|efSearch\(/, 'the method pair and the snake_case property are gone');
const vector = new Float32Array([1]);
const invalidNumbers = [-1, 1.5, NaN, Infinity, -Infinity, 2 ** 32, 2 ** 32 + 1];
const numericError = /must be an integer between 0 and 4294967295/;

for (const value of invalidNumbers) {
    assert.throws(() => new FlatIndex(value, 'l2'), numericError);
    for (const position of [0, 2, 3, 4]) {
        const args = [1, 'l2', 10, 2, 10];
        args[position] = value;
        assert.throws(() => new ApproxIndex(...args), numericError);
    }
}
// Boundary validation preserves the core's distinct zero rules.
assert.throws(() => new FlatIndex(0, 'l2'), /dimension/);
assert.throws(() => new ApproxIndex(0, 'l2', 10, 2, 10), /dimension/);
assert.throws(() => new ApproxIndex(1, 'l2', 0, 2, 10), /capacity/);
assert.throws(() => new ApproxIndex(1, 'l2', 10, 0, 10), /M/);
const zeroConstruction = new ApproxIndex(1, 'l2', 10, 2, 0);
zeroConstruction.free();
// The JavaScript integer boundary is wider than the core's byte-size bound:
// a vector's f32 components must also fit in wasm32's address-size arithmetic.
const largestDimension = 2 ** 30 - 1;
const maxDimension = new FlatIndex(largestDimension, 'l2');
assert.equal(maxDimension.dimension(), largestDimension);
maxDimension.free();
for (const dim of [2 ** 30, 2 ** 32 - 1]) {
    assert.throws(() => new FlatIndex(dim, 'l2'), /overflows usize/);
}

for (const create of [() => new FlatIndex(1, 'l2'), () => new ApproxIndex(1, 'l2', 10, 2, 10)]) {
    const index = create();
    const max = 2n ** 64n - 1n;
    index.add(0n, vector);
    index.add(max, vector);
    for (const k of invalidNumbers) {
        assert.throws(() => index.search(vector, k), numericError);
    }
    assert.throws(() => index.search(vector, 0), /k/);
    const unlimited = index.search(vector, 2 ** 32 - 1);
    assert.equal(unlimited.length, 2);
    unlimited.free();
    if (index instanceof ApproxIndex) {
        assert.equal(index.efSearch, 50, 'documented default');
        assert.equal(index.ef_search, undefined, 'the snake_case property is gone');
        index.efSearch = 73;
        for (const ef of invalidNumbers) {
            assert.throws(() => { index.efSearch = ef; }, numericError);
            assert.equal(index.efSearch, 73);
        }
        for (const ef of [0, 2 ** 32 - 1]) {
            index.efSearch = ef;
            assert.equal(index.efSearch, ef);
        }
    }
    for (const id of [-1n, 2n ** 64n, 2n ** 70n]) {
        assert.throws(() => index.add(id, vector), /id must be between/);
        assert.throws(() => index.contains(id), /id must be between/);
        assert.throws(() => index.remove(id), /id must be between/);
        assert.throws(() => index.get(id), /id must be between/);
        assert.throws(() => index.get_vector(id), /id must be between/);
        assert.equal(index.size(), 2);
        assert.equal(index.contains(0n), true);
        assert.equal(index.contains(max), true);
    }
    const results = index.search(vector, 2);
    assert.deepEqual([...results.ids], [0n, max]);
    assert.equal(results.distances.length, 2);
    results.free();
    index.remove(max);
    assert.equal(index.contains(max), false);
    index.free();
}
// The delete lifecycle, through the generated JS rather than through Rust.
// `remove` shipped without any way to observe or reclaim what it left behind,
// which in a browser is the runtime where that matters most.
{
    const index = new ApproxIndex(1, 'l2', 16, 4, 16);
    for (let i = 0; i < 8; i += 1) index.add(BigInt(i), Float32Array.from([i]));
    assert.equal(index.tombstones(), 0);
    index.remove(3n);
    index.remove(5n);
    assert.equal(index.size(), 6);
    assert.equal(index.tombstones(), 2);
    index.compact();
    assert.equal(index.tombstones(), 0);
    assert.equal(index.size(), 6);
    for (const id of [0n, 1n, 2n, 4n, 6n, 7n]) assert.equal(index.contains(id), true);
    assert.equal(index.contains(3n), false);

    // A stored vector was unreadable from ApproxIndex under either spelling.
    index.add(100n, Float32Array.from([42]));
    assert.deepEqual([...index.get_vector(100n)], [42]);
    assert.deepEqual([...index.get(100n)], [42]);
    // A lookup miss is a value, not an error (RFC 0011).
    assert.equal(index.get_vector(999n), undefined);
    assert.equal(index.get(999n), undefined);
    assert.equal(index.contains(999n), false);
    assert.throws(() => index.remove(999n), /not found/, 'remove is not named get');
    index.free();
}

// The seed was hardcoded, so construction could not be reproduced from JS.
// It stays optional: omitting it must keep the previous default of 42.
{
    const defaulted = new ApproxIndex(2, 'l2', 16, 4, 16);
    assert.equal(defaulted.seed(), 42n);
    const seeded = new ApproxIndex(2, 'l2', 16, 4, 16, 1234);
    assert.equal(seeded.seed(), 1234n);
    assert.equal(seeded.m(), 4);
    assert.equal(seeded.ef_construction(), 16);
    defaulted.free();
    seeded.free();
}

{
    const index = new ApproxIndex(2, 'l2', 16, 4, 16, 7);
    index.add(101n, Float32Array.from([1, 0]));
    const bytes = index.toBytes();
    assert.ok(bytes instanceof Uint8Array);
    assert.deepEqual([...bytes.slice(0, 4)], [...Buffer.from('VNDB')]);
    const loaded = ApproxIndex.fromBytes(bytes);
    assert.equal(loaded.size(), 1);
    assert.deepEqual([...loaded.get(101n)], [1, 0]);
    loaded.free();
    index.free();
}

console.log('Generated JavaScript bindings: passed');

// Public JS argument conversion, error identity, and filter exclusivity are
// invisible to the Rust-only bindings tests.
for (const create of [() => new FlatIndex(1, 'l2'), () => new ApproxIndex(1, 'l2', 16, 4, 16)]) {
    const index = create();
    const query = Float32Array.of(0);
    const ids = [0n, 2n ** 32n, BigInt(Number.MAX_SAFE_INTEGER), 2n ** 64n - 1n];
    ids.forEach((id, i) => index.add(id, Float32Array.of(i)));
    function searchIds(options) {
        const hits = index.search(query, 4, options);
        try { return [...hits.ids]; } finally { hits.free(); }
    }
    try {
        assert.deepEqual(searchIds({ allow: [] }), []);
        assert.deepEqual(searchIds({ deny: [] }), ids);
        assert.deepEqual(searchIds({ allow: [2 ** 32, Number.MAX_SAFE_INTEGER] }), ids.slice(1, 3));
        assert.deepEqual(searchIds({ allow: BigUint64Array.of(ids[3]) }), [ids[3]]);
        for (const options of [
            { allow: [], deny: [] },
            { allow: [0n], predicate: () => true },
            { deny: [0n], predicate: () => true },
        ]) assert.throws(() => searchIds(options), /at most one/);
        for (const invalid of [{}, new Set(), '0', 0, { length: 0 }]) {
            assert.throws(() => searchIds({ allow: invalid }), /array/);
            assert.throws(() => searchIds({ deny: invalid }), /array/);
        }
        for (const invalid of [-1, 1.5, NaN, Infinity, 2 ** 53, -1n, 2n ** 64n, '0', true]) {
            assert.throws(() => searchIds({ allow: [invalid] }));
        }
        for (const invalid of [[1n, 0n], [0n, 0n]]) {
            assert.throws(() => searchIds({ allow: invalid }), /sorted/);
            assert.throws(() => searchIds({ deny: invalid }), /sorted/);
        }
        const error = new Error('metadata unavailable');
        let calls = 0;
        assert.throws(() => searchIds({ predicate() { calls++; throw error; } }), e => e === error);
        assert.equal(calls, 1);
        assert.deepEqual(searchIds(), ids, 'callback errors leave the index usable');
        assert.throws(() => searchIds({ get allow() { throw error; } }), e => e === error);
        const broken = [];
        broken[Symbol.iterator] = () => { throw error; };
        assert.throws(() => searchIds({ allow: broken }), e => e === error);
        if (index instanceof ApproxIndex) {
            for (const field of ['efSearch', 'maxEfSearch']) {
                for (const invalid of ['1', true, -1, 0.5, NaN, Infinity, 2 ** 32]) {
                    assert.throws(() => searchIds({ [field]: invalid }));
                }
            }
            assert.throws(() => searchIds({ get efSearch() { throw error; } }), e => e === error);
            for (const invalid of ['1', true, 1n]) assert.throws(() => searchIds(invalid));
            assert.deepEqual(searchIds({ allow: ids, efSearch: 1, maxEfSearch: 100 }), ids);
        }
    } finally { index.free(); }
}
{
    const index = new ApproxIndex(1, 'l2', 16, 4, 16);
    const other = new ApproxIndex(1, 'l2', 16, 4, 16);
    index.add(1n, Float32Array.of(0));
    other.add(2n, Float32Array.of(0));
    const hits = index.search(Float32Array.of(0), 1, {
        predicate() {
            const nested = other.search(Float32Array.of(0), 1);
            try { return nested.length === 1; } finally { nested.free(); }
        },
    });
    assert.deepEqual([...hits.ids], [1n]);
    hits.free();
    index.free();
    other.free();
}
console.log('Generated JavaScript filtered-search boundaries: passed');
