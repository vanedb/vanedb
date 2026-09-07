// Run against wasm-pack's generated Node package, including its JS conversions.
const assert = require('node:assert/strict');
const path = require('node:path');
const { FlatIndex, ApproxIndex } = require(path.resolve(process.argv[2]));
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
        index.ef_search = 73;
        for (const ef of invalidNumbers) {
            assert.throws(() => { index.ef_search = ef; }, numericError);
            assert.equal(index.ef_search, 73);
        }
        for (const ef of [0, 2 ** 32 - 1]) {
            index.ef_search = ef;
            assert.equal(index.ef_search, ef);
        }
    }
    for (const id of [-1n, 2n ** 64n, 2n ** 70n]) {
        assert.throws(() => index.add(id, vector), /id must be between/);
        assert.throws(() => index.contains(id), /id must be between/);
        assert.throws(() => index.remove(id), /id must be between/);
        if (index.get) assert.throws(() => index.get(id), /id must be between/);
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
console.log('Generated JavaScript bindings: passed');
