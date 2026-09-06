// Run against wasm-pack's generated Node package, including its JS conversions.
const assert = require('node:assert/strict');
const path = require('node:path');
const { FlatIndex, ApproxIndex } = require(path.resolve(process.argv[2]));
const vector = new Float32Array([1]);

for (const create of [() => new FlatIndex(1, 'l2'), () => new ApproxIndex(1, 'l2', 10, 2, 10)]) {
    const index = create();
    const max = 2n ** 64n - 1n;
    index.add(0n, vector);
    index.add(max, vector);
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
