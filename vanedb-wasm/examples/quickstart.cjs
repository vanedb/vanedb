const { ApproxIndex } = require('../pkg/vanedb_wasm.js');

const index = new ApproxIndex(3, 'cosine', 100, 16, 200);
const ids = new BigUint64Array([101n, 202n]);
const vectors = new Float32Array([1, 0, 0, 0, 1, 0]);
index.add_batch(ids, vectors);
const hits = index.search(new Float32Array([1, 0, 0]), 1);
try {
    if (hits.ids[0] !== 101n || hits.distances[0] !== 0) {
        throw new Error('Unexpected nearest neighbor');
    }
    console.log(`Nearest id: ${hits.ids[0]}, distance: ${hits.distances[0]}`);
} finally {
    hits.free();
    index.free();
}
