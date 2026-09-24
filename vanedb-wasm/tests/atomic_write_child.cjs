// A real installed fileStorage write, paused at a deterministic interruption
// point. Only the parent kills this process; no timing-based race is needed.
const fs = require('node:fs/promises');
const { fileStorage } = require('@vanedb/wasm');
const [directory, replacement, checkpoint] = process.argv.slice(2);

async function pause() {
    process.send({ checkpoint });
    await new Promise(() => {});
}

const open = fs.open;
fs.open = async function (...args) {
    const handle = await open.apply(this, args);
    if (checkpoint === 'partial-write') {
        const writeFile = handle.writeFile;
        handle.writeFile = async function (bytes) {
            await writeFile.call(this, bytes.subarray(0, Math.floor(bytes.length / 2)));
            await pause();
        };
    }
    return handle;
};
const rename = fs.rename;
fs.rename = async function (...args) {
    if (checkpoint === 'before-rename') await pause();
    await rename.apply(this, args);
    if (checkpoint === 'after-rename') await pause();
};

fs.readFile(replacement)
    .then(bytes => fileStorage(directory).put('atomic.vndb', bytes))
    .then(() => { throw new Error('write completed without reaching checkpoint'); })
    .catch(error => { console.error(error); process.exit(1); });
