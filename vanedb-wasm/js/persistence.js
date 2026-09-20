// Shared save/load wiring for the wasm ApproxIndex class.
// `defaultStorage` is chosen by the Node or browser entry, so application
// code can say `index.save(name)` in both runtimes.

export function installPersistence(ApproxIndex, defaultStorage) {
  const fromBytes = ApproxIndex.fromBytes.bind(ApproxIndex);
  ApproxIndex.fromBytes = function fromBytesWrapped(bytes) {
    return fromBytes(asInputBytes(bytes));
  };
  ApproxIndex.prototype.save = async function save(name, storage) {
    assertIndexName(name);
    // `toBytes()` is already a JS-owned copy: wasm-bindgen returns a
    // `Vec<u8>` as `getArrayU8FromWasm0(..).slice()` and then frees the wasm
    // side, so nothing here aliases linear memory. An earlier version copied
    // again on a misdiagnosis of a WebKitGTK hang that was really the Blob
    // put path, fixed in web-storage.js. Hand the bytes straight to storage.
    await (storage ?? defaultStorage).put(name, this.toBytes());
  };
  ApproxIndex.load = async function load(name, storage) {
    assertIndexName(name);
    const bytes = await (storage ?? defaultStorage).get(name);
    if (bytes == null) return null;
    return ApproxIndex.fromBytes(bytes);
  };
}

function asInputBytes(bytes) {
  if (bytes instanceof Uint8Array) return bytes;
  if (typeof ArrayBuffer !== 'undefined' && bytes instanceof ArrayBuffer) {
    return new Uint8Array(bytes);
  }
  throw new TypeError('fromBytes requires a Uint8Array or ArrayBuffer');
}

// Shared with the Node CJS copy in node-storage.cjs: a name that is a path
// would save in IndexedDB and throw on the filesystem, so the adapters
// disagree. A single path segment works in both.
function assertIndexName(name) {
  if (typeof name !== 'string' || name.length === 0) {
    throw new Error('name must be a non-empty string');
  }
  if (name === '.' || name === '..' || /[\\/]/.test(name)) {
    throw new Error('name must be a file name, not a path');
  }
}
