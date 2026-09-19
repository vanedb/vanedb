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
    // Copy off wasm linear memory before handing bytes to storage. WebKit
    // cannot structured-clone a Uint8Array that views WebAssembly.Memory,
    // and Buffer#slice is a view rather than a copy.
    const copy = new Uint8Array(this.toBytes());
    await (storage ?? defaultStorage).put(name, copy);
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
