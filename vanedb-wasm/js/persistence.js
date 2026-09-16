// Shared save/load wiring for the wasm ApproxIndex class.
// `defaultStorage` is chosen by the Node or browser entry, so application
// code can say `index.save(name)` in both runtimes.

export function installPersistence(ApproxIndex, defaultStorage) {
  ApproxIndex.prototype.save = async function save(name, storage) {
    assertIndexName(name);
    // Copy off wasm linear memory before handing bytes to storage. WebKit
    // cannot structured-clone a Uint8Array that views WebAssembly.Memory,
    // and Buffer#slice is a view rather than a copy.
    const bytes = new Uint8Array(this.toBytes());
    await (storage ?? defaultStorage).put(name, bytes);
  };
  ApproxIndex.load = async function load(name, storage) {
    assertIndexName(name);
    const bytes = await (storage ?? defaultStorage).get(name);
    if (bytes == null) return null;
    return ApproxIndex.fromBytes(bytes);
  };
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
