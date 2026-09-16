// Shared save/load wiring for the wasm ApproxIndex class.
// `defaultStorage` is chosen by the Node or browser entry, so application
// code can say `index.save(name)` in both runtimes.

export function installPersistence(ApproxIndex, defaultStorage) {
  ApproxIndex.prototype.save = async function save(name, storage) {
    if (typeof name !== 'string' || name.length === 0) {
      throw new Error('name must be a non-empty string');
    }
    // Copy off wasm linear memory before handing bytes to storage. WebKit
    // cannot structured-clone a Uint8Array that views WebAssembly.Memory,
    // and Buffer#slice is a view rather than a copy.
    const bytes = new Uint8Array(this.toBytes());
    await (storage ?? defaultStorage).put(name, bytes);
  };
  ApproxIndex.load = async function load(name, storage) {
    if (typeof name !== 'string' || name.length === 0) {
      throw new Error('name must be a non-empty string');
    }
    const bytes = await (storage ?? defaultStorage).get(name);
    if (bytes == null) return null;
    return ApproxIndex.fromBytes(bytes);
  };
}
