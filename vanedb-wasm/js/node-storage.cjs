// Node storage adapters. The filesystem is the default; indexedDbStorage
// exists so the package's named exports match the browser build.

const fs = require('node:fs/promises');
const path = require('node:path');

function assertFileName(name) {
  if (typeof name !== 'string' || name.length === 0) {
    throw new Error('name must be a non-empty string');
  }
  if (path.basename(name) !== name || name === '.' || name === '..') {
    throw new Error('name must be a file name inside the storage directory, not a path');
  }
}

function installPersistence(ApproxIndex, defaultStorage) {
  ApproxIndex.prototype.save = async function save(name, storage) {
    assertIndexName(name);
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

// Keep in lockstep with vanedb-wasm/js/persistence.js.
function assertIndexName(name) {
  if (typeof name !== 'string' || name.length === 0) {
    throw new Error('name must be a non-empty string');
  }
  if (name === '.' || name === '..' || /[\\/]/.test(name)) {
    throw new Error('name must be a file name, not a path');
  }
}

function fileStorage(directory = process.cwd()) {
  return {
    async put(name, bytes) {
      assertFileName(name);
      const dest = path.join(directory, name);
      const temp = path.join(
        directory,
        `.${name}.${process.pid}.${Math.random().toString(16).slice(2)}.tmp`,
      );
      try {
        const handle = await fs.open(temp, 'w');
        try {
          await handle.writeFile(bytes);
          // Same durability as core `save`: data on disk before the rename.
          await handle.sync();
        } finally {
          await handle.close();
        }
        await fs.rename(temp, dest);
      } catch (error) {
        await fs.unlink(temp).catch(() => {});
        throw error;
      }
    },
    async get(name) {
      assertFileName(name);
      try {
        return new Uint8Array(await fs.readFile(path.join(directory, name)));
      } catch (error) {
        if (error && error.code === 'ENOENT') return null;
        throw error;
      }
    },
    async delete(name) {
      assertFileName(name);
      try {
        await fs.unlink(path.join(directory, name));
      } catch (error) {
        if (!error || error.code !== 'ENOENT') throw error;
      }
    },
  };
}

function indexedDbStorage() {
  throw new Error(
    'indexedDbStorage is not available in Node; pass fileStorage() or another Storage',
  );
}

module.exports = { fileStorage, indexedDbStorage, installPersistence };
