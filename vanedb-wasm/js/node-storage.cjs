// Node storage adapters. The filesystem is the default; indexedDbStorage
// exists so the package's named exports match the browser build.
//
// The save/load wiring lives once, in persistence.js; the package build
// generates the CommonJS twin this file's consumer requires.

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


// `directory` omitted means the process's working directory *at the time of
// each call*, not at import. The package installs `fileStorage()` as the
// default when it is first required, and a service that `chdir`s afterwards
// would otherwise keep writing wherever it happened to start.
function fileStorage(directory) {
  const dir = () => directory ?? process.cwd();
  return {
    async put(name, bytes) {
      assertFileName(name);
      const base = dir();
      const dest = path.join(base, name);
      const temp = path.join(
        base,
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
        return new Uint8Array(await fs.readFile(path.join(dir(), name)));
      } catch (error) {
        if (error && error.code === 'ENOENT') return null;
        throw error;
      }
    },
    async delete(name) {
      assertFileName(name);
      try {
        await fs.unlink(path.join(dir(), name));
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

module.exports = { fileStorage, indexedDbStorage };
