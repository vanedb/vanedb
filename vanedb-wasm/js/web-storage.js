// Browser storage adapters. IndexedDB is the default; fileStorage exists so
// the package's named exports are the same in Node and the browser, and
// throws if called — there is no filesystem here.

const STORE = 'indexes';

function asUint8Array(value) {
  if (value == null) return null;
  if (value instanceof Uint8Array) return value;
  if (value instanceof ArrayBuffer) return new Uint8Array(value);
  if (typeof Blob !== 'undefined' && value instanceof Blob) {
    return value.arrayBuffer().then((buffer) => new Uint8Array(buffer));
  }
  throw new TypeError('stored value is not bytes');
}

// Store an ArrayBuffer, not a Blob: a Blob wrapping the bytes hangs the
// IndexedDB transaction on WebKitGTK (Playwright's Linux WebKit). The bytes
// from `toBytes()` are already a JS-owned copy, so this is not about wasm
// memory. It normalises whatever `put()` was handed -- possibly a view at an
// offset into a larger buffer, such as a Node Buffer from the 8 KiB pool --
// to a buffer of exactly `bytes.length`, which is what structured clone
// should store. get() accepts ArrayBuffer, Uint8Array and Blob, so an older
// Blob write still loads.
function ownedArrayBuffer(bytes) {
  if (!(bytes instanceof Uint8Array)) {
    throw new TypeError('put requires a Uint8Array');
  }
  if (bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength) {
    return bytes.buffer;
  }
  return bytes.slice().buffer;
}

export function indexedDbStorage(dbName = 'vanedb') {
  const open = () => {
    if (typeof indexedDB === 'undefined') {
      return Promise.reject(new Error('indexedDbStorage requires IndexedDB'));
    }
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(dbName, 1);
      request.onupgradeneeded = () => {
        const db = request.result;
        if (!db.objectStoreNames.contains(STORE)) {
          db.createObjectStore(STORE);
        }
      };
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
      request.onblocked = () => reject(new Error('IndexedDB open blocked'));
    });
  };

  return {
    async put(name, bytes) {
      const payload = ownedArrayBuffer(bytes);
      const db = await open();
      try {
        await new Promise((resolve, reject) => {
          // `durability: 'strict'` asks the engine to flush before `complete`
          // fires; Chromium defaults to relaxed. Ignored where unsupported.
          const tx = db.transaction(STORE, 'readwrite', { durability: 'strict' });
          tx.oncomplete = () => resolve();
          tx.onabort = () => reject(tx.error || new Error('IndexedDB put aborted'));
          tx.onerror = () => reject(tx.error);
          const request = tx.objectStore(STORE).put(payload, name);
          request.onerror = () => reject(request.error);
        });
      } finally {
        db.close();
      }
      // WebKit can drop a just-completed put if navigation starts in the
      // same turn as close(). One macrotask yield is enough for the packaged
      // reload test (save, then `location.replace`, then load) to pass on
      // Chrome, Firefox and WebKit under scripts/test_web_package.py. That is
      // what this buys -- it is not a flush to disk, which no page API can
      // demand; `durability: 'strict'` above is the closest the spec offers.
      await new Promise((resolve) => setTimeout(resolve, 0));
    },
    async get(name) {
      const db = await open();
      try {
        const value = await new Promise((resolve, reject) => {
          const tx = db.transaction(STORE, 'readonly');
          tx.onabort = () => reject(tx.error || new Error('IndexedDB get aborted'));
          tx.onerror = () => reject(tx.error);
          const request = tx.objectStore(STORE).get(name);
          request.onsuccess = () => resolve(request.result ?? null);
          request.onerror = () => reject(request.error);
        });
        return await asUint8Array(value);
      } finally {
        db.close();
      }
    },
    async delete(name) {
      const db = await open();
      try {
        await new Promise((resolve, reject) => {
          const tx = db.transaction(STORE, 'readwrite');
          tx.oncomplete = () => resolve();
          tx.onabort = () => reject(tx.error || new Error('IndexedDB delete aborted'));
          tx.onerror = () => reject(tx.error);
          const request = tx.objectStore(STORE).delete(name);
          request.onerror = () => reject(request.error);
        });
      } finally {
        db.close();
      }
    },
  };
}

export function fileStorage() {
  throw new Error(
    'fileStorage is not available in the browser; pass indexedDbStorage() or another Storage',
  );
}
