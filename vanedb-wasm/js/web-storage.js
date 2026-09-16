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

export function indexedDbStorage(dbName = 'vanedb') {
  if (typeof indexedDB === 'undefined') {
    throw new Error('indexedDbStorage requires IndexedDB');
  }
  const open = () =>
    new Promise((resolve, reject) => {
      const request = indexedDB.open(dbName, 1);
      request.onupgradeneeded = () => {
        const db = request.result;
        if (!db.objectStoreNames.contains(STORE)) {
          db.createObjectStore(STORE);
        }
      };
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });

  return {
    async put(name, bytes) {
      const db = await open();
      try {
        await new Promise((resolve, reject) => {
          const tx = db.transaction(STORE, 'readwrite');
          tx.oncomplete = () => resolve();
          tx.onerror = () => reject(tx.error);
          tx.objectStore(STORE).put(new Blob([bytes]), name);
        });
      } finally {
        db.close();
      }
    },
    async get(name) {
      const db = await open();
      try {
        const value = await new Promise((resolve, reject) => {
          const tx = db.transaction(STORE, 'readonly');
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
          tx.onerror = () => reject(tx.error);
          tx.objectStore(STORE).delete(name);
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
