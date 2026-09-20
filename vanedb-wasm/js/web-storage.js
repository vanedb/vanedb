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

// wasm-bindgen's Vec<u8> is a view of WebAssembly.Memory. WebKit refuses to
// structured-clone that buffer, and a Blob wrapping it hangs the IndexedDB
// transaction on WebKitGTK (Playwright's Linux WebKit). Copy onto a JS-owned
// ArrayBuffer and store that; get() already accepts ArrayBuffer, Uint8Array
// and Blob so a future Blob write still loads.
function ownedArrayBuffer(bytes) {
  if (!(bytes instanceof Uint8Array)) {
    throw new TypeError('put requires a Uint8Array');
  }
  // `new Uint8Array(typedArray)` copies off wasm linear memory onto a
  // JS-owned buffer of exactly `bytes.length`. `.slice().buffer` is the
  // same for a real Uint8Array; the constructor is unambiguous.
  return new Uint8Array(bytes).buffer;
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
          const tx = db.transaction(STORE, 'readwrite');
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
      // same turn as close(). `await save()` is the durability barrier, so
      // the yield belongs here rather than in every caller.
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
