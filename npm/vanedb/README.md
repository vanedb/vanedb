# VaneDB

Vector search in Node.js and the browser. Bring your own embeddings.

```sh
npm install vanedb
```

This package re-exports [`@vanedb/wasm`](https://www.npmjs.com/package/@vanedb/wasm),
the WebAssembly build of the Rust engine, at a matching version. Use whichever
specifier you prefer — they are the same code.

```js
import init, { ApproxIndex } from 'vanedb';

await init();          // no-op under Node, loads the module in a browser

const index = new ApproxIndex(3, 'cosine', 100, 16, 200);
index.add(101n, new Float32Array([1, 0, 0]));

const hits = index.search(new Float32Array([1, 0, 0]), 1);
console.log(hits.ids[0], hits.distances[0]);   // 101n, 0

hits.free();
index.free();
```

`require('vanedb')` works too.

Exports `ApproxIndex`, `FlatIndex`, `SearchResults`, `version`, `init` (default)
and `initSync`. The full API is documented in
[`@vanedb/wasm`](https://www.npmjs.com/package/@vanedb/wasm).

Persistence, disk mapping and `upsert` are not available in WebAssembly.

## Source

[github.com/vanedb/vanedb](https://github.com/vanedb/vanedb)
