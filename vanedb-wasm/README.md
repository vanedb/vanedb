# VaneDB for JavaScript

Vector search in Node.js and the browser, using the Rust engine compiled to
WebAssembly. Bring your own embeddings.

```sh
npm install @vanedb/wasm
```

One package serves both runtimes through conditional `exports`, so the same
source works either way.

```js
import init, { ApproxIndex } from '@vanedb/wasm';

await init();          // no-op under Node, loads the module in a browser

const index = new ApproxIndex(3, 'cosine', 100, 16, 200);
index.add(101n, new Float32Array([1, 0, 0]));

const hits = index.search(new Float32Array([1, 0, 0]), 1);
console.log(hits.ids[0], hits.distances[0]);   // 101n, 0

hits.free();
index.free();
```

`require('@vanedb/wasm')` works too. In a browser, serve your page over HTTP
and use the same import — `init()` fetches the wasm module.

## Indexes

`new ApproxIndex(dimension, metric, capacity, m, ef_construction, seed?)` —
approximate search over an HNSW graph. Capacity is a reserve hint, not a limit:
the index grows past it. `m` and `ef_construction` control graph construction.
`seed` defaults to 42 and fixes the topology for a given insertion order. Read
them back with `m()`, `ef_construction()`, `capacity()` and `seed()`. Set
`index.ef_search` to trade search speed for recall.

`new FlatIndex(dimension, metric)` — exact search, same surface minus the graph
parameters.

Both provide `add`, `add_batch`, `search`, `get`, `get_vector`, `remove`,
`contains`, `size()`, `metric()` and `dimension()`. Both spellings of the read
exist so a program is not tied to one index type. The module also exports
`version()`.

## Deleting

`remove(id)` tombstones a vector: it stops appearing in results immediately but
keeps its graph links, which may be the only route between live
neighbourhoods. `tombstones()` counts what that has cost and `compact()`
reclaims it — worth calling once churn accumulates, since a browser is the most
memory-constrained runtime this package targets.

`upsert(id, vector)` replaces a vector in one operation, inserting it if the id
is absent. It is not `remove` then `add`: those can fail between the halves and
leave the id deleted.

`search` takes an optional beam width — `search(query, k, 64)` — that applies to
that query alone and leaves `index.ef_search` untouched. Use it to spend extra
recall on one hard query without paying for it on every later one.

Persistence (`save`/`load`) and disk mapping are not available in WebAssembly:
there is no filesystem to map.

## Values

Metrics are strings: `"l2"` is squared Euclidean distance, `"cosine"` is cosine
distance, `"dot"` is negative dot product. Lower distances rank first.

Single IDs are unsigned 64-bit `bigint`; batch IDs are a `BigUint64Array`.
JavaScript typed arrays wrap out-of-range values when constructed, so validate
IDs before putting them in a batch array. Vectors are finite `Float32Array`
values matching the index dimension; batch vectors are flattened in row order.
Invalid inputs throw.

`search` returns a `SearchResults` whose `ids` and `distances` share positions.
Copy what you need, then call `free()` on the result and the index to release
WebAssembly memory.

## Building from source

See the [repository](https://github.com/vanedb/vanedb/tree/main/vanedb-wasm).
