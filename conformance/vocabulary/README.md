# Binding vocabulary

[RFC 0011](../../docs/rfcs/0011-api-vocabulary-before-1-0.md) settled one
vocabulary across the four bindings of the Rust engine: the same meaning
everywhere, spelled per language idiom. This table is the contract, and the
last row names the test that asserts each column, so a binding cannot drift
from it without a red test. The frozen C++ engine is not a column: it is
reference code, not a shipped binding.

| Concept | Rust | Python | WebAssembly | C ABI |
|---|---|---|---|---|
| Lookup miss | `get(id) -> Result<Option<Vec<f32>>>`; `Result<Option<Cow<[f32]>>>` on `DiskIndex` | `get(id) -> list[float] \| None`; no `KeyError` anywhere | `get(id)` returns `undefined` | `VANEDB_RS_NOT_FOUND` status |
| `get_vector` | alias of `get` on every type | same | same | same |
| Count | `len()` | `len(index)`; `size()` kept as an alias | `size()` | `vanedb_rs_store_len` / `_index_len` / `_disk_len` |
| Empty | `is_empty()` | `not index` | `size() === 0` | `_len() == 0` |
| Beam default | `ef_search()` / `set_ef_search()` | `ef_search` property | `efSearch` property | `vanedb_rs_index_ef_search` / `_set_ef_search` |
| `Metric` | enum | `enum.IntEnum`: `.name`, `.value`, iterable, hashable, picklable | string | `uint32_t` |
| `contains(id)` | yes | yes | yes | `_contains` |
| `remove(id)` of a missing id | `Err(VaneError::NotFound)` | `ValueError` | throws | `VANEDB_RS_NOT_FOUND` |
| Asserted by | `vanedb/tests/public_surface.rs` | `vanedb-py/tests/test_api_parity.py` | `vanedb-wasm/tests/web.rs` and, for the JavaScript spellings, `vanedb-wasm/tests/node.cjs` | `vanedb-capi/tests/capi.rs` |

Rules the table implies:

- `contains(id)` stays on every binding; it is cheaper than `get` when the
  vector is not needed.
- `remove(id)` on a missing id stays an error: a caller that removes what is
  not there has a bug, and `remove` is not named `get`.
- No binding gains a method the others cannot express.
- Anything a later RFC adds follows this vocabulary.

## Per-query search options (RFC 0004)

Filtered search landed with RFC 0004 and follows the vocabulary above: one
filter per query, chosen from a predicate, an allow list or a deny list, with
the beam override and widening cap spelled per language idiom. The C ABI
takes the filter as arguments of `vanedb_rs_*_search_filtered`, not a struct;
a null callback and null lists mean unfiltered, and it exposes no widening
cap of its own.

| Option | Rust (`SearchParams`) | Python (`search` keyword) | WebAssembly (`search` options object) | C ABI (`*_search_filtered` arguments) |
|---|---|---|---|---|
| Predicate | `.filter(Filter::Predicate(&f))` | `filter=` | `predicate` | `vanedb_rs_filter_fn filter, void *user_data` |
| Allow list | `.filter(Filter::Allow(&ids))` | `allow_ids=` | `allow` | `const uint64_t *allow, size_t allow_len` |
| Deny list | `.filter(Filter::Deny(&ids))` | `deny_ids=` | `deny` | `const uint64_t *deny, size_t deny_len` |
| Per-query beam (`ApproxIndex` only) | `.ef_search(n)` | `ef_search=` | a number, or `efSearch` in the object | `size_t ef_search`; `0` means the handle's own |
| Widening cap (`ApproxIndex` only) | `.max_ef_search(n)` | `max_ef_search=` | `maxEfSearch` | not exposed; the default cap applies |
| Asserted by | `vanedb/tests/search_correctness.rs` | `vanedb-py/tests/test_filtered_search.py` | `vanedb-wasm/tests/web.rs` (`test_wasm_filtered_search`) and `vanedb-wasm/tests/node.cjs` | `vanedb-capi/tests/capi.rs` (`filtered_search_argument_contract_across_indexes`) |

## Running the column tests

```sh
cargo test -p vanedb --features disk --test public_surface the_count_is_len_and_is_empty_agrees_on_every_type
cargo test -p vanedb --features disk --test public_surface a_lookup_miss_is_none_and_a_removal_miss_is_an_error
cargo test -p vanedb-capi --test capi the_vocabulary_of_rfc_0011
python -m pytest vanedb-py/tests/test_api_parity.py
cargo test -p vanedb --features disk --test search_correctness filtered_search_exact_matches_reference
cargo test -p vanedb-capi --test capi filtered_search_argument_contract_across_indexes
python -m pytest vanedb-py/tests/test_filtered_search.py
```

The WebAssembly column runs under `wasm-pack test --node --locked` in
`vanedb-wasm/`, and `vanedb-wasm/tests/node.cjs` checks the generated
TypeScript declarations against the JavaScript spellings (`efSearch`, and
`Float32Array | undefined` for both read spellings).
