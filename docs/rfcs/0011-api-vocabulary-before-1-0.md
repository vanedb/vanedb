# RFC 0011: API vocabulary before 1.0

- Status: implemented (0.2.0, unreleased)
- Milestone: 0.2.0
- Tracking issue: #206; folds #85 and #86
- Supersedes / superseded by: none

## Problem

Several naming questions were settled provisionally before 0.1.1 and become
permanent at 1.0. They were tracked as cross-engine parity issues (#85, #86)
when two Python packages shipped; the C++ package is no longer published, so
what remains is one engine's vocabulary across four bindings:

- `get` and `get_vector` raise on a miss. The roadmap's precedent survey
  (py-lmdb, plyvel, rocksdict, python-rocksdb, redis-py, usearch,
  `collections.abc.Mapping.get`) found returning `None` is the settled
  convention for a method named `get`.
- The Rust core spells the count both `len()` and `size()` on the same types.
- Python `Metric` is a PyO3 `eq_int` class without `.name`, `.value` or
  iteration, unlike every other Python enum a user has met.
- `ef_search` is a property in Python and WebAssembly and a getter/setter
  pair in Rust and C; the WebAssembly property is spelled `ef_search` while
  its own search option is spelled `efSearch`.
- `is_empty` exists in Rust and Python, not in WebAssembly or C (#182).

0.2.0 is the last cheap moment: the user count is zero and 0.x permits a
breaking change in a minor release. Decision 13 of the 2026-09-13 review
accepted the `get` change; this RFC records it and settles the rest in one
place so nothing is renamed twice.

## Decision

One vocabulary, chosen per language idiom, with the same meaning everywhere.
A lookup miss is a value, not an error. The count has one canonical spelling
per binding. Every rename lands in 0.2.0 together, with a changelog entry per
binding.

## Design

| Concept | Rust | Python | WebAssembly | C ABI |
|---|---|---|---|---|
| Lookup miss | `get(id) -> Result<Option<Vec<f32>>>` (`Option<Cow>` on `DiskIndex`; see the decisions below) | `get(id) -> list[float] \| None`; `KeyError` removed | `get(id)` returns `undefined` | unchanged: `VANEDB_RS_NOT_FOUND` status |
| `get_vector` | alias of `get`, kept on every type (#85 settled the pair) | same | same | same |
| Count | `len()`; `size()` removed | `len(index)`; `size()` kept as an alias, documented as such | `size()` (Map/Set convention) | `_len` accessors on all three handles (present since #152) |
| Empty | `is_empty()` | truth testing (`not index`) | `size() === 0` | `_len() == 0` |
| Beam default | `ef_search()` / `set_ef_search()` | `ef_search` property | property renamed `efSearch` (via `wasm_bindgen(getter, setter, js_name)`) | `_ef_search` / `_set_ef_search` |
| `Metric` | enum | `enum.IntEnum`-like: `.name`, `.value`, iterable, hashable, picklable | string | `uint32_t` |
| Errors | `VaneError` | deferred to #262 | deferred to #262 | status codes |

Rules that follow:

- `contains(id)` stays on every binding; it is cheaper than `get` when the
  vector is not needed.
- `remove(id)` on a missing id stays an error: a caller that removes what is
  not there has a bug, and `remove` is not named `get`.
- No binding gains a method the others cannot express.

## Decisions recorded

- 2026-09-20, implementation: the Rust `get` returns `Result<Option<Vec<f32>>>`
  on `FlatIndex` and `ApproxIndex`, not `Option<&[f32]>` as first written.
  Both keep their vectors behind a `RwLock`, so a slice cannot be borrowed
  out of `&self` without handing the caller a lock guard; the owned copy 0.1.1
  shipped stays, wrapped in `Option`. `DiskIndex` borrows from its mapping
  and returns `Option<Cow<'_, [f32]>>` as the table says.
- 2026-09-20, implementation: with `KeyError` gone from the Python binding,
  `remove` of a missing id raises `ValueError`, the validation bucket it left
  in 0.1.1. Typed Python exception classes and a wasm `Error` with `code`
  were in the first draft of the Errors row; nothing in this RFC's acceptance
  criteria or migration section asks for them, this implementation adds no
  exception classes, and they are tracked in #262.
- 2026-09-20, implementation: the C ABI already carried every accessor the
  table names (`_len` on all three handles, `_ef_search`/`_set_ef_search`),
  so that column is asserted, not changed, and the regenerated header is
  identical.

## The precedent survey (moved from the roadmap)

An independent precedent survey done for vanedb#153 found that `.get()`
returning `None` on a miss is more settled across keyed stores than raising:
py-lmdb, plyvel, rocksdict, python-rocksdb, redis-py, usearch and
`collections.abc.Mapping.get` all do it. That makes a raising method named
`get` the outlier, independently of which exception it raises. The frozen C++
package was inconsistent with itself (`FlatIndex.get` and `DiskIndex.get`
returned `None`, `ApproxIndex.get_vector` threw), so there was no convention
on that side to match. 0.1.1 kept the raising behaviour because the Rust core
had no `Option`-returning accessor to bind and `contains` was already the
non-raising probe.

## Alternatives rejected

- **Add `try_get` and keep `get` raising.** Two names for one read, forever,
  to avoid one breaking change at a time when nobody is broken.
- **Keep both `len` and `size` in Rust.** Two names for one count in the one
  language whose API guidelines pick `len`.
- **Make Python `size()` go away too.** It is harmless, already shipped, and
  is the spelling C++ and JavaScript users type first.

## Compatibility and migration

Breaking in 0.2.0 for Rust (`size` removed, `get` returns `Option`), Python
(`get` returns `None`), and WebAssembly (`get` returns `undefined`, method
pair replaced by a property). The C ABI is unchanged. `CHANGELOG.md` lists
each with a one-line migration. No file-format change.

## Acceptance criteria

- [x] Rust: `get`/`get_vector` return `Option`; `size()` removed; `len()`
      and `is_empty()` on all three types; docs updated.
- [x] Python: `get` returns `None`; `KeyError` no longer raised by any method;
      `Metric` has `.name`, `.value`, iteration; stubs updated.
- [x] WebAssembly: `get` returns `undefined`; `efSearch` property; TypeScript
      declarations updated.
- [x] C ABI: `_len` accessors present (landed in #152); header regenerated,
      byte-identical; nothing removed.
- [x] Conformance: the binding-parity test table in `conformance/` lists the
      vocabulary above and each binding's test asserts its column
      (`conformance/vocabulary/README.md`).
- [x] `CHANGELOG.md` entries; README API section updated; #85, #86 and the
      roadmap's open question closed by the PR.

## Evidence required before the claim

Binding test suites on every CI platform; no hardware claim.

## Out of scope

New functionality. Anything RFC 0004 (filters) or 0009 (payloads) adds
follows this vocabulary when it lands.
