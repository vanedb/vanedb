# Legacy graph compatibility fixtures

These fixed files protect the old graph readers while the public graph format
is developed. They are **not** the future VNDB graph format and do not establish
a new cross-engine compatibility promise.

`generate.py` encodes the old layouts directly with fixed-width little-endian
fields. It does not call either engine's serializer. The committed fixture bytes
and `SHA256SUMS` are the anchor: do not regenerate them to make a changed loader
pass. Add a new fixture when extending compatibility.

Every graph has dimension 2, capacity 4, three occupied slots, `M = 2`, both
beam widths 16, entry slot 0, and maximum level 1. Its vectors are
`[1, 0]`, `[0, 1]`, and `[0.8, 0.2]`; IDs are 101, 202, and `UINT64_MAX`.
Levels are `[1, 0, 1]`. Layer-0 adjacency is `[[1, 2], [0, 2], [0, 1]]`;
layer 1 connects slots 0 and 2. This exercises graph structure beyond a flat
single-layer case.

- Rust files live in `vanedb/tests/fixtures/legacy_graph/` so they also travel
  with the crate's tests. v1 has capacity-sized arrays; v2 has count-sized
  arrays. Both cover L2, cosine, and dot. The additional v2 cosine fixture
  retains a deleted slot whose ID has been reused by a live slot.
- C++ files live in `cpp/tests/fixtures/legacy_graph/`. v1/v2 have
  capacity-sized arrays; v3 has count-sized arrays. Versions 1, 2, and 3 use
  L2, cosine, and dot respectively. v2/v3 include MT19937 state seeded with 42.
  The `indexed` fixtures use libstdc++'s 624 words plus position; `state`
  fixtures use the 624-word initial-state form used by libc++ and
  [MSVC](https://github.com/microsoft/STL/blob/main/stl/inc/random).
  Tests select the runtime's layout. The legacy RNG stream is therefore not
  a universal wire format.

Rust checks IDs, vectors, metrics, graph preservation and subsequent insertion:
it decodes the topology directly, including tombstones, then verifies that a
mutated graph survives save/load. These fixtures are Rust-only. The C++ engine
has its own legacy graph format (`QVRD`) and no fixture here targets it.

Verify the committed files from the repository root:

```sh
shasum -a 256 -c conformance/legacy_graph/SHA256SUMS
cargo test -p vanedb --lib fixed_legacy --locked
ctest --test-dir cpp/build -R 'legacy' --output-on-failure
```
