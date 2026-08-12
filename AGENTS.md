# vanedb — agent guide

Embeddable vector database for edge AI. Rust workspace with Python/wasm/C
bindings. Two sibling repos: [vanedb-cpp](https://github.com/vanedb/vanedb-cpp)
(header-only C++ twin) and [vanedb-bench](https://github.com/vanedb/vanedb-bench)
(cross-language benchmark harness that pins both engines by rev).

## Workspace

| Crate | What it is | Notes |
|---|---|---|
| `vanedb` | Core: `VectorStore`, `HnswIndex`, `MmapVectorStore` (feature `mmap`), SIMD distance kernels (NEON/AVX2/scalar) | |
| `vanedb-py` | PyO3 bindings | **Excluded from workspace CI** — needs libpython; build with maturin |
| `vanedb-wasm` | wasm-bindgen bindings | needs `wasm32-unknown-unknown` target |
| `vanedb-capi` | C ABI (`vanedb_rs_*` symbols, metric as u32: 0=L2, 1=Cosine, 2=Dot) | header regenerated with cbindgen |

## Build & test

These are exactly what CI runs — use the same invocations:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --exclude vanedb-py --all-targets --features mmap -- -D warnings
cargo test --workspace --exclude vanedb-py --features mmap
```

Never run plain `cargo test --workspace`: `vanedb-py` fails to link outside a
maturin/Python environment. To work on the Python bindings (not covered by CI —
verify locally):

```bash
python -m venv .venv && . .venv/bin/activate && pip install maturin pytest
cd vanedb-py && maturin develop --release && pytest tests
```

wasm tests:

```bash
rustup target add wasm32-unknown-unknown   # plus wasm-pack and node
cd vanedb-wasm && wasm-pack test --node
```

Feature caveats: `gpu-metal` builds/tests only on macOS; `gpu-cuda` needs a CUDA
toolchain. iOS/Android are build-only CI targets (cross-compile; can't run
tests). Don't attempt any of these from a Linux cloud sandbox — CI covers them.

## Invariants — do not break

- **Serialization**: bincode 2.x with `bincode::config::legacy()`, which is the
  bincode-1 wire format. Files saved by any released version must keep loading.
  Never bump to bincode 3.x — it is a protest release that fails compilation by
  design. Dependabot is configured to ignore that major.
- **HNSW file format**: current v2 writes count-sized arrays; legacy v1
  (capacity-sized) must remain loadable. `tests/corruption_tests.rs` mirrors the
  on-disk layout — update the mirror struct in lockstep with any format change,
  and add a fixture proving old files still load.
- **HNSW build parity with vanedb-cpp**: new nodes get `M` initial links (not
  `2M`); overflowing reverse links are pruned by sort+truncate, not the
  diversity heuristic. The two engines must build equivalent graphs.
- **Top-k, not full sorts**: `SearchResult`'s `Ord` tie-breaks on id, which
  makes full sorts slow. Search paths use `select_nth_unstable` + truncate +
  small sort (see `store/vector_store.rs` and `mmap.rs`) — keep that pattern.
- **SIMD kernels** use multi-accumulator unrolling (4 accumulators for l2/dot,
  2-way for cosine) because single-accumulator FMA loops are latency-bound.
  Keep NEON, AVX2, and scalar paths semantically in sync; scalar is the
  reference.

## Cross-repo duties

- Algorithm and persistence-format changes must be ported to vanedb-cpp (file an
  issue there if not doing it in the same session). Feature-level API additions
  (e.g. `add_batch`, metadata) are deliberately **not** ported — vanedb-cpp's
  maintenance posture defers those to this repo.
- After perf-relevant merges, vanedb-bench's pins should be refreshed and the
  snapshot re-run (on dedicated hardware, not CI).

## Performance work

- CI's "Performance Regression Check" compares criterion baselines against main
  with critcmp — it gates obvious regressions but is noisy like any shared
  runner.
- Local noise floor is ~3%; fsync-heavy benches (save paths) can spike ~18% with
  no code change. Never believe a single-run delta: verify with interleaved
  A-B-A(-B) runs before claiming a regression or a win.
- Never make performance claims from cloud/CI timings. Benchmarks are meaningful
  only on dedicated hardware.
- `examples/profile_hnsw_build.rs` builds a 10k×128 index and reports build time
  + recall@10 — the quick profiling loop for graph-construction changes.

## Conventions

- Conventional commits: `feat(scope):`, `fix:`, `perf(scope):`, `chore(deps):`.
- Run `cargo fmt --all` before committing. Don't chain fmt-check and commit with
  `;` — a failed check won't stop the commit.
- `main` is protected: PRs only, required status checks, strict up-to-date
  policy (update the branch after any main movement), auto-merge disabled.

## Environment setup (cloud agents: Codex, Claude Code web, etc.)

Core work needs nothing beyond stable Rust with rustfmt+clippy and network
access to crates.io on first fetch (`Cargo.lock` is committed). For the binding
crates, run [`scripts/agent-setup.sh`](scripts/agent-setup.sh):

```bash
scripts/agent-setup.sh          # core only (components; no-op on stock images)
scripts/agent-setup.sh wasm py  # add wasm and/or python toolchains as needed
```
