# vanedb — agent guide

Embeddable vector database for edge AI. This monorepo contains the Rust engine
and its Python/wasm/C bindings, the header-only C++ engine, the cross-engine
benchmark harness, and the shared conformance contract.

## Workspace

| Crate | What it is | Notes |
|---|---|---|
| `vanedb` | Core: `FlatIndex`, `ApproxIndex`, `DiskIndex` (feature `disk`), SIMD distance kernels (NEON/AVX2/scalar) | |
| `vanedb-py` | PyO3 bindings | Excluded from workspace Cargo tests; CI builds, installs, and tests a wheel |
| `vanedb-wasm` | wasm-bindgen bindings | needs `wasm32-unknown-unknown` target |
| `vanedb-capi` | C ABI (`vanedb_rs_*` symbols, metric as u32: 0=L2, 1=Cosine, 2=Dot) | header regenerated with cbindgen |
| `cpp` | Header-only C++ engine, C ABI, and supplementary `vanedb-cpp` Python package | CMake project; Python imports as `vanedb_cpp` |
| `bench` | Cross-engine benchmark harness | Separate Cargo workspace; builds both local C ABIs |
| `conformance` | Shared fixtures and contract tests | Cross-load and behavioral parity belong here |

## Build & test

Use these local checks; the workflows add platform and installed-artifact checks:

```bash
actionlint
cargo fmt --all -- --check
cargo clippy --workspace --exclude vanedb-py --all-targets --features disk --locked -- -D warnings
cargo test --workspace --exclude vanedb-py --features disk --locked
cargo fmt --manifest-path bench/Cargo.toml --all -- --check
cargo clippy --manifest-path bench/Cargo.toml --all-targets --locked -- -D warnings
cargo test --manifest-path bench/Cargo.toml --locked
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build --parallel
ctest --test-dir cpp/build --output-on-failure
```

Never run plain `cargo test --workspace`: `vanedb-py` fails to link outside a
maturin/Python environment. To work on the Python bindings locally (CI also tests the installed wheel):

```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install maturin pytest numpy packaging
cd vanedb-py
maturin develop --release --locked
python -m pytest tests
```

Verify a locally built C++ wheel carries the deployment floor rather than the
host macOS generation (#52) — the tag must not name the host OS version:

```bash
python -m build --wheel --outdir /tmp/vanedb-cpp-wheel cpp
ls /tmp/vanedb-cpp-wheel        # macosx_11_0_arm64 on Apple Silicon, never macosx_<host>_*
python -m pip install --force-reinstall --no-deps /tmp/vanedb-cpp-wheel/*.whl   # must not be rejected
otool -l <extracted .so> | grep -A3 LC_BUILD_VERSION                           # minos 11.0 on arm64
```

wasm tests:

```bash
rustup target add wasm32-unknown-unknown   # plus wasm-pack and node
cd vanedb-wasm && wasm-pack test --node --locked
```

Feature caveats: `gpu-metal` builds/tests only on macOS. CUDA is excluded from
1.0.0 and is a required, high-priority follow-up in [the roadmap](docs/ROADMAP.md).
Do not reintroduce an unimplemented CUDA feature or claim support without the
roadmap's NVIDIA hardware, correctness, lifecycle and performance evidence. Mobile CI builds iOS ARM64 and Android ARM64/x86-64, then runs C ABI
acceptance on an iOS ARM64 simulator and Android x86-64 emulator. The CI-built
Android ARM64 bundle also passes locally on an Android 15 emulator with 16 KiB
pages; see the release evidence. The September 7 decision accepts simulators
and emulators for 1.0.0 verification. Physical-device checks remain a follow-up, and
no physical-device success may be claimed without a recorded run. Node wasm tests likewise
do not prove browser integration; CI also runs headless Chrome. Match each
platform claim to the actual build and runtime evidence; use the relevant host
and toolchain for checks that cannot run locally.

## Invariants — do not break

- **Persistence contract**: `DiskIndex` already uses shared `VNDB` v1:
  literal `VNDB` magic, fixed-width little-endian fields, and fixtures in
  `vanedb/tests/fixtures/conformance/vndb/` that each engine reads and reproduces. Preserve that
  contract. Approximate graphs now use the shared VNDB v2 format in
  `conformance/graph/`; keep its golden files, cross-load preservation and legacy
  reader checks passing. The contract preserves reads of valid VNDB v1/v2 files
  written by VaneDB 1.0.0 throughout the Rust engine in VaneDB 1.x, within
  documented resource limits. Never reinterpret
  existing format/kind/continuation identifiers; retain old readers when adding
  a new encoding. Future insertion topology is not guaranteed. Keep all corruption checks.
- **Legacy Rust persistence remains stable during the VNDB transition**:
  bincode stays on 2.x with `bincode::config::legacy()` (the bincode-1 wire
  format); Dependabot ignores the intentionally uncompilable 3.x major. Keep
  legacy v1 files loadable, update the mirror in `tests/corruption_tests.rs` in
  lockstep with layout changes, and add a fixture for every compatibility fix.
  These are transition safeguards, not a public-version promise.
- **HNSW cross-engine parity is semantic, not byte-for-byte adjacency**:
  independently built graphs may differ. Each graph must satisfy structural
  invariants and recall expectations, and either engine must preserve the graph
  it loads from the other engine's `VNDB` file.
- **HNSW construction choices are performance-sensitive**: new nodes receive
  `M` initial links (`2M` is only the level-0 reverse-link cap), and overflowing
  reverse lists use distance sort+truncate rather than the diversity heuristic.
  Change either only with cross-engine conformance and interleaved benchmarks.
- **Bounded top-k, not full sorts and not quickselect**: `SearchResult`'s
  `Ord` tie-breaks on id, which makes full sorts slow — but `select_nth_unstable`
  over the whole candidate array is also wrong, because it swaps ~n 16-byte
  structs through a buffer that outgrows L1. Scans stream into a k-element heap
  (`flat/topk.rs`, used by `flat/mod.rs` and `disk.rs`); the graph search sorts
  its ef-bounded candidate set. Keep the bound.
- **SIMD kernels** use multi-accumulator unrolling (4 accumulators for l2/dot,
  2-way for cosine) because single-accumulator FMA loops are latency-bound.
  Keep NEON, AVX2, and scalar paths semantically in sync; scalar is the
  reference.

## Cross-engine duties

**The C++ engine is frozen.** Rust is the engine that ships; `cpp/` is
reference code and the other arm of the benchmark comparison. Do not port
features to it. Change it only to fix a defect in the engine itself, or to
keep a comparison honest.

- Keep it building and passing its own tests, and preserve both shared VNDB
  formats (`bench/tests/cross_engine_format.rs` and `cross_engine_graph.rs`).
  Loaded Rust graph tombstones remain traversable but absent from results;
  this file compatibility does not add a C++ public delete API.
- Do not port feature-level API additions. `add_batch`, growable capacity and
  delete are Rust-only by decision, not by omission.
- Where a Rust change makes a benchmark row measure something different from
  the C++ row, annotate the row rather than porting to match. Two rows in the
  snapshot are annotated for exactly this reason.
- After perf-relevant merges, rerun `bench/` on dedicated hardware. One commit
  identifies both engines, so never reintroduce external revision pins or a
  vendored engine submodule.

The comparison is a dated experiment, not a standing obligation. It has
already paid: vanedb#32, #77, #109 and #110 all exist because of it.

## Performance work

- CI's performance workflow compares criterion baselines against main with
  critcmp. It catches obvious regressions but is noisy like any shared runner.
- Local noise floor is ~3%; fsync-heavy benches (save paths) can spike ~18% with
  no code change. Never believe a single-run delta: verify with interleaved
  A-B-A(-B) runs before claiming a regression or a win.
- Never make performance claims from cloud/CI timings. Benchmarks are meaningful
  only on dedicated hardware.
- `examples/profile_index_build.rs` builds a 10k×128 index and reports build time
  + recall@10 — the quick profiling loop for graph-construction changes.

## Conventions

- Conventional commits: `feat(scope):`, `fix:`, `perf(scope):`, `chore(deps):`.
- Run `cargo fmt --all` before committing. Don't chain fmt-check and commit with
  `;` — a failed check won't stop the commit.
- `main` is protected: PRs only, `Required CI Gate` is the aggregate required
  check, strict up-to-date policy (update the branch after any main movement),
  auto-merge disabled.

## Environment setup (cloud agents: Codex, Claude Code web, etc.)

Core work needs nothing beyond stable Rust with rustfmt+clippy and network
access to crates.io on first fetch (`Cargo.lock` is committed). For the binding
crates, run [`scripts/agent-setup.sh`](scripts/agent-setup.sh):

```bash
scripts/agent-setup.sh          # core only (components; no-op on stock images)
scripts/agent-setup.sh wasm py  # add wasm and/or python toolchains as needed
```
