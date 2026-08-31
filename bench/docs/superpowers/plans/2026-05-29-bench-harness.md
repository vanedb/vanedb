# vanedb-bench Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Rust + criterion harness that drives both VaneDB implementations through their C ABIs (`vanedb_cpp_*` linked from the C++ static lib, `vanedb_rs_*` from the `vanedb-capi` crate), on identical seeded workloads, reporting latency for all ops plus recall@k for HNSW, with a `RESULTS.md` generator.

**Architecture:** `build.rs` builds `libvanedb_cpp_capi.a` from the `vendor/vanedb-cpp` git submodule via the `cmake` crate and emits link directives; `ffi.rs` declares the C++ `extern "C"` symbols; the Rust core is reached through the `vanedb-capi` dependency's `#[no_mangle]` functions. Both families are reached as single non-inlined calls, so call overhead is equal. `workloads.rs` generates deterministic vectors via splitmix64 (no `rand` dependency, so the harness is immune to rand-version drift). `ground_truth.rs` computes brute-force top-k for recall@k. criterion benches compare cpp-vs-rs per op; `report.rs` emits the summary table.

**Tech Stack:** Rust 2021, `criterion`, `cmake` (build-dep), the C++ toolchain (for the submodule build), `vanedb-capi` (git dep).

**Spec:** `docs/superpowers/specs/2026-05-28-vanedb-bench-design.md`, all sections.

**Prerequisite:** Plans 1 (`vanedb-cpp` C API) and 2 (`vanedb` `vanedb-capi` crate) must be merged first — this harness links artifacts both produce.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `Cargo.toml` | criterion dev-dep, `vanedb-capi` git dep, `cmake` build-dep; 4 `[[bench]]` entries; `[[bin]] report` |
| `build.rs` | Build + link C++ `vanedb_cpp_capi` static lib via cmake; link C++ stdlib |
| `vendor/vanedb-cpp` | git submodule, pinned SHA |
| `src/lib.rs` | Module decls + the `Impl` enum and safe-call dispatch |
| `src/ffi.rs` | `extern "C"` declarations for `vanedb_cpp_*`; re-export `vanedb_rs_*` from `vanedb_capi` |
| `src/workloads.rs` | splitmix64 deterministic vector/query generation |
| `src/ground_truth.rs` | brute-force top-k + `recall_at_k` |
| `benches/distance.rs` | distance latency: cpp vs rs |
| `benches/store.rs` | VectorStore add + search latency: cpp vs rs |
| `benches/hnsw.rs` | HNSW build + search latency: cpp vs rs |
| `benches/mmap.rs` | MMap build/open + search latency: cpp vs rs |
| `src/bin/report.rs` | Standalone timing + recall@k → `RESULTS.md` |

**Metric encoding:** `0=L2, 1=Cosine, 2=Dot` (both C ABIs take the metric as an int).

---

### Task 1: Scaffold cargo project + C++ submodule + build.rs linking

> **Highest-risk task** — the cmake/link wiring is platform-sensitive. Expect to adjust the link-search path in Step 4 after the first build reveals where CMake placed `libvanedb_cpp_capi.a`.

**Files:**
- Create: `Cargo.toml`, `build.rs`, `src/lib.rs`, `src/ffi.rs`
- Add submodule: `vendor/vanedb-cpp`

- [ ] **Step 1: Add the C++ implementation as a pinned submodule**

Run:
```bash
git submodule add https://github.com/vanedb/vanedb-cpp vendor/vanedb-cpp
git -C vendor/vanedb-cpp checkout main && git -C vendor/vanedb-cpp pull
git add .gitmodules vendor/vanedb-cpp
```

- [ ] **Step 2: Create the manifest**

Create `Cargo.toml`:

```toml
[package]
name = "vanedb-bench"
version = "0.1.0"
edition = "2021"

[dependencies]
vanedb-capi = { git = "https://github.com/vanedb/vanedb", branch = "main" }

[build-dependencies]
cmake = "0.1"

[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }

[[bench]]
name = "distance"
harness = false
[[bench]]
name = "store"
harness = false
[[bench]]
name = "hnsw"
harness = false
[[bench]]
name = "mmap"
harness = false

[[bin]]
name = "report"
path = "src/bin/report.rs"
```

(Pin `vanedb-capi` to an exact `rev = "<sha>"` once Plan 2 is merged; `branch = "main"` is fine for initial bring-up.)

- [ ] **Step 3: Create build.rs (build + link the C++ static lib)**

Create `build.rs`:

```rust
fn main() {
    let dst = cmake::Config::new("vendor/vanedb-cpp")
        .define("VANEDB_BUILD_CAPI", "ON")
        .define("VANEDB_BUILD_TESTS", "OFF")
        .define("VANEDB_BUILD_BENCHMARKS", "OFF")
        .define("VANEDB_BUILD_PYTHON", "OFF")
        .define("VANEDB_BUILD_EXAMPLES", "OFF")
        .build_target("vanedb_cpp_capi")
        .build();

    // CMake (non-install) places the archive under <dst>/build.
    println!("cargo:rustc-link-search=native={}/build", dst.display());
    println!("cargo:rustc-link-lib=static=vanedb_cpp_capi");

    // C++ standard library.
    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=dylib=c++");
    } else {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }
    println!("cargo:rerun-if-changed=vendor/vanedb-cpp/capi/vanedb_capi.cpp");
}
```

- [ ] **Step 4: Create ffi.rs (declare C++ symbols; re-export Rust symbols)**

Create `src/ffi.rs`:

```rust
//! Both implementations are reached through their C ABI as non-inlined calls.
use std::os::raw::c_char;

// --- C++ side: declared here, resolved from libvanedb_cpp_capi.a ---
extern "C" {
    pub fn vanedb_cpp_l2_sq(a: *const f32, b: *const f32, dim: usize) -> f32;
    pub fn vanedb_cpp_cosine_distance(a: *const f32, b: *const f32, dim: usize) -> f32;
    pub fn vanedb_cpp_dot_product(a: *const f32, b: *const f32, dim: usize) -> f32;

    pub fn vanedb_cpp_store_new(dim: usize, metric: u32) -> *mut std::ffi::c_void;
    pub fn vanedb_cpp_store_add(s: *mut std::ffi::c_void, id: u64, v: *const f32) -> i32;
    pub fn vanedb_cpp_store_search(s: *mut std::ffi::c_void, q: *const f32, k: usize,
                                   out_ids: *mut u64, out_dists: *mut f32) -> usize;
    pub fn vanedb_cpp_store_free(s: *mut std::ffi::c_void);

    pub fn vanedb_cpp_hnsw_new(dim: usize, metric: u32, capacity: usize, m: usize,
                               ef_construction: usize, seed: u64) -> *mut std::ffi::c_void;
    pub fn vanedb_cpp_hnsw_add(h: *mut std::ffi::c_void, id: u64, v: *const f32) -> i32;
    pub fn vanedb_cpp_hnsw_search(h: *mut std::ffi::c_void, q: *const f32, k: usize, ef: usize,
                                  out_ids: *mut u64, out_dists: *mut f32) -> usize;
    pub fn vanedb_cpp_hnsw_save(h: *mut std::ffi::c_void, path: *const c_char) -> i32;
    pub fn vanedb_cpp_hnsw_load(path: *const c_char) -> *mut std::ffi::c_void;
    pub fn vanedb_cpp_hnsw_free(h: *mut std::ffi::c_void);

    pub fn vanedb_cpp_mmap_build(path: *const c_char, dim: usize, metric: u32,
                                 ids: *const u64, vecs: *const f32, n: usize) -> i32;
    pub fn vanedb_cpp_mmap_open(path: *const c_char) -> *mut std::ffi::c_void;
    pub fn vanedb_cpp_mmap_search(m: *mut std::ffi::c_void, q: *const f32, k: usize,
                                  out_ids: *mut u64, out_dists: *mut f32) -> usize;
    pub fn vanedb_cpp_mmap_free(m: *mut std::ffi::c_void);
}

// --- Rust side: re-exported from the vanedb-capi crate (same #[no_mangle] symbols) ---
pub use vanedb_capi::{
    vanedb_rs_cosine_distance, vanedb_rs_dot_product, vanedb_rs_l2_sq,
    vanedb_rs_hnsw_add, vanedb_rs_hnsw_free, vanedb_rs_hnsw_load, vanedb_rs_hnsw_new,
    vanedb_rs_hnsw_save, vanedb_rs_hnsw_search,
    vanedb_rs_mmap_build, vanedb_rs_mmap_free, vanedb_rs_mmap_open, vanedb_rs_mmap_search,
    vanedb_rs_store_add, vanedb_rs_store_free, vanedb_rs_store_new, vanedb_rs_store_search,
};
```

- [ ] **Step 5: Create lib.rs**

Create `src/lib.rs`:

```rust
pub mod ffi;
pub mod ground_truth;
pub mod workloads;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Impl { Cpp, Rs }
```

- [ ] **Step 6: Verify it builds and links**

Run:
```bash
cargo build 2>&1 | tail -15
```
Expected: the C++ submodule configures+builds, the archive links, and the crate compiles. **If linking fails with "cannot find -lvanedb_cpp_capi"**, run `find target -name 'libvanedb_cpp_capi.a'` and adjust the `rustc-link-search` path in `build.rs` Step 3 to the directory that actually contains it (CMake generators differ; it may be under `<dst>/build/<config>` or just `<dst>/build`).

- [ ] **Step 7: Commit**

```bash
git add Cargo.toml build.rs src/lib.rs src/ffi.rs .gitmodules vendor/vanedb-cpp
git commit -m "feat(bench): scaffold harness, C++ submodule, FFI linking"
```

---

### Task 2: Deterministic workloads

**Files:**
- Create: `src/workloads.rs`
- Test: inline `#[cfg(test)]`

- [ ] **Step 1: Write the failing test**

Create `src/workloads.rs`:

```rust
//! Deterministic vector generation via splitmix64 — no rand dependency, so the
//! harness is reproducible regardless of any implementation's RNG choices.

pub struct Workload {
    pub dim: usize,
    pub vectors: Vec<f32>, // row-major, n * dim
    pub ids: Vec<u64>,
    pub queries: Vec<f32>, // row-major, n_queries * dim
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

fn unit_f32(state: &mut u64) -> f32 {
    // 24-bit mantissa → [0,1)
    (splitmix64(state) >> 40) as f32 / (1u64 << 24) as f32
}

pub fn generate(seed: u64, dim: usize, n: usize, n_queries: usize) -> Workload {
    let mut s = seed;
    let mut vectors = Vec::with_capacity(n * dim);
    for _ in 0..n * dim { vectors.push(unit_f32(&mut s)); }
    let ids = (0..n as u64).collect();
    let mut queries = Vec::with_capacity(n_queries * dim);
    for _ in 0..n_queries * dim { queries.push(unit_f32(&mut s)); }
    Workload { dim, vectors, ids, queries }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn deterministic_and_shaped() {
        let a = generate(42, 8, 100, 5);
        let b = generate(42, 8, 100, 5);
        assert_eq!(a.vectors, b.vectors); // same seed => identical
        assert_eq!(a.vectors.len(), 800);
        assert_eq!(a.queries.len(), 40);
        assert_eq!(a.ids.len(), 100);
        let c = generate(43, 8, 100, 5);
        assert_ne!(a.vectors, c.vectors); // different seed => different
    }
}
```

- [ ] **Step 2: Run to verify it passes** (this task is generation logic; the test validates determinism)

Run: `cargo test workloads 2>&1 | tail -5`
Expected: `test workloads::tests::deterministic_and_shaped ... ok`.

- [ ] **Step 3: Commit**

```bash
git add src/workloads.rs
git commit -m "feat(bench): deterministic splitmix64 workloads"
```

---

### Task 3: Ground truth + recall@k

**Files:**
- Create: `src/ground_truth.rs`

- [ ] **Step 1: Write the failing test**

Create `src/ground_truth.rs`:

```rust
//! Brute-force top-k (the reference) and recall@k for approximate results.

fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| { let d = x - y; d * d }).sum()
}

/// Exact top-k ids for one query against row-major `vectors` (n * dim).
pub fn brute_force_topk(vectors: &[f32], ids: &[u64], dim: usize, query: &[f32], k: usize) -> Vec<u64> {
    let mut scored: Vec<(f32, u64)> = ids.iter().enumerate()
        .map(|(i, &id)| (l2_sq(query, &vectors[i * dim..(i + 1) * dim]), id))
        .collect();
    scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    scored.iter().take(k).map(|&(_, id)| id).collect()
}

/// recall@k = |returned ∩ truth| / k, for one query.
pub fn recall_at_k(returned: &[u64], truth: &[u64]) -> f32 {
    if truth.is_empty() { return 1.0; }
    let hits = returned.iter().filter(|id| truth.contains(id)).count();
    hits as f32 / truth.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn topk_and_recall() {
        let vectors = [0.0, 0.0,  1.0, 1.0,  5.0, 5.0]; // ids 0,1,2
        let ids = [0u64, 1, 2];
        let q = [0.1, 0.1];
        let truth = brute_force_topk(&vectors, &ids, 2, &q, 2);
        assert_eq!(truth, vec![0, 1]); // nearest two
        assert!((recall_at_k(&[0, 1], &truth) - 1.0).abs() < 1e-6);
        assert!((recall_at_k(&[0, 2], &truth) - 0.5).abs() < 1e-6);
    }
}
```

- [ ] **Step 2: Run to verify it passes**

Run: `cargo test ground_truth 2>&1 | tail -5`
Expected: `test ground_truth::tests::topk_and_recall ... ok`.

- [ ] **Step 3: Commit**

```bash
git add src/ground_truth.rs
git commit -m "feat(bench): brute-force ground truth + recall@k"
```

---

### Task 4: Distance + store benches

**Files:**
- Create: `benches/distance.rs`
- Create: `benches/store.rs`

- [ ] **Step 1: Write the distance bench**

Create `benches/distance.rs`:

```rust
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;
use vanedb_bench::{ffi, workloads};

fn bench_distance(c: &mut Criterion) {
    for &dim in &[128usize, 768] {
        let w = workloads::generate(1, dim, 2, 0);
        let a = &w.vectors[0..dim];
        let b = &w.vectors[dim..2 * dim];
        let mut g = c.benchmark_group(format!("l2_sq/dim={dim}"));
        g.bench_with_input(BenchmarkId::new("cpp", dim), &dim, |bn, &d| {
            bn.iter(|| unsafe { ffi::vanedb_cpp_l2_sq(black_box(a.as_ptr()), black_box(b.as_ptr()), d) });
        });
        g.bench_with_input(BenchmarkId::new("rs", dim), &dim, |bn, &d| {
            bn.iter(|| unsafe { ffi::vanedb_rs_l2_sq(black_box(a.as_ptr()), black_box(b.as_ptr()), d) });
        });
        g.finish();
    }
}

criterion_group!(benches, bench_distance);
criterion_main!(benches);
```

- [ ] **Step 2: Run it**

Run: `cargo bench --bench distance 2>&1 | tail -20`
Expected: criterion reports `l2_sq/dim=128/cpp`, `/rs`, `l2_sq/dim=768/cpp`, `/rs` with timings; no panics.

- [ ] **Step 3: Write the store bench**

Create `benches/store.rs`:

```rust
use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;
use vanedb_bench::{ffi, workloads};

fn bench_store_search(c: &mut Criterion) {
    let dim = 128usize;
    for &n in &[1_000usize, 10_000] {
        let w = workloads::generate(2, dim, n, 1);
        let q = &w.queries[0..dim];
        let mut g = c.benchmark_group(format!("store_search/n={n}"));

        g.bench_function("cpp", |bn| unsafe {
            let s = ffi::vanedb_cpp_store_new(dim, 0);
            for i in 0..n { ffi::vanedb_cpp_store_add(s, w.ids[i], w.vectors[i * dim..].as_ptr()); }
            let mut ids = [0u64; 10]; let mut ds = [0f32; 10];
            bn.iter(|| ffi::vanedb_cpp_store_search(s, black_box(q.as_ptr()), 10, ids.as_mut_ptr(), ds.as_mut_ptr()));
            ffi::vanedb_cpp_store_free(s);
        });
        g.bench_function("rs", |bn| unsafe {
            let s = ffi::vanedb_rs_store_new(dim, 0);
            for i in 0..n { ffi::vanedb_rs_store_add(s, w.ids[i], w.vectors[i * dim..].as_ptr()); }
            let mut ids = [0u64; 10]; let mut ds = [0f32; 10];
            bn.iter(|| ffi::vanedb_rs_store_search(s, black_box(q.as_ptr()), 10, ids.as_mut_ptr(), ds.as_mut_ptr()));
            ffi::vanedb_rs_store_free(s);
        });
        g.finish();
    }
}

criterion_group!(benches, bench_store_search);
criterion_main!(benches);
```

- [ ] **Step 4: Run it**

Run: `cargo bench --bench store 2>&1 | tail -20`
Expected: `store_search/n=1000/cpp`, `/rs`, `store_search/n=10000/cpp`, `/rs` timings; no panics.

- [ ] **Step 5: Commit**

```bash
git add benches/distance.rs benches/store.rs
git commit -m "feat(bench): distance + store benches (cpp vs rs)"
```

---

### Task 5: HNSW + mmap benches

**Files:**
- Create: `benches/hnsw.rs`
- Create: `benches/mmap.rs`

- [ ] **Step 1: Write the HNSW bench** (build + search; identical seed/params for both)

Create `benches/hnsw.rs`:

```rust
use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;
use vanedb_bench::{ffi, workloads};

const DIM: usize = 128;
const N: usize = 10_000;
const M: usize = 16;
const EFC: usize = 200;
const EFS: usize = 50;
const SEED: u64 = 7;

fn bench_hnsw(c: &mut Criterion) {
    let w = workloads::generate(3, DIM, N, 1);
    let q = &w.queries[0..DIM];

    let mut build = c.benchmark_group("hnsw_build");
    build.sample_size(10);
    build.bench_function("cpp", |bn| unsafe {
        bn.iter(|| {
            let h = ffi::vanedb_cpp_hnsw_new(DIM, 0, N, M, EFC, SEED);
            for i in 0..N { ffi::vanedb_cpp_hnsw_add(h, w.ids[i], w.vectors[i * DIM..].as_ptr()); }
            ffi::vanedb_cpp_hnsw_free(black_box(h));
        });
    });
    build.bench_function("rs", |bn| unsafe {
        bn.iter(|| {
            let h = ffi::vanedb_rs_hnsw_new(DIM, 0, N, M, EFC, SEED);
            for i in 0..N { ffi::vanedb_rs_hnsw_add(h, w.ids[i], w.vectors[i * DIM..].as_ptr()); }
            ffi::vanedb_rs_hnsw_free(black_box(h));
        });
    });
    build.finish();

    // Pre-build once each for the search benchmark.
    let mut search = c.benchmark_group("hnsw_search");
    unsafe {
        let hc = ffi::vanedb_cpp_hnsw_new(DIM, 0, N, M, EFC, SEED);
        let hr = ffi::vanedb_rs_hnsw_new(DIM, 0, N, M, EFC, SEED);
        for i in 0..N {
            ffi::vanedb_cpp_hnsw_add(hc, w.ids[i], w.vectors[i * DIM..].as_ptr());
            ffi::vanedb_rs_hnsw_add(hr, w.ids[i], w.vectors[i * DIM..].as_ptr());
        }
        let mut ids = [0u64; 10]; let mut ds = [0f32; 10];
        search.bench_function("cpp", |bn| bn.iter(|| ffi::vanedb_cpp_hnsw_search(hc, black_box(q.as_ptr()), 10, EFS, ids.as_mut_ptr(), ds.as_mut_ptr())));
        search.bench_function("rs", |bn| bn.iter(|| ffi::vanedb_rs_hnsw_search(hr, black_box(q.as_ptr()), 10, EFS, ids.as_mut_ptr(), ds.as_mut_ptr())));
        ffi::vanedb_cpp_hnsw_free(hc);
        ffi::vanedb_rs_hnsw_free(hr);
    }
    search.finish();
}

criterion_group!(benches, bench_hnsw);
criterion_main!(benches);
```

- [ ] **Step 2: Run it**

Run: `cargo bench --bench hnsw 2>&1 | tail -25`
Expected: `hnsw_build/cpp`, `/rs`, `hnsw_search/cpp`, `/rs` timings; no panics. (recall is measured in Task 6's report, not here.)

- [ ] **Step 3: Write the mmap bench**

Create `benches/mmap.rs`:

```rust
use criterion::{criterion_group, criterion_main, Criterion};
use std::ffi::CString;
use std::hint::black_box;
use vanedb_bench::{ffi, workloads};

const DIM: usize = 128;
const N: usize = 10_000;

fn bench_mmap_search(c: &mut Criterion) {
    let w = workloads::generate(4, DIM, N, 1);
    let q = &w.queries[0..DIM];
    let cpp_path = CString::new("bench_cpp.mmap").unwrap();
    let rs_path = CString::new("bench_rs.mmap").unwrap();

    unsafe {
        ffi::vanedb_cpp_mmap_build(cpp_path.as_ptr(), DIM, 0, w.ids.as_ptr(), w.vectors.as_ptr(), N);
        ffi::vanedb_rs_mmap_build(rs_path.as_ptr(), DIM, 0, w.ids.as_ptr(), w.vectors.as_ptr(), N);
        let mc = ffi::vanedb_cpp_mmap_open(cpp_path.as_ptr());
        let mr = ffi::vanedb_rs_mmap_open(rs_path.as_ptr());
        let mut ids = [0u64; 10]; let mut ds = [0f32; 10];
        let mut g = c.benchmark_group("mmap_search");
        g.bench_function("cpp", |bn| bn.iter(|| ffi::vanedb_cpp_mmap_search(mc, black_box(q.as_ptr()), 10, ids.as_mut_ptr(), ds.as_mut_ptr())));
        g.bench_function("rs", |bn| bn.iter(|| ffi::vanedb_rs_mmap_search(mr, black_box(q.as_ptr()), 10, ids.as_mut_ptr(), ds.as_mut_ptr())));
        g.finish();
        ffi::vanedb_cpp_mmap_free(mc);
        ffi::vanedb_rs_mmap_free(mr);
    }
}

criterion_group!(benches, bench_mmap_search);
criterion_main!(benches);
```

- [ ] **Step 4: Run it**

Run: `cargo bench --bench mmap 2>&1 | tail -15`
Expected: `mmap_search/cpp`, `/rs` timings; no panics.

- [ ] **Step 5: Commit**

```bash
git add benches/hnsw.rs benches/mmap.rs
git commit -m "feat(bench): HNSW + mmap benches (cpp vs rs)"
```

---

### Task 6: RESULTS.md report (timing summary + HNSW recall@k)

**Files:**
- Create: `src/bin/report.rs`

- [ ] **Step 1: Write the report binary**

Create `src/bin/report.rs`. It does its own median-of-N timing (criterion owns the rigorous distributions; this is the digestible table) and computes HNSW recall against brute-force truth.

```rust
use std::hint::black_box;
use std::time::Instant;
use vanedb_bench::{ffi, ground_truth, workloads};

const DIM: usize = 128;
const N: usize = 10_000;
const K: usize = 10;

fn median_ns(mut f: impl FnMut()) -> u128 {
    let mut samples: Vec<u128> = (0..50).map(|_| { let t = Instant::now(); f(); t.elapsed().as_nanos() }).collect();
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn main() {
    let w = workloads::generate(99, DIM, N, 1);
    let q = &w.queries[0..DIM];
    let mut out = String::from("# VaneDB Benchmark Results\n\n");
    out.push_str(&format!("Workload: dim={DIM}, n={N}, k={K}, L2.\n\n"));
    out.push_str("| Op | C++ (ns) | Rust (ns) | ratio (rs/cpp) |\n|---|---:|---:|---:|\n");

    unsafe {
        // Distance
        let a = &w.vectors[0..DIM]; let b = &w.vectors[DIM..2 * DIM];
        let cpp = median_ns(|| { black_box(ffi::vanedb_cpp_l2_sq(a.as_ptr(), b.as_ptr(), DIM)); });
        let rs = median_ns(|| { black_box(ffi::vanedb_rs_l2_sq(a.as_ptr(), b.as_ptr(), DIM)); });
        out.push_str(&format!("| l2_sq | {cpp} | {rs} | {:.2} |\n", rs as f64 / cpp as f64));

        // Store search
        let sc = ffi::vanedb_cpp_store_new(DIM, 0);
        let sr = ffi::vanedb_rs_store_new(DIM, 0);
        for i in 0..N {
            ffi::vanedb_cpp_store_add(sc, w.ids[i], w.vectors[i * DIM..].as_ptr());
            ffi::vanedb_rs_store_add(sr, w.ids[i], w.vectors[i * DIM..].as_ptr());
        }
        let mut ids = [0u64; K]; let mut ds = [0f32; K];
        let cpp = median_ns(|| { ffi::vanedb_cpp_store_search(sc, q.as_ptr(), K, ids.as_mut_ptr(), ds.as_mut_ptr()); });
        let rs = median_ns(|| { ffi::vanedb_rs_store_search(sr, q.as_ptr(), K, ids.as_mut_ptr(), ds.as_mut_ptr()); });
        out.push_str(&format!("| store_search | {cpp} | {rs} | {:.2} |\n", rs as f64 / cpp as f64));
        ffi::vanedb_cpp_store_free(sc);
        ffi::vanedb_rs_store_free(sr);

        // HNSW search + recall@k vs brute-force truth
        let truth = ground_truth::brute_force_topk(&w.vectors, &w.ids, DIM, q, K);
        let hc = ffi::vanedb_cpp_hnsw_new(DIM, 0, N, 16, 200, 7);
        let hr = ffi::vanedb_rs_hnsw_new(DIM, 0, N, 16, 200, 7);
        for i in 0..N {
            ffi::vanedb_cpp_hnsw_add(hc, w.ids[i], w.vectors[i * DIM..].as_ptr());
            ffi::vanedb_rs_hnsw_add(hr, w.ids[i], w.vectors[i * DIM..].as_ptr());
        }
        let mut ic = [0u64; K]; let mut dc = [0f32; K];
        let mut ir = [0u64; K]; let mut dr = [0f32; K];
        let nc = ffi::vanedb_cpp_hnsw_search(hc, q.as_ptr(), K, 50, ic.as_mut_ptr(), dc.as_mut_ptr());
        let nr = ffi::vanedb_rs_hnsw_search(hr, q.as_ptr(), K, 50, ir.as_mut_ptr(), dr.as_mut_ptr());
        let rec_c = ground_truth::recall_at_k(&ic[..nc], &truth);
        let rec_r = ground_truth::recall_at_k(&ir[..nr], &truth);
        let cpp = median_ns(|| { ffi::vanedb_cpp_hnsw_search(hc, q.as_ptr(), K, 50, ic.as_mut_ptr(), dc.as_mut_ptr()); });
        let rs = median_ns(|| { ffi::vanedb_rs_hnsw_search(hr, q.as_ptr(), K, 50, ir.as_mut_ptr(), dr.as_mut_ptr()); });
        out.push_str(&format!("| hnsw_search | {cpp} | {rs} | {:.2} |\n", rs as f64 / cpp as f64));
        ffi::vanedb_cpp_hnsw_free(hc);
        ffi::vanedb_rs_hnsw_free(hr);
        out.push_str(&format!("\nHNSW recall@{K}: C++ {rec_c:.3}, Rust {rec_r:.3}\n"));
    }

    std::fs::write("RESULTS.md", &out).unwrap();
    print!("{out}");
}
```

- [ ] **Step 2: Run the report**

Run: `cargo run --release --bin report 2>&1 | tail -20`
Expected: prints the table, writes `RESULTS.md`, recall values are in `[0,1]` and both reasonably high (e.g. ≥0.8) confirming the indexes are quality-comparable.

- [ ] **Step 3: Verify RESULTS.md exists**

Run: `test -f RESULTS.md && head -12 RESULTS.md`
Expected: the file exists and contains the table.

- [ ] **Step 4: Add RESULTS.md + criterion output to .gitignore (keep generated artifacts out of git, except a committed snapshot if desired)**

Create/append `.gitignore`:

```gitignore
/target
*.mmap
*.bin
```

(Decision: keep `RESULTS.md` tracked as the published snapshot, or ignore it — leave it tracked for the portfolio story.)

- [ ] **Step 5: Commit**

```bash
git add src/bin/report.rs .gitignore RESULTS.md
git commit -m "feat(bench): RESULTS.md report with timings + HNSW recall@k"
```

---

## Self-Review

- **Spec coverage:** §3 scope — distance (Task 4), store (Task 4), HNSW build+search (Task 5) + recall@k (Task 6), mmap (Task 5). §4 architecture — C++ via cmake+FFI, Rust via vanedb-capi dep, both single non-inlined calls (Task 1). §6 fairness — identical seeded inputs (workloads, Task 2), same harness (criterion), HNSW triple = latency + recall (Tasks 5–6), same seed/params for both impls. §7 output — criterion HTML (built-in) + RESULTS.md (Task 6).
- **Placeholder scan:** none. The build.rs link path is flagged as adjust-on-first-build, with the exact diagnostic command — that's an instruction, not a placeholder.
- **Type/signature consistency:** `ffi.rs` C++ decls match the Plan 1 header (opaque handles as `*mut c_void`, metric `u32`, HNSW search takes `ef`, search returns `usize` count); Rust re-exports match the Plan 2 `#[no_mangle]` names exactly; `recall_at_k`/`brute_force_topk` signatures match their call sites in `report.rs`.
- **Cross-plan dependency:** explicitly requires Plans 1 & 2 merged (Task 1 links both artifacts; ffi.rs re-export names must equal Plan 2's exported symbols).
- **Fairness caveat to verify at execution:** confirm the Rust `vanedb_rs_*` calls are not inlined away (they are `#[no_mangle] extern "C"` in a separate crate, so they won't be) — if criterion shows implausibly low Rust distance numbers vs C++, check that `black_box` wraps the pointers (it does in Task 4).
