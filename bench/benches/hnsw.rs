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
    // Asserts inside the timed loop are symmetric across engines and cost
    // nanoseconds against ~0.1 ms inserts; they turn a failed engine into a
    // loud failure instead of an infinitely fast one.
    build.bench_function("cpp", |bn| unsafe {
        bn.iter(|| {
            let h = ffi::vanedb_cpp_hnsw_new(DIM, 0, N, M, EFC, SEED);
            assert!(!h.is_null());
            for i in 0..N {
                assert_eq!(
                    ffi::vanedb_cpp_hnsw_add(h, w.ids[i], w.vectors[i * DIM..].as_ptr()),
                    0
                );
            }
            ffi::vanedb_cpp_hnsw_free(black_box(h));
        });
    });
    build.bench_function("rs", |bn| unsafe {
        bn.iter(|| {
            let h = ffi::vanedb_rs_hnsw_new(DIM, 0, N, M, EFC, SEED);
            assert!(!h.is_null());
            for i in 0..N {
                assert_eq!(
                    ffi::vanedb_rs_hnsw_add(h, w.ids[i], w.vectors[i * DIM..].as_ptr()),
                    0
                );
            }
            ffi::vanedb_rs_hnsw_free(black_box(h));
        });
    });
    build.finish();

    // Pre-build once each for the search benchmark.
    let mut search = c.benchmark_group("hnsw_search");
    unsafe {
        let hc = ffi::vanedb_cpp_hnsw_new(DIM, 0, N, M, EFC, SEED);
        let hr = ffi::vanedb_rs_hnsw_new(DIM, 0, N, M, EFC, SEED);
        assert!(!hc.is_null() && !hr.is_null(), "hnsw_new failed");
        for i in 0..N {
            assert_eq!(
                ffi::vanedb_cpp_hnsw_add(hc, w.ids[i], w.vectors[i * DIM..].as_ptr()),
                0
            );
            assert_eq!(
                ffi::vanedb_rs_hnsw_add(hr, w.ids[i], w.vectors[i * DIM..].as_ptr()),
                0
            );
        }
        let mut ids = [0u64; 10];
        let mut ds = [0f32; 10];
        // Warmup outside the timed loops doubles as a liveness check.
        assert_eq!(
            ffi::vanedb_cpp_hnsw_search(hc, q.as_ptr(), 10, EFS, ids.as_mut_ptr(), ds.as_mut_ptr()),
            10
        );
        assert_eq!(
            ffi::vanedb_rs_hnsw_search(hr, q.as_ptr(), 10, EFS, ids.as_mut_ptr(), ds.as_mut_ptr()),
            10
        );
        search.bench_function("cpp", |bn| {
            bn.iter(|| {
                ffi::vanedb_cpp_hnsw_search(
                    hc,
                    black_box(q.as_ptr()),
                    10,
                    EFS,
                    ids.as_mut_ptr(),
                    ds.as_mut_ptr(),
                )
            })
        });
        search.bench_function("rs", |bn| {
            bn.iter(|| {
                ffi::vanedb_rs_hnsw_search(
                    hr,
                    black_box(q.as_ptr()),
                    10,
                    EFS,
                    ids.as_mut_ptr(),
                    ds.as_mut_ptr(),
                )
            })
        });
        ffi::vanedb_cpp_hnsw_free(hc);
        ffi::vanedb_rs_hnsw_free(hr);
    }
    search.finish();
}

criterion_group!(benches, bench_hnsw);
criterion_main!(benches);
