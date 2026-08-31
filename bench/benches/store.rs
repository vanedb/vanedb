use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;
use vanedb_bench::{ffi, workloads};

fn bench_store_search(c: &mut Criterion) {
    let dim = 128usize;
    for &n in &[1_000usize, 10_000] {
        let w = workloads::generate(2, dim, n, 1);
        let q = &w.queries[0..dim];
        let mut g = c.benchmark_group(format!("store_search/n={n}"));

        unsafe {
            // Measurement policy: both engines' stores stay resident for the
            // whole group, built interleaved — matching hnsw.rs and the report
            // bin. A brute-force scan is cache-bound, so residency decides the
            // winner; every path must use the same policy.
            let sc = ffi::vanedb_cpp_store_new(dim, 0);
            let sr = ffi::vanedb_rs_store_new(dim, 0);
            assert!(!sc.is_null() && !sr.is_null(), "store_new failed");
            for i in 0..n {
                let v = w.vectors[i * dim..].as_ptr();
                assert_eq!(ffi::vanedb_cpp_store_add(sc, w.ids[i], v), 0);
                assert_eq!(ffi::vanedb_rs_store_add(sr, w.ids[i], v), 0);
            }
            let mut ids = [0u64; 10];
            let mut ds = [0f32; 10];
            // Warmup outside the timed loops doubles as a liveness check: a
            // failed engine would otherwise "win" by returning instantly.
            assert_eq!(
                ffi::vanedb_cpp_store_search(sc, q.as_ptr(), 10, ids.as_mut_ptr(), ds.as_mut_ptr()),
                10
            );
            assert_eq!(
                ffi::vanedb_rs_store_search(sr, q.as_ptr(), 10, ids.as_mut_ptr(), ds.as_mut_ptr()),
                10
            );
            g.bench_function("cpp", |bn| {
                bn.iter(|| {
                    ffi::vanedb_cpp_store_search(
                        sc,
                        black_box(q.as_ptr()),
                        10,
                        ids.as_mut_ptr(),
                        ds.as_mut_ptr(),
                    )
                })
            });
            g.bench_function("rs", |bn| {
                bn.iter(|| {
                    ffi::vanedb_rs_store_search(
                        sr,
                        black_box(q.as_ptr()),
                        10,
                        ids.as_mut_ptr(),
                        ds.as_mut_ptr(),
                    )
                })
            });
            ffi::vanedb_cpp_store_free(sc);
            ffi::vanedb_rs_store_free(sr);
        }
        g.finish();
    }
}

criterion_group!(benches, bench_store_search);
criterion_main!(benches);
