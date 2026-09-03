use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use std::hint::black_box;
use std::time::{Duration, Instant};
use vanedb_bench::coverage::groups;
use vanedb_bench::{ffi, workloads};

const DIM: usize = 128;

fn bench_store_add(c: &mut Criterion) {
    const N: usize = 10_000;
    let w = workloads::generate(5, DIM, N, 0);
    let mut g = c.benchmark_group(format!("{}/n={N}", groups::STORE_ADD));
    g.throughput(Throughput::Elements(N as u64));
    g.sample_size(10);

    // Neither C ABI exposes a reserve hook, so both engines start empty and
    // grow on demand: the preallocation policy is equal by construction.
    // store_new and store_free sit outside the measured interval — only the
    // adds are the throughput under test (#62 was this mistake in hnsw_build).
    g.bench_function("cpp", |bn| {
        bn.iter_custom(|iterations| {
            let mut elapsed = Duration::ZERO;
            for _ in 0..iterations {
                let s = unsafe { ffi::vanedb_cpp_store_new(DIM, 0) };
                assert!(!s.is_null(), "cpp store_new failed");
                let start = Instant::now();
                for i in 0..N {
                    assert_eq!(
                        unsafe {
                            ffi::vanedb_cpp_store_add(s, w.ids[i], w.vectors[i * DIM..].as_ptr())
                        },
                        0
                    );
                }
                elapsed += start.elapsed();
                unsafe { ffi::vanedb_cpp_store_free(black_box(s)) };
            }
            elapsed
        });
    });
    g.bench_function("rs", |bn| {
        bn.iter_custom(|iterations| {
            let mut elapsed = Duration::ZERO;
            for _ in 0..iterations {
                let s = unsafe { ffi::vanedb_rs_store_new(DIM, 0) };
                assert!(!s.is_null(), "rs store_new failed");
                let start = Instant::now();
                for i in 0..N {
                    assert_eq!(
                        unsafe {
                            ffi::vanedb_rs_store_add(s, w.ids[i], w.vectors[i * DIM..].as_ptr())
                        },
                        0
                    );
                }
                elapsed += start.elapsed();
                unsafe { ffi::vanedb_rs_store_free(black_box(s)) };
            }
            elapsed
        });
    });
    g.finish();
}

fn bench_store_search(c: &mut Criterion) {
    for &n in &[1_000usize, 10_000] {
        let w = workloads::generate(2, DIM, n, 1);
        let q = &w.queries[0..DIM];
        let mut g = c.benchmark_group(format!("{}/n={n}", groups::STORE_SEARCH));

        unsafe {
            // Measurement policy: both engines' stores stay resident for the
            // whole group, built interleaved — matching hnsw.rs and the report
            // bin. A brute-force scan is cache-bound, so residency decides the
            // winner; every path must use the same policy.
            let sc = ffi::vanedb_cpp_store_new(DIM, 0);
            let sr = ffi::vanedb_rs_store_new(DIM, 0);
            assert!(!sc.is_null() && !sr.is_null(), "store_new failed");
            for i in 0..n {
                let v = w.vectors[i * DIM..].as_ptr();
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

criterion_group!(benches, bench_store_add, bench_store_search);
criterion_main!(benches);
