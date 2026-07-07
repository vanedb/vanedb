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
            for i in 0..n {
                ffi::vanedb_cpp_store_add(s, w.ids[i], w.vectors[i * dim..].as_ptr());
            }
            let mut ids = [0u64; 10];
            let mut ds = [0f32; 10];
            bn.iter(|| {
                ffi::vanedb_cpp_store_search(
                    s,
                    black_box(q.as_ptr()),
                    10,
                    ids.as_mut_ptr(),
                    ds.as_mut_ptr(),
                )
            });
            ffi::vanedb_cpp_store_free(s);
        });
        g.bench_function("rs", |bn| unsafe {
            let s = ffi::vanedb_rs_store_new(dim, 0);
            for i in 0..n {
                ffi::vanedb_rs_store_add(s, w.ids[i], w.vectors[i * dim..].as_ptr());
            }
            let mut ids = [0u64; 10];
            let mut ds = [0f32; 10];
            bn.iter(|| {
                ffi::vanedb_rs_store_search(
                    s,
                    black_box(q.as_ptr()),
                    10,
                    ids.as_mut_ptr(),
                    ds.as_mut_ptr(),
                )
            });
            ffi::vanedb_rs_store_free(s);
        });
        g.finish();
    }
}

criterion_group!(benches, bench_store_search);
criterion_main!(benches);
