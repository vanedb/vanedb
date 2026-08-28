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
            bn.iter(|| unsafe {
                ffi::vanedb_cpp_l2_sq(black_box(a.as_ptr()), black_box(b.as_ptr()), d)
            });
        });
        g.bench_with_input(BenchmarkId::new("rs", dim), &dim, |bn, &d| {
            bn.iter(|| unsafe {
                ffi::vanedb_rs_l2_sq(black_box(a.as_ptr()), black_box(b.as_ptr()), d)
            });
        });
        g.finish();
    }
}

criterion_group!(benches, bench_distance);
criterion_main!(benches);
