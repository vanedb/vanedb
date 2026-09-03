use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;
use vanedb_bench::coverage::groups;
use vanedb_bench::{ffi, workloads};

/// Generic over the two calls rather than taking function pointers: a pointer
/// table would make every metric an indirect call and shift the numbers
/// against the published snapshot. Monomorphised closures keep the direct call
/// the l2_sq figures were measured with.
fn bench_metric(
    c: &mut Criterion,
    name: &str,
    cpp: impl Fn(*const f32, *const f32, usize) -> f32,
    rs: impl Fn(*const f32, *const f32, usize) -> f32,
) {
    for &dim in &[128usize, 768] {
        let w = workloads::generate(1, dim, 2, 0);
        let a = &w.vectors[0..dim];
        let b = &w.vectors[dim..2 * dim];
        let mut g = c.benchmark_group(format!("{name}/dim={dim}"));
        g.bench_with_input(BenchmarkId::new("cpp", dim), &dim, |bn, &d| {
            bn.iter(|| cpp(black_box(a.as_ptr()), black_box(b.as_ptr()), d));
        });
        g.bench_with_input(BenchmarkId::new("rs", dim), &dim, |bn, &d| {
            bn.iter(|| rs(black_box(a.as_ptr()), black_box(b.as_ptr()), d));
        });
        g.finish();
    }
}

fn bench_distance(c: &mut Criterion) {
    bench_metric(
        c,
        groups::L2_SQ,
        |a, b, d| unsafe { ffi::vanedb_cpp_l2_sq(a, b, d) },
        |a, b, d| unsafe { ffi::vanedb_rs_l2_sq(a, b, d) },
    );
    bench_metric(
        c,
        groups::COSINE,
        |a, b, d| unsafe { ffi::vanedb_cpp_cosine_distance(a, b, d) },
        |a, b, d| unsafe { ffi::vanedb_rs_cosine_distance(a, b, d) },
    );
    bench_metric(
        c,
        groups::DOT,
        |a, b, d| unsafe { ffi::vanedb_cpp_dot_product(a, b, d) },
        |a, b, d| unsafe { ffi::vanedb_rs_dot_product(a, b, d) },
    );
}

criterion_group!(benches, bench_distance);
criterion_main!(benches);
