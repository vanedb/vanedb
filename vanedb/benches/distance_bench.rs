use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use vanedb::distance::{self, DistanceMetric};

fn gen_vectors(dim: usize) -> (Vec<f32>, Vec<f32>) {
    let a: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.01).sin()).collect();
    let b: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.02).cos()).collect();
    (a, b)
}

fn bench_distance(c: &mut Criterion) {
    let metrics = [
        ("L2", DistanceMetric::L2),
        ("Cosine", DistanceMetric::Cosine),
        ("Dot", DistanceMetric::Dot),
    ];
    let dims = [128, 768, 1536];

    for (name, metric) in &metrics {
        let mut group = c.benchmark_group(*name);
        let dist_fn = distance::distance_fn(*metric);
        for &dim in &dims {
            let (a, b) = gen_vectors(dim);
            group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |bench, _| {
                bench.iter(|| dist_fn(black_box(&a), black_box(&b)));
            });
        }
        group.finish();
    }
}

criterion_group!(benches, bench_distance);
criterion_main!(benches);
