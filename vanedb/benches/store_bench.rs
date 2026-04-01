use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use vanedb::{DistanceMetric, VectorStore};

fn gen_data(n: usize, dim: usize) -> Vec<Vec<f32>> {
    (0..n)
        .map(|i| {
            (0..dim)
                .map(|d| ((i * 31 + d * 7) % 1000) as f32 / 100.0)
                .collect()
        })
        .collect()
}

fn bench_store_search(c: &mut Criterion) {
    let dim = 128;
    let mut group = c.benchmark_group("VectorStore_search");

    for &n in &[1_000, 10_000] {
        let store = VectorStore::new(dim, DistanceMetric::L2).unwrap();
        let data = gen_data(n, dim);
        for (i, v) in data.iter().enumerate() {
            store.add(i as u64, v).unwrap();
        }
        let query: Vec<f32> = (0..dim).map(|d| (d * 13 % 1000) as f32 / 100.0).collect();

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| store.search(&query, 10).unwrap());
        });
    }
    group.finish();
}

fn bench_store_add(c: &mut Criterion) {
    let dim = 128;
    let data = gen_data(10_000, dim);

    c.bench_function("VectorStore_add_10k", |bench| {
        bench.iter(|| {
            let store = VectorStore::new(dim, DistanceMetric::L2).unwrap();
            for (i, v) in data.iter().enumerate() {
                store.add(i as u64, v).unwrap();
            }
        });
    });
}

criterion_group!(benches, bench_store_search, bench_store_add);
criterion_main!(benches);
