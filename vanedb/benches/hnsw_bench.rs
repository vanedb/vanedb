use criterion::{criterion_group, criterion_main, Criterion};
use vanedb::{DistanceMetric, HnswIndex};

fn gen_data(n: usize, dim: usize) -> Vec<Vec<f32>> {
    (0..n)
        .map(|i| {
            (0..dim)
                .map(|d| ((i * 31 + d * 7) % 1000) as f32 / 100.0)
                .collect()
        })
        .collect()
}

fn bench_hnsw_build(c: &mut Criterion) {
    let dim = 128;
    let n = 10_000;
    let data = gen_data(n, dim);

    c.bench_function("HNSW_build_10k", |bench| {
        bench.iter(|| {
            let idx = HnswIndex::builder(dim, DistanceMetric::L2)
                .capacity(n)
                .m(16)
                .ef_construction(200)
                .seed(42)
                .build()
                .unwrap();
            for (i, v) in data.iter().enumerate() {
                idx.add(i as u64, v).unwrap();
            }
        });
    });
}

fn bench_hnsw_search(c: &mut Criterion) {
    let dim = 128;
    let n = 10_000;
    let data = gen_data(n, dim);

    let idx = HnswIndex::builder(dim, DistanceMetric::L2)
        .capacity(n)
        .m(16)
        .ef_construction(200)
        .seed(42)
        .build()
        .unwrap();
    for (i, v) in data.iter().enumerate() {
        idx.add(i as u64, v).unwrap();
    }
    idx.set_ef_search(50);

    let query: Vec<f32> = (0..dim).map(|d| (d * 13 % 1000) as f32 / 100.0).collect();

    c.bench_function("HNSW_search_10k", |bench| {
        bench.iter(|| idx.search(&query, 10).unwrap());
    });
}

fn bench_hnsw_save_load(c: &mut Criterion) {
    let dim = 128;
    let n = 10_000;
    let data = gen_data(n, dim);
    let path = std::env::temp_dir().join("vanedb_bench_hnsw.bin");

    let idx = HnswIndex::builder(dim, DistanceMetric::L2)
        .capacity(n)
        .m(16)
        .ef_construction(200)
        .seed(42)
        .build()
        .unwrap();
    for (i, v) in data.iter().enumerate() {
        idx.add(i as u64, v).unwrap();
    }

    c.bench_function("HNSW_save_10k", |bench| {
        bench.iter(|| idx.save(&path).unwrap());
    });

    c.bench_function("HNSW_load_10k", |bench| {
        bench.iter(|| HnswIndex::load(&path).unwrap());
    });

    let _ = std::fs::remove_file(&path);
}

criterion_group!(
    benches,
    bench_hnsw_build,
    bench_hnsw_search,
    bench_hnsw_save_load
);
criterion_main!(benches);
