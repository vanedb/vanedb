#![cfg(feature = "gpu-metal")]

use vanedb::distance::{self, DistanceMetric};
use vanedb::gpu::{GpuMetric, MetalCompute};
use vanedb::VectorStore;

#[test]
fn gpu_search_matches_brute_force() {
    let gpu = MetalCompute::new().unwrap();
    let dim = 128;
    let n = 500;
    let k = 10;

    let flat: Vec<f32> = (0..n * dim)
        .map(|i| ((i * 31 + 7) % 1000) as f32 / 100.0)
        .collect();
    let ids: Vec<u64> = (0..n as u64).collect();

    let store = VectorStore::new(dim, DistanceMetric::L2).unwrap();
    for i in 0..n {
        store.add(i as u64, &flat[i * dim..(i + 1) * dim]).unwrap();
    }

    let buf = gpu.upload(&flat, n, dim).unwrap();
    let query: Vec<f32> = (0..dim).map(|d| (d * 13 % 1000) as f32 / 100.0).collect();

    let cpu_results = store.search(&query, k).unwrap();
    let gpu_results = gpu.search(&query, &ids, &buf, k, GpuMetric::L2).unwrap();

    assert_eq!(cpu_results.len(), gpu_results.len());
    for (cpu, gpu_r) in cpu_results.iter().zip(gpu_results.iter()) {
        assert_eq!(cpu.id, gpu_r.id, "CPU and GPU disagree on nearest neighbor");
    }
}

#[test]
fn gpu_all_metrics() {
    let gpu = MetalCompute::new().unwrap();
    let dim = 128;
    let n = 100;
    let flat: Vec<f32> = (0..n * dim)
        .map(|i| (i as f32 * 0.01).sin() + 0.1)
        .collect();
    let query: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.02).cos() + 0.1).collect();
    let buf = gpu.upload(&flat, n, dim).unwrap();

    for (gpu_metric, cpu_metric) in [
        (GpuMetric::L2, DistanceMetric::L2),
        (GpuMetric::Cosine, DistanceMetric::Cosine),
        (GpuMetric::Dot, DistanceMetric::Dot),
    ] {
        let gpu_dists = gpu.distances(&query, &buf, gpu_metric).unwrap();
        let cpu_fn = distance::distance_fn(cpu_metric);
        for i in 0..n {
            let cpu_d = cpu_fn(&query, &flat[i * dim..(i + 1) * dim]);
            assert!(
                (gpu_dists[i] - cpu_d).abs() < 1e-2,
                "{gpu_metric:?} vector {i}: gpu={} cpu={cpu_d}",
                gpu_dists[i]
            );
        }
    }
}
