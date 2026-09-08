#![cfg(feature = "gpu-metal")]

use vanedb::distance::{self, Metric};
use vanedb::gpu::{GpuMetric, MetalCompute};
use vanedb::FlatIndex;

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

    let store = FlatIndex::new(dim, Metric::L2).unwrap();
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
        (GpuMetric::L2, Metric::L2),
        (GpuMetric::Cosine, Metric::Cosine),
        (GpuMetric::Dot, Metric::Dot),
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

/// GPU and scalar must agree across the tiny/normal matrix.
///
/// `MetalCompute::distances` gates on
/// `buffer.scalar_fallback || needs_scalar(query)`, and a QA sweep found that
/// changing that `||` to `&&` survives the suite. It called that the
/// safety-critical survivor: under `&&`, a buffer classified as needing the
/// scalar path still reaches the GPU when the query looks ordinary.
///
/// Measured rather than assumed, it is an equivalent mutant *on this
/// hardware*. With the mutation applied, every combination of tiny and normal
/// buffer against tiny and normal query, under both L2 and cosine, returns
/// bit-identical results to the scalar reference — delta 0.000e0 in all eight.
/// This Apple GPU does not flush the subnormal intermediates the guard exists
/// to avoid, so no fixture on this machine can distinguish the operators.
///
/// The guard is still correct: `needs_scalar`'s comment describes hardware
/// that does flush, and that hardware is what it protects against. What cannot
/// be claimed is that any test here proves it. This asserts the agreement that
/// *is* observable, which is what would break first on a device that flushes.
#[test]
fn gpu_and_scalar_agree_on_subnormal_and_ordinary_inputs() {
    let Ok(gpu) = MetalCompute::new() else {
        return; // no Metal device on this machine
    };
    let dim = 4;
    let n = 4;
    // Small enough that squares and differences of squares are subnormal.
    let tiny = f32::MIN_POSITIVE.sqrt() / f32::EPSILON / 8.0;

    let corpora = [
        (
            "tiny",
            (0..n * dim)
                .map(|i| tiny * ((i % 3) as f32 + 1.0))
                .collect::<Vec<f32>>(),
        ),
        (
            "ordinary",
            (0..n * dim)
                .map(|i| (i % 3) as f32 + 1.0)
                .collect::<Vec<f32>>(),
        ),
    ];
    let queries = [
        ("tiny", vec![tiny; dim]),
        (
            "ordinary",
            (0..dim).map(|d| d as f32 + 1.0).collect::<Vec<f32>>(),
        ),
    ];
    let metrics = [
        ("l2", GpuMetric::L2, Metric::L2),
        ("cosine", GpuMetric::Cosine, Metric::Cosine),
        ("dot", GpuMetric::Dot, Metric::Dot),
    ];

    for (corpus_name, vectors) in &corpora {
        let buffer = gpu.upload(vectors, n, dim).unwrap();
        for (query_name, query) in &queries {
            for (metric_name, gpu_metric, cpu_metric) in metrics {
                let got = gpu.distances(query, &buffer, gpu_metric).unwrap();
                let reference = distance::distance_fn(cpu_metric);
                for i in 0..n {
                    let expected = reference(query, &vectors[i * dim..(i + 1) * dim]);
                    let tolerance = expected.abs() * 1e-5 + 1e-30;
                    assert!(
                        (got[i] - expected).abs() <= tolerance,
                        "{corpus_name} corpus / {query_name} query / {metric_name}, \
                         slot {i}: GPU gave {}, scalar gives {expected}",
                        got[i]
                    );
                }
            }
        }
    }
}
