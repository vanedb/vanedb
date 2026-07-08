//! Profiling target: builds a 10k x 128d HNSW index repeatedly so an external
//! sampler (macOS `sample`, samply, perf) can attribute build-phase hot spots.
//! Temporary tool for investigating the 2.5x build gap vs vanedb-cpp
//! (vanedb-bench#1); not part of the public API surface.

use vanedb::{DistanceMetric, HnswIndex};

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

fn main() {
    const DIM: usize = 128;
    const N: usize = 10_000;
    let mut s = 3u64;
    let vectors: Vec<f32> = (0..N * DIM)
        .map(|_| (splitmix64(&mut s) >> 40) as f32 / (1u64 << 24) as f32)
        .collect();

    let rounds: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(10);

    let mut last = None;
    for round in 0..rounds {
        let start = std::time::Instant::now();
        let idx = HnswIndex::builder(DIM, DistanceMetric::L2)
            .capacity(N)
            .m(16)
            .ef_construction(200)
            .seed(7)
            .build()
            .unwrap();
        for i in 0..N {
            idx.add(i as u64, &vectors[i * DIM..(i + 1) * DIM]).unwrap();
        }
        println!("round {round}: {:?} (size {})", start.elapsed(), idx.size());
        last = Some(idx);
    }

    // recall@10 over 100 queries vs brute force
    let ef: usize = std::env::args()
        .nth(2)
        .and_then(|a| a.parse().ok())
        .unwrap_or(50);
    let idx = last.unwrap();
    idx.set_ef_search(ef);
    let brute = vanedb::VectorStore::new(DIM, DistanceMetric::L2).unwrap();
    for i in 0..N {
        brute
            .add(i as u64, &vectors[i * DIM..(i + 1) * DIM])
            .unwrap();
    }
    let mut recall = 0.0f64;
    for _ in 0..100 {
        let q: Vec<f32> = (0..DIM)
            .map(|_| (splitmix64(&mut s) >> 40) as f32 / (1u64 << 24) as f32)
            .collect();
        let truth: std::collections::HashSet<u64> =
            brute.search(&q, 10).unwrap().iter().map(|r| r.id).collect();
        let got = idx.search(&q, 10).unwrap();
        recall += got.iter().filter(|r| truth.contains(&r.id)).count() as f64 / 10.0;
    }
    println!("recall@10 (100 queries, ef={ef}): {:.3}", recall / 100.0);
}
