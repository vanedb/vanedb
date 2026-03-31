use vanedb::{DistanceMetric, HnswIndex, VectorStore};

/// Brute-force search using VectorStore as ground truth, then check HNSW recall.
#[test]
fn hnsw_recall_vs_brute_force() {
    let dim = 32;
    let n = 500;
    let k = 10;

    let hnsw = HnswIndex::builder(dim, DistanceMetric::L2)
        .capacity(n)
        .m(16)
        .ef_construction(200)
        .seed(42)
        .build()
        .unwrap();

    let brute = VectorStore::new(dim, DistanceMetric::L2).unwrap();

    // Generate deterministic vectors
    for i in 0..n as u64 {
        let v: Vec<f32> = (0..dim)
            .map(|d| ((i * 31 + d as u64 * 7) % 1000) as f32 / 100.0)
            .collect();
        hnsw.add(i, &v).unwrap();
        brute.add(i, &v).unwrap();
    }

    hnsw.set_ef_search(100);

    // Test 10 queries
    let mut total_recall = 0.0;
    for q in 0..10u64 {
        let query: Vec<f32> = (0..dim)
            .map(|d| ((q * 17 + d as u64 * 13) % 1000) as f32 / 100.0)
            .collect();

        let hnsw_results = hnsw.search(&query, k).unwrap();
        let brute_results = brute.search(&query, k).unwrap();

        let brute_ids: std::collections::HashSet<u64> =
            brute_results.iter().map(|r| r.id).collect();
        let hits = hnsw_results
            .iter()
            .filter(|r| brute_ids.contains(&r.id))
            .count();

        total_recall += hits as f64 / k as f64;
    }

    let avg_recall = total_recall / 10.0;
    assert!(
        avg_recall >= 0.8,
        "HNSW recall too low: {avg_recall:.2} (expected >= 0.80)"
    );
}

#[test]
fn hnsw_cosine_search() {
    let idx = HnswIndex::builder(3, DistanceMetric::Cosine)
        .capacity(100)
        .seed(42)
        .build()
        .unwrap();

    idx.add(1, &[1.0, 0.0, 0.0]).unwrap(); // right
    idx.add(2, &[0.0, 1.0, 0.0]).unwrap(); // up
    idx.add(3, &[-1.0, 0.0, 0.0]).unwrap(); // left

    let results = idx.search(&[0.9, 0.1, 0.0], 1).unwrap();
    assert_eq!(results[0].id, 1);
}

#[test]
fn hnsw_save_load_roundtrip() {
    let dim = 8;
    let path = std::env::temp_dir().join("vanedb_test_hnsw.bin");

    // Build and populate index
    let idx = HnswIndex::builder(dim, DistanceMetric::L2)
        .capacity(100)
        .seed(42)
        .build()
        .unwrap();
    for i in 0..20u64 {
        let v: Vec<f32> = (0..dim).map(|d| (i * 10 + d as u64) as f32).collect();
        idx.add(i, &v).unwrap();
    }
    idx.set_ef_search(100);

    // Save
    idx.save(&path).unwrap();

    // Load
    let loaded = HnswIndex::load(&path).unwrap();

    // Verify metadata
    assert_eq!(loaded.dimension(), dim);
    assert_eq!(loaded.size(), 20);
    assert_eq!(loaded.get_ef_search(), 100);

    // Verify vectors
    for i in 0..20u64 {
        assert_eq!(idx.get_vector(i).unwrap(), loaded.get_vector(i).unwrap());
    }

    // Verify search produces same results
    let query = vec![5.0; dim];
    let orig_results = idx.search(&query, 5).unwrap();
    let load_results = loaded.search(&query, 5).unwrap();
    assert_eq!(orig_results.len(), load_results.len());
    for (a, b) in orig_results.iter().zip(load_results.iter()) {
        assert_eq!(a.id, b.id);
    }

    // Cleanup
    let _ = std::fs::remove_file(&path);
}
