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
fn hnsw_save_size_proportional_to_count_not_capacity() {
    // Issue #18: a capacity-1000 index holding 10 vectors must not write
    // ~capacity worth of data. 10 vectors x 32 dims x 4 bytes is ~1.3 KB of
    // payload; 20 KB allows generous encoding overhead, while the full
    // pre-allocated arrays would exceed 140 KB.
    let path = std::env::temp_dir().join("vanedb_test_hnsw_compact.bin");
    let idx = HnswIndex::builder(32, DistanceMetric::L2)
        .capacity(1000)
        .seed(42)
        .build()
        .unwrap();
    for i in 0..10u64 {
        let v: Vec<f32> = (0..32).map(|d| (i * 32 + d) as f32).collect();
        idx.add(i, &v).unwrap();
    }
    idx.save(&path).unwrap();
    let size = std::fs::metadata(&path).unwrap().len();
    let _ = std::fs::remove_file(&path);
    assert!(
        size < 20_000,
        "saved file is {size} bytes — it scales with capacity, not inserted count"
    );
}

#[test]
fn hnsw_empty_index_save_load_roundtrip() {
    // v2 stores zero-length arrays for an empty index; load must re-expand
    // to full capacity so subsequent adds work.
    let path = std::env::temp_dir().join("vanedb_test_hnsw_empty.bin");
    let idx = HnswIndex::builder(4, DistanceMetric::L2)
        .capacity(10)
        .seed(42)
        .build()
        .unwrap();
    idx.save(&path).unwrap();
    let loaded = HnswIndex::load(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    assert_eq!(loaded.size(), 0);
    assert_eq!(loaded.capacity(), 10);
    assert!(loaded.search(&[0.0; 4], 3).unwrap().is_empty());
    loaded.add(1, &[1.0, 2.0, 3.0, 4.0]).unwrap();
    assert_eq!(loaded.size(), 1);
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

#[test]
fn hnsw_concurrent_search() {
    use std::sync::Arc;
    use std::thread;

    let idx = Arc::new(
        HnswIndex::builder(8, DistanceMetric::L2)
            .capacity(200)
            .seed(42)
            .build()
            .unwrap(),
    );

    // Add vectors sequentially
    for i in 0..100u64 {
        let v: Vec<f32> = (0..8).map(|d| (i + d as u64) as f32).collect();
        idx.add(i, &v).unwrap();
    }

    // 10 concurrent search threads
    let mut handles = vec![];
    for t in 0..10u64 {
        let idx = Arc::clone(&idx);
        handles.push(thread::spawn(move || {
            let query: Vec<f32> = (0..8).map(|d| (t * 10 + d as u64) as f32).collect();
            let results = idx.search(&query, 5).unwrap();
            assert_eq!(results.len(), 5);
        }));
    }

    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn hnsw_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<HnswIndex>();
}

/// Drive the thread-local visited bitmap through at least one u16 epoch wrap
/// (~65k searches) and verify the zero-out fallback path still produces
/// correct nearest-neighbor results. Mirrors the wrap test from
/// vanedb-cpp PR #8 ("search_layer epoch wrap").
#[test]
fn hnsw_visited_bitmap_epoch_wrap_correctness() {
    let dim = 4;
    let n = 100usize;
    let idx = HnswIndex::builder(dim, DistanceMetric::L2)
        .capacity(n)
        .seed(42)
        .build()
        .unwrap();
    for i in 0..n as u64 {
        let v: Vec<f32> = (0..dim).map(|d| (i + d as u64) as f32).collect();
        idx.add(i, &v).unwrap();
    }

    // First search establishes a baseline before any wrap.
    let query: Vec<f32> = vec![5.0, 5.0, 5.0, 5.0];
    let baseline = idx.search(&query, 5).unwrap();

    // Drive the per-thread epoch counter past the u16 boundary. 70_000 searches
    // wraps once and exercises the zero-out fallback inside `VisitedBuffer::begin`.
    for _ in 0..70_000 {
        let _ = idx.search(&query, 5).unwrap();
    }

    let after_wrap = idx.search(&query, 5).unwrap();
    assert_eq!(baseline.len(), after_wrap.len());
    for (a, b) in baseline.iter().zip(after_wrap.iter()) {
        assert_eq!(a.id, b.id, "result drifted across epoch wrap");
    }
}
