#![cfg(feature = "disk")]

use vanedb::{DiskIndex, DiskIndexBuilder, FlatIndex, Metric};

#[test]
fn mmap_matches_brute_force() {
    let dim = 16;
    let path = std::env::temp_dir().join(format!(
        "vanedb_test_mmap_vs_brute-{}.bin",
        std::process::id()
    ));

    let mut builder = DiskIndexBuilder::new(dim, Metric::L2).unwrap();
    let brute = FlatIndex::new(dim, Metric::L2).unwrap();

    for i in 0..100u64 {
        let v: Vec<f32> = (0..dim)
            .map(|d| ((i * 31 + d as u64 * 7) % 1000) as f32 / 100.0)
            .collect();
        builder.add(i, &v).unwrap();
        brute.add(i, &v).unwrap();
    }
    builder.save(&path).unwrap();

    // SAFETY: this test does not modify the file while it is mapped.
    let mmap = unsafe { DiskIndex::open(&path) }.unwrap();
    assert_eq!(mmap.size(), 100);

    for q in 0..5u64 {
        let query: Vec<f32> = (0..dim)
            .map(|d| ((q * 17 + d as u64 * 13) % 1000) as f32 / 100.0)
            .collect();

        let mmap_results = mmap.search(&query, 5).unwrap();
        let brute_results = brute.search(&query, 5).unwrap();

        assert_eq!(mmap_results.len(), brute_results.len());
        for (a, b) in mmap_results.iter().zip(brute_results.iter()) {
            assert_eq!(a.id, b.id, "query {q}: mmap vs brute mismatch");
        }
    }

    let _ = std::fs::remove_file(&path);
}

#[test]
fn mmap_cosine_search() {
    let path = std::env::temp_dir().join(format!(
        "vanedb_test_mmap_cosine-{}.bin",
        std::process::id()
    ));
    let mut builder = DiskIndexBuilder::new(3, Metric::Cosine).unwrap();
    builder.add(1, &[1.0, 0.0, 0.0]).unwrap();
    builder.add(2, &[0.0, 1.0, 0.0]).unwrap();
    builder.add(3, &[-1.0, 0.0, 0.0]).unwrap();
    builder.save(&path).unwrap();

    // SAFETY: this test does not modify the file while it is mapped.
    let store = unsafe { DiskIndex::open(&path) }.unwrap();
    let results = store.search(&[0.9, 0.1, 0.0], 1).unwrap();
    assert_eq!(results[0].id, 1);

    let _ = std::fs::remove_file(&path);
}

#[test]
fn mmap_concurrent_search() {
    use std::sync::Arc;
    use std::thread;

    let dim = 8;
    let path = std::env::temp_dir().join(format!(
        "vanedb_test_mmap_concurrent-{}.bin",
        std::process::id()
    ));

    let mut builder = DiskIndexBuilder::new(dim, Metric::L2).unwrap();
    for i in 0..50u64 {
        let v: Vec<f32> = (0..dim).map(|d| (i + d as u64) as f32).collect();
        builder.add(i, &v).unwrap();
    }
    builder.save(&path).unwrap();

    // SAFETY: this test does not modify the file while it is mapped.
    let store = Arc::new(unsafe { DiskIndex::open(&path) }.unwrap());

    let mut handles = vec![];
    for t in 0..10u64 {
        let store = Arc::clone(&store);
        handles.push(thread::spawn(move || {
            let query: Vec<f32> = (0..dim).map(|d| (t * 5 + d as u64) as f32).collect();
            let results = store.search(&query, 3).unwrap();
            assert_eq!(results.len(), 3);
        }));
    }
    for h in handles {
        h.join().unwrap();
    }

    let _ = std::fs::remove_file(&path);
}

#[test]
fn mmap_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<DiskIndex>();
}

#[test]
fn atomic_replacement_preserves_an_open_mapping() {
    let path = std::env::temp_dir().join(format!(
        "vanedb_atomic_replacement_{}.vndb",
        std::process::id()
    ));
    let mut builder = DiskIndexBuilder::new(2, Metric::L2).unwrap();
    builder.add(1, &[1.0, 2.0]).unwrap();
    builder.save(&path).unwrap();
    // SAFETY: save replaces the path; it never modifies the mapped file.
    let old = unsafe { DiskIndex::open(&path) }.unwrap();
    builder.add(2, &[3.0, 4.0]).unwrap();
    builder.save(&path).unwrap();
    // SAFETY: the replacement file remains unchanged until both indexes drop.
    let new = unsafe { DiskIndex::open(&path) }.unwrap();
    assert_eq!(old.len(), 1);
    assert_eq!(old.get(1).unwrap().as_ref(), [1.0, 2.0]);
    assert!(!old.contains(2));
    assert_eq!(new.len(), 2);
    assert_eq!(new.get(2).unwrap().as_ref(), [3.0, 4.0]);
    drop((old, new));
    std::fs::remove_file(path).unwrap();
}
