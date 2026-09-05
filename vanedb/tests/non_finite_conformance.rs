use std::cmp::Ordering;

use vanedb::{Index, Metric, SearchResult, Store, VaneError};

#[cfg(feature = "disk")]
use vanedb::{DiskStore, DiskStoreBuilder};

fn cases() -> Vec<(&'static str, f32)> {
    include_str!("../../conformance/non_finite_vectors.tsv")
        .lines()
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(|line| {
            let (name, value) = line.split_once('\t').expect("valid TSV fixture");
            (name, value.parse::<f32>().expect("valid f32 fixture value"))
        })
        .collect()
}

fn assert_non_finite_error(result: vanedb::Result<()>, expected_input: &'static str) {
    assert!(matches!(
        result,
        Err(VaneError::NonFiniteValue { input }) if input == expected_input
    ));
}

#[test]
fn shared_cases_are_non_finite() {
    let cases = cases();
    assert_eq!(cases.len(), 3);
    for (name, value) in cases {
        assert!(!value.is_finite(), "{name} unexpectedly parsed as finite");
    }
}

#[test]
fn vector_store_rejects_non_finite_vectors_queries_and_batches() {
    for (name, value) in cases() {
        let store = Store::new(2, Metric::L2).unwrap();
        assert_non_finite_error(store.add(1, &[value, 0.0]), "vector");
        assert_eq!(store.len(), 0, "{name} add mutated the store");

        assert_non_finite_error(
            store.add_batch(&[1, 2], &[0.0, 0.0, value, 0.0]),
            "vector batch",
        );
        assert_eq!(store.len(), 0, "{name} batch mutated the store");

        store.add(2, &[0.0, 0.0]).unwrap();
        let error = store.search(&[value, 0.0], 1).unwrap_err();
        assert!(matches!(
            error,
            VaneError::NonFiniteValue { input: "query" }
        ));
    }
}

#[test]
fn hnsw_rejects_non_finite_vectors_queries_and_batches() {
    for (name, value) in cases() {
        let index = Index::builder(2, Metric::L2).capacity(4).build().unwrap();
        assert_non_finite_error(index.add(1, &[value, 0.0]), "vector");
        assert_eq!(index.size(), 0, "{name} add mutated the index");

        assert_non_finite_error(
            index.add_batch(&[1, 2], &[0.0, 0.0, value, 0.0]),
            "vector batch",
        );
        assert_eq!(index.size(), 0, "{name} batch mutated the index");

        index.add(2, &[0.0, 0.0]).unwrap();
        let error = index.search(&[value, 0.0], 1).unwrap_err();
        assert!(matches!(
            error,
            VaneError::NonFiniteValue { input: "query" }
        ));
    }
}

#[cfg(feature = "disk")]
#[test]
fn mmap_builder_rejects_non_finite_vectors() {
    for (name, value) in cases() {
        let mut builder = DiskStoreBuilder::new(2, Metric::L2).unwrap();
        assert_non_finite_error(builder.add(1, &[value, 0.0]), "vector");
        assert_eq!(builder.size(), 0, "{name} add mutated the builder");
    }
}

#[cfg(feature = "disk")]
#[test]
fn mmap_store_rejects_non_finite_queries() {
    let path = std::env::temp_dir().join("vanedb_non_finite_query_conformance.bin");
    let mut builder = DiskStoreBuilder::new(2, Metric::L2).unwrap();
    builder.add(1, &[0.0, 0.0]).unwrap();
    builder.save(&path).unwrap();
    let store = DiskStore::open(&path).unwrap();

    for (_, value) in cases() {
        let error = store.search(&[value, 0.0], 1).unwrap_err();
        assert!(matches!(
            error,
            VaneError::NonFiniteValue { input: "query" }
        ));
    }
    drop(store);
    let _ = std::fs::remove_file(path);
}

#[test]
fn finite_results_sort_before_non_finite_results() {
    for (_, value) in cases() {
        let finite = SearchResult::new(2, 0.0);
        let non_finite = SearchResult::new(1, value);
        assert_eq!(finite.cmp(&non_finite), Ordering::Less);
    }
}

#[test]
fn finite_exact_match_outranks_overflowed_distance() {
    let store = Store::new(2, Metric::L2).unwrap();
    store.add(1, &[f32::MAX, f32::MAX]).unwrap();
    store.add(2, &[0.0, 0.0]).unwrap();
    let results = store.search(&[0.0, 0.0], 2).unwrap();
    assert_eq!(results[0].id, 2);
    assert_eq!(results[0].distance, 0.0);
    assert!(!results[1].distance.is_finite());
}
