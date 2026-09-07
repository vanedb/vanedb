use vanedb_wasm::*;
use wasm_bindgen_test::*;

#[wasm_bindgen_test]
fn test_version() {
    // Compared against the same source as the binding, so the two cannot
    // drift apart the way a pair of literals did.
    assert_eq!(version(), env!("CARGO_PKG_VERSION"));
}

#[wasm_bindgen_test]
fn test_vector_store_basic() {
    let store = WasmStore::new(3.0, "l2").unwrap();
    store.add(1u64.into(), &[1.0, 0.0, 0.0]).unwrap();
    store.add(2u64.into(), &[0.0, 1.0, 0.0]).unwrap();
    assert_eq!(store.size(), 2);
    assert_eq!(store.dimension(), 3);
    assert!(store.contains(1u64.into()).unwrap());
    assert!(!store.contains(99u64.into()).unwrap());
}

#[wasm_bindgen_test]
fn test_vector_store_search() {
    let store = WasmStore::new(2.0, "l2").unwrap();
    store.add(1u64.into(), &[0.0, 0.0]).unwrap();
    store.add(2u64.into(), &[1.0, 0.0]).unwrap();
    store.add(3u64.into(), &[10.0, 10.0]).unwrap();

    let hits = store.search(&[0.0, 0.1], 2.0).unwrap();
    assert_eq!(hits.length(), 2);
    assert_eq!(hits.distances().len(), 2);
    assert_eq!(hits.ids()[0], 1); // closest
}

#[wasm_bindgen_test]
fn test_hnsw_basic() {
    let idx = WasmIndex::new(3.0, "l2", 100.0, 16.0, 200.0).unwrap();
    idx.add(1u64.into(), &[1.0, 0.0, 0.0]).unwrap();
    idx.add(2u64.into(), &[0.0, 1.0, 0.0]).unwrap();
    assert_eq!(idx.size(), 2);
    assert!(idx.contains(1u64.into()).unwrap());
}

#[wasm_bindgen_test]
fn test_hnsw_search() {
    let idx = WasmIndex::new(3.0, "l2", 100.0, 16.0, 200.0).unwrap();
    idx.add(1u64.into(), &[0.0, 0.0, 0.0]).unwrap();
    idx.add(2u64.into(), &[10.0, 10.0, 10.0]).unwrap();

    let hits = idx.search(&[0.0, 0.0, 0.0], 1.0).unwrap();
    assert_eq!(hits.ids()[0], 1);
}

#[wasm_bindgen_test]
fn test_cosine_metric() {
    let store = WasmStore::new(2.0, "cosine").unwrap();
    store.add(1u64.into(), &[1.0, 0.0]).unwrap();
    store.add(2u64.into(), &[0.0, 1.0]).unwrap();
    let hits = store.search(&[0.9, 0.1], 1.0).unwrap();
    assert_eq!(hits.ids()[0], 1);
}

#[wasm_bindgen_test]
fn test_invalid_metric() {
    let result = WasmStore::new(3.0, "invalid");
    assert!(result.is_err());
}

#[wasm_bindgen_test]
fn test_store_add_batch() {
    let store = WasmStore::new(2.0, "l2").unwrap();
    let ids = [1u64, 2, 3];
    let flat = [0.0f32, 0.0, 1.0, 1.0, 5.0, 5.0];
    store.add_batch(&ids, &flat).unwrap();
    assert_eq!(store.size(), 3);
    let results = store.search(&[0.9, 0.9], 1.0).unwrap();
    assert_eq!(results.ids()[0], 2);

    // duplicate -> Err, all-or-nothing
    assert!(store.add_batch(&[4, 1], &flat[..4]).is_err());
    assert_eq!(store.size(), 3);
    assert!(!store.contains(4u64.into()).unwrap());
}

#[wasm_bindgen_test]
fn test_hnsw_add_batch() {
    let index = WasmIndex::new(2.0, "l2", 100.0, 16.0, 200.0).unwrap();
    let ids = [10u64, 20];
    let flat = [0.0f32, 0.0, 1.0, 1.0];
    index.add_batch(&ids, &flat).unwrap();
    assert_eq!(index.size(), 2);
    let results = index.search(&[0.1, 0.1], 1.0).unwrap();
    assert_eq!(results.ids()[0], 10);
}

// --- #39: ids must survive the JS boundary without f32 narrowing ---

const PRECISION_IDS: [u64; 4] = [1 << 24, (1 << 24) + 1, 1 << 53, u64::MAX];

#[wasm_bindgen_test]
fn store_search_round_trips_ids_beyond_f32_precision() {
    let store = WasmStore::new(2.0, "l2").unwrap();
    for (i, id) in PRECISION_IDS.iter().enumerate() {
        store.add((*id).into(), &[i as f32, 0.0]).unwrap();
    }
    let mut got = store.search(&[0.0, 0.0], 4.0).unwrap().ids();
    got.sort_unstable();
    let mut want = PRECISION_IDS.to_vec();
    want.sort_unstable();
    assert_eq!(got, want);
}

#[wasm_bindgen_test]
fn hnsw_search_round_trips_ids_beyond_f32_precision() {
    let index = WasmIndex::new(2.0, "l2", 16.0, 16.0, 100.0).unwrap();
    for (i, id) in PRECISION_IDS.iter().enumerate() {
        index.add((*id).into(), &[i as f32, 0.0]).unwrap();
    }
    let mut got = index.search(&[0.0, 0.0], 4.0).unwrap().ids();
    got.sort_unstable();
    let mut want = PRECISION_IDS.to_vec();
    want.sort_unstable();
    assert_eq!(got, want);
}

#[wasm_bindgen_test]
fn test_non_finite_vectors_and_queries_are_rejected() {
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let store = WasmStore::new(2.0, "l2").unwrap();
        assert!(store.add(1u64.into(), &[value, 0.0]).is_err());
        assert_eq!(store.size(), 0);
        store.add(2u64.into(), &[0.0, 0.0]).unwrap();
        assert!(store.search(&[value, 0.0], 1.0).is_err());

        let index = WasmIndex::new(2.0, "l2", 4.0, 2.0, 10.0).unwrap();
        assert!(index.add(1u64.into(), &[value, 0.0]).is_err());
        assert_eq!(index.size(), 0);
    }
}

#[wasm_bindgen_test]
fn every_metric_round_trips_and_is_reportable() {
    // Dot had no coverage here, and neither class could report the metric it
    // was built with. The reported spelling is the one the constructor takes,
    // so it can be fed straight back.
    for name in ["l2", "cosine", "dot"] {
        let store = WasmStore::new(2.0, name).unwrap();
        assert_eq!(store.metric(), name);
        let index = WasmIndex::new(2.0, name, 16.0, 4.0, 40.0).unwrap();
        assert_eq!(index.metric(), name);
        // Round-trips through the constructor it names.
        assert!(WasmStore::new(2.0, &store.metric()).is_ok());
    }
}

#[wasm_bindgen_test]
fn dot_ranks_by_largest_inner_product() {
    let store = WasmStore::new(2.0, "dot").unwrap();
    store.add(1u64.into(), &[1.0, 0.0]).unwrap();
    store.add(2u64.into(), &[4.0, 0.0]).unwrap();
    store.add(3u64.into(), &[0.0, 1.0]).unwrap();
    let hits = store.search(&[1.0, 0.0], 3.0).unwrap();
    let ids = hits.ids();
    assert_eq!(ids[0], 2, "dot must rank the largest inner product first");
    assert_eq!(ids[2], 3);
}
