use vanedb_wasm::*;
use wasm_bindgen_test::*;

#[wasm_bindgen_test]
fn test_version() {
    assert_eq!(version(), "0.1.0");
}

#[wasm_bindgen_test]
fn test_vector_store_basic() {
    let store = WasmVectorStore::new(3, "l2").unwrap();
    store.add(1, &[1.0, 0.0, 0.0]).unwrap();
    store.add(2, &[0.0, 1.0, 0.0]).unwrap();
    assert_eq!(store.size(), 2);
    assert_eq!(store.dimension(), 3);
    assert!(store.contains(1));
    assert!(!store.contains(99));
}

#[wasm_bindgen_test]
fn test_vector_store_search() {
    let store = WasmVectorStore::new(2, "l2").unwrap();
    store.add(1, &[0.0, 0.0]).unwrap();
    store.add(2, &[1.0, 0.0]).unwrap();
    store.add(3, &[10.0, 10.0]).unwrap();

    let hits = store.search(&[0.0, 0.1], 2).unwrap();
    assert_eq!(hits.length(), 2);
    assert_eq!(hits.distances().len(), 2);
    assert_eq!(hits.ids()[0], 1); // closest
}

#[wasm_bindgen_test]
fn test_hnsw_basic() {
    let idx = WasmHnswIndex::new(3, "l2", 100, 16, 200).unwrap();
    idx.add(1, &[1.0, 0.0, 0.0]).unwrap();
    idx.add(2, &[0.0, 1.0, 0.0]).unwrap();
    assert_eq!(idx.size(), 2);
    assert!(idx.contains(1));
}

#[wasm_bindgen_test]
fn test_hnsw_search() {
    let idx = WasmHnswIndex::new(3, "l2", 100, 16, 200).unwrap();
    idx.add(1, &[0.0, 0.0, 0.0]).unwrap();
    idx.add(2, &[10.0, 10.0, 10.0]).unwrap();

    let hits = idx.search(&[0.0, 0.0, 0.0], 1).unwrap();
    assert_eq!(hits.ids()[0], 1);
}

#[wasm_bindgen_test]
fn test_cosine_metric() {
    let store = WasmVectorStore::new(2, "cosine").unwrap();
    store.add(1, &[1.0, 0.0]).unwrap();
    store.add(2, &[0.0, 1.0]).unwrap();
    let hits = store.search(&[0.9, 0.1], 1).unwrap();
    assert_eq!(hits.ids()[0], 1);
}

#[wasm_bindgen_test]
fn test_invalid_metric() {
    let result = WasmVectorStore::new(3, "invalid");
    assert!(result.is_err());
}

#[wasm_bindgen_test]
fn test_store_add_batch() {
    let store = WasmVectorStore::new(2, "l2").unwrap();
    let ids = [1u64, 2, 3];
    let flat = [0.0f32, 0.0, 1.0, 1.0, 5.0, 5.0];
    store.add_batch(&ids, &flat).unwrap();
    assert_eq!(store.size(), 3);
    let results = store.search(&[0.9, 0.9], 1).unwrap();
    assert_eq!(results.ids()[0], 2);

    // duplicate -> Err, all-or-nothing
    assert!(store.add_batch(&[4, 1], &flat[..4]).is_err());
    assert_eq!(store.size(), 3);
    assert!(!store.contains(4));
}

#[wasm_bindgen_test]
fn test_hnsw_add_batch() {
    let index = WasmHnswIndex::new(2, "l2", 100, 16, 200).unwrap();
    let ids = [10u64, 20];
    let flat = [0.0f32, 0.0, 1.0, 1.0];
    index.add_batch(&ids, &flat).unwrap();
    assert_eq!(index.size(), 2);
    let results = index.search(&[0.1, 0.1], 1).unwrap();
    assert_eq!(results.ids()[0], 10);
}

// --- #39: ids must survive the JS boundary without f32 narrowing ---

const PRECISION_IDS: [u64; 4] = [1 << 24, (1 << 24) + 1, 1 << 53, u64::MAX];

#[wasm_bindgen_test]
fn store_search_round_trips_ids_beyond_f32_precision() {
    let store = WasmVectorStore::new(2, "l2").unwrap();
    for (i, id) in PRECISION_IDS.iter().enumerate() {
        store.add(*id, &[i as f32, 0.0]).unwrap();
    }
    let mut got = store.search(&[0.0, 0.0], 4).unwrap().ids();
    got.sort_unstable();
    let mut want = PRECISION_IDS.to_vec();
    want.sort_unstable();
    assert_eq!(got, want);
}

#[wasm_bindgen_test]
fn hnsw_search_round_trips_ids_beyond_f32_precision() {
    let index = WasmHnswIndex::new(2, "l2", 16, 16, 100).unwrap();
    for (i, id) in PRECISION_IDS.iter().enumerate() {
        index.add(*id, &[i as f32, 0.0]).unwrap();
    }
    let mut got = index.search(&[0.0, 0.0], 4).unwrap().ids();
    got.sort_unstable();
    let mut want = PRECISION_IDS.to_vec();
    want.sort_unstable();
    assert_eq!(got, want);
}
