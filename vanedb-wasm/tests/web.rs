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

    let flat = store.search(&[0.0, 0.1], 2).unwrap();
    assert_eq!(flat.len(), 4); // [id0, dist0, id1, dist1]
    assert_eq!(flat[0] as u64, 1); // closest
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

    let flat = idx.search(&[0.0, 0.0, 0.0], 1).unwrap();
    assert_eq!(flat[0] as u64, 1);
}

#[wasm_bindgen_test]
fn test_cosine_metric() {
    let store = WasmVectorStore::new(2, "cosine").unwrap();
    store.add(1, &[1.0, 0.0]).unwrap();
    store.add(2, &[0.0, 1.0]).unwrap();
    let flat = store.search(&[0.9, 0.1], 1).unwrap();
    assert_eq!(flat[0] as u64, 1);
}

#[wasm_bindgen_test]
fn test_invalid_metric() {
    let result = WasmVectorStore::new(3, "invalid");
    assert!(result.is_err());
}
