use vanedb_wasm::*;
use wasm_bindgen::JsValue;
use wasm_bindgen_test::*;

#[wasm_bindgen_test]
fn test_version() {
    // Compared against the same source as the binding, so the two cannot
    // drift apart the way a pair of literals did.
    assert_eq!(version(), env!("CARGO_PKG_VERSION"));
}

#[wasm_bindgen_test]
fn test_vector_store_basic() {
    let store = WasmStore::new(3.0, &JsValue::from_str("l2")).unwrap();
    store.add(1u64.into(), &[1.0, 0.0, 0.0]).unwrap();
    store.add(2u64.into(), &[0.0, 1.0, 0.0]).unwrap();
    assert_eq!(store.size(), 2);
    assert_eq!(store.dimension(), 3);
    assert!(store.contains(1u64.into()).unwrap());
    assert!(!store.contains(99u64.into()).unwrap());
}

#[wasm_bindgen_test]
fn test_vector_store_search() {
    let store = WasmStore::new(2.0, &JsValue::from_str("l2")).unwrap();
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
    let idx = WasmIndex::new(3.0, &JsValue::from_str("l2"), 100.0, 16.0, 200.0, None).unwrap();
    idx.add(1u64.into(), &[1.0, 0.0, 0.0]).unwrap();
    idx.add(2u64.into(), &[0.0, 1.0, 0.0]).unwrap();
    assert_eq!(idx.size(), 2);
    assert!(idx.contains(1u64.into()).unwrap());
}

#[wasm_bindgen_test]
fn test_hnsw_search() {
    let idx = WasmIndex::new(3.0, &JsValue::from_str("l2"), 100.0, 16.0, 200.0, None).unwrap();
    idx.add(1u64.into(), &[0.0, 0.0, 0.0]).unwrap();
    idx.add(2u64.into(), &[10.0, 10.0, 10.0]).unwrap();

    let hits = idx.search(&[0.0, 0.0, 0.0], 1.0, None).unwrap();
    assert_eq!(hits.ids()[0], 1);
}

#[wasm_bindgen_test]
fn test_cosine_metric() {
    let store = WasmStore::new(2.0, &JsValue::from_str("cosine")).unwrap();
    store.add(1u64.into(), &[1.0, 0.0]).unwrap();
    store.add(2u64.into(), &[0.0, 1.0]).unwrap();
    let hits = store.search(&[0.9, 0.1], 1.0).unwrap();
    assert_eq!(hits.ids()[0], 1);
}

#[wasm_bindgen_test]
fn test_invalid_metric() {
    let result = WasmStore::new(3.0, &JsValue::from_str("invalid"));
    assert!(result.is_err());
}

#[wasm_bindgen_test]
fn test_store_add_batch() {
    let store = WasmStore::new(2.0, &JsValue::from_str("l2")).unwrap();
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
    let index = WasmIndex::new(2.0, &JsValue::from_str("l2"), 100.0, 16.0, 200.0, None).unwrap();
    let ids = [10u64, 20];
    let flat = [0.0f32, 0.0, 1.0, 1.0];
    index.add_batch(&ids, &flat).unwrap();
    assert_eq!(index.size(), 2);
    let results = index.search(&[0.1, 0.1], 1.0, None).unwrap();
    assert_eq!(results.ids()[0], 10);
}

// --- #39: ids must survive the JS boundary without f32 narrowing ---

const PRECISION_IDS: [u64; 4] = [1 << 24, (1 << 24) + 1, 1 << 53, u64::MAX];

#[wasm_bindgen_test]
fn store_search_round_trips_ids_beyond_f32_precision() {
    let store = WasmStore::new(2.0, &JsValue::from_str("l2")).unwrap();
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
    let index = WasmIndex::new(2.0, &JsValue::from_str("l2"), 16.0, 16.0, 100.0, None).unwrap();
    for (i, id) in PRECISION_IDS.iter().enumerate() {
        index.add((*id).into(), &[i as f32, 0.0]).unwrap();
    }
    let mut got = index.search(&[0.0, 0.0], 4.0, None).unwrap().ids();
    got.sort_unstable();
    let mut want = PRECISION_IDS.to_vec();
    want.sort_unstable();
    assert_eq!(got, want);
}

#[wasm_bindgen_test]
fn test_non_finite_vectors_and_queries_are_rejected() {
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let store = WasmStore::new(2.0, &JsValue::from_str("l2")).unwrap();
        assert!(store.add(1u64.into(), &[value, 0.0]).is_err());
        assert_eq!(store.size(), 0);
        store.add(2u64.into(), &[0.0, 0.0]).unwrap();
        assert!(store.search(&[value, 0.0], 1.0).is_err());

        let index = WasmIndex::new(2.0, &JsValue::from_str("l2"), 4.0, 2.0, 10.0, None).unwrap();
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
        let store = WasmStore::new(2.0, &JsValue::from_str(name)).unwrap();
        assert_eq!(store.metric(), name);
        let index = WasmIndex::new(2.0, &JsValue::from_str(name), 16.0, 4.0, 40.0, None).unwrap();
        assert_eq!(index.metric(), name);
        // Round-trips through the constructor it names.
        assert!(WasmStore::new(2.0, &JsValue::from_str(&store.metric())).is_ok());
    }
}

#[wasm_bindgen_test]
fn dot_ranks_by_largest_inner_product() {
    let store = WasmStore::new(2.0, &JsValue::from_str("dot")).unwrap();
    store.add(1u64.into(), &[1.0, 0.0]).unwrap();
    store.add(2u64.into(), &[4.0, 0.0]).unwrap();
    store.add(3u64.into(), &[0.0, 1.0]).unwrap();
    let hits = store.search(&[1.0, 0.0], 3.0).unwrap();
    let ids = hits.ids();
    assert_eq!(ids[0], 2, "dot must rank the largest inner product first");
    assert_eq!(ids[2], 3);
}

/// `remove` shipped without `tombstones` or `compact`, so a browser app that
/// churns entries — the workload an embedded vector DB exists for — grew
/// without bound in the most memory-constrained runtime this project targets,
/// with no API to measure or reclaim it.
#[wasm_bindgen_test]
fn a_deleted_entry_can_be_measured_and_reclaimed() {
    let index = WasmIndex::new(1.0, &JsValue::from_str("l2"), 16.0, 4.0, 16.0, None).unwrap();
    for id in 0..8u64 {
        index.add(id.into(), &[id as f32]).unwrap();
    }
    assert_eq!(index.tombstones(), 0);

    index.remove(3u64.into()).unwrap();
    index.remove(5u64.into()).unwrap();
    assert_eq!(index.size(), 6);
    assert_eq!(index.tombstones(), 2, "a deletion must be observable");

    index.compact().unwrap();
    assert_eq!(index.tombstones(), 0, "compaction must reclaim the slots");
    assert_eq!(index.size(), 6, "compaction must keep the live set");
    for id in [0u64, 1, 2, 4, 6, 7] {
        assert!(index.contains(id.into()).unwrap(), "{id} was live");
    }
    assert!(!index.contains(3u64.into()).unwrap());
}

/// A JS caller could not read a stored vector back from `ApproxIndex` at all —
/// `FlatIndex` had `get`, this had neither spelling — while the README claimed
/// wasm supports "lookup methods".
#[wasm_bindgen_test]
fn a_stored_vector_can_be_read_back() {
    let index = WasmIndex::new(3.0, &JsValue::from_str("l2"), 16.0, 4.0, 16.0, None).unwrap();
    index.add(7u64.into(), &[1.0, 2.0, 3.0]).unwrap();
    assert_eq!(index.get_vector(7u64.into()).unwrap(), vec![1.0, 2.0, 3.0]);
    assert_eq!(index.get(7u64.into()).unwrap(), vec![1.0, 2.0, 3.0]);
    assert!(index.get_vector(99u64.into()).is_err());
}

/// Both read spellings exist on both index types (#85). `ApproxIndex` had the
/// pair and `FlatIndex` only `get`, so a program written against the graph
/// index broke on the one swap the pair exists to make painless.
#[wasm_bindgen_test]
fn both_read_spellings_exist_on_the_exact_index_too() {
    let store = WasmStore::new(3.0, &JsValue::from_str("l2")).unwrap();
    store.add(7u64.into(), &[1.0, 2.0, 3.0]).unwrap();
    assert_eq!(store.get(7u64.into()).unwrap(), vec![1.0, 2.0, 3.0]);
    assert_eq!(store.get_vector(7u64.into()).unwrap(), vec![1.0, 2.0, 3.0]);
    assert!(store.get_vector(99u64.into()).is_err());
}

/// The seed was hardcoded to 42, so reproducible graph construction was
/// impossible from JS. Omitting it must keep the previous default.
#[wasm_bindgen_test]
fn the_construction_seed_is_settable_and_defaults_as_before() {
    let defaulted = WasmIndex::new(2.0, &JsValue::from_str("l2"), 16.0, 4.0, 16.0, None).unwrap();
    assert_eq!(defaulted.seed(), 42);

    let seeded =
        WasmIndex::new(2.0, &JsValue::from_str("l2"), 16.0, 4.0, 16.0, Some(1234.0)).unwrap();
    assert_eq!(seeded.seed(), 1234);
    assert_eq!(seeded.m(), 4);
    assert_eq!(seeded.ef_construction(), 16);
    assert_eq!(seeded.capacity(), 16);
}

/// Every other mutator takes `&self`; `remove` alone took `&mut self`, which in
/// wasm-bindgen means a JS caller holding any other borrow of the object gets
/// "recursive use of an object detected" instead of a deletion.
#[wasm_bindgen_test]
fn remove_does_not_require_an_exclusive_borrow() {
    let index = WasmIndex::new(1.0, &JsValue::from_str("l2"), 16.0, 4.0, 16.0, None).unwrap();
    index.add(1u64.into(), &[1.0]).unwrap();
    let shared = &index;
    shared.remove(1u64.into()).unwrap();
    assert_eq!(shared.size(), 0);
}

/// A non-string `metric` used to trap inside wasm-bindgen's string marshalling
/// with `RuntimeError: memory access out of bounds`, an opaque VM fault where
/// every other bad constructor argument throws a descriptive error. Nothing
/// covered it: the suite passed on the trapping build, because every test
/// passes a string. Reverting the check would leave it green.
#[wasm_bindgen_test]
fn a_non_string_metric_is_rejected_not_trapped() {
    for value in [
        JsValue::from_f64(5.0),
        JsValue::from_bool(true),
        JsValue::NULL,
        JsValue::UNDEFINED,
        JsValue::from(js_sys::Array::new()),
        JsValue::from(js_sys::Object::new()),
    ] {
        let store = WasmStore::new(2.0, &value);
        assert!(store.is_err(), "{value:?} was accepted as a metric");
        let index = WasmIndex::new(2.0, &value, 16.0, 4.0, 40.0, None);
        assert!(index.is_err(), "{value:?} was accepted as a metric");
    }
    // A string that is not a metric stays a metric error, not a type error.
    assert!(WasmStore::new(2.0, &JsValue::from_str("nope")).is_err());
}

/// `upsert` is one locked operation where `remove` then `add` is two.
///
/// wasm shipped without it while Rust, Python and the C ABI had it, and the
/// README stated the omission without a reason. There is none: persistence is
/// absent because a browser has no filesystem, but this is pure in-memory
/// state, and `add` and `remove` were already exposed.
#[wasm_bindgen_test]
fn upsert_replaces_in_place_and_inserts_when_absent() {
    let index = WasmIndex::new(3.0, &JsValue::from_str("l2"), 100.0, 16.0, 200.0, None).unwrap();
    index.add(1u64.into(), &[0.0, 0.0, 0.0]).unwrap();

    index.upsert(1u64.into(), &[5.0, 5.0, 5.0]).unwrap();
    assert_eq!(index.size(), 1, "replacing must not grow the index");
    assert_eq!(index.get(1u64.into()).unwrap(), vec![5.0, 5.0, 5.0]);
    assert_eq!(index.tombstones(), 1, "the replaced slot is tombstoned");

    index.upsert(2u64.into(), &[1.0, 1.0, 1.0]).unwrap();
    assert_eq!(index.size(), 2, "an absent id is inserted, not refused");

    // The property that makes this worth having over remove-then-add: a
    // rejected upsert leaves the entry alone, where the two-call form can
    // fail after the removal and lose it.
    assert!(index.upsert(1u64.into(), &[1.0, 1.0]).is_err());
    assert!(
        index.contains(1u64.into()).unwrap(),
        "a failed upsert must not delete"
    );
    assert_eq!(index.get(1u64.into()).unwrap(), vec![5.0, 5.0, 5.0]);
}

/// A beam width for one query, leaving the index's own setting alone.
///
/// `ef_search` is a property, so the only way to widen a single hard query was
/// to raise it, search, and lower it again -- three calls, and every search in
/// between pays. Rust, Python and the C ABI all take it per call.
#[wasm_bindgen_test]
fn per_query_ef_search_leaves_the_shared_setting_alone() {
    let index = WasmIndex::new(2.0, &JsValue::from_str("l2"), 200.0, 4.0, 8.0, Some(7.0)).unwrap();
    for i in 0..60u64 {
        let x = i as f32;
        index.add(i.into(), &[x, x]).unwrap();
    }
    index.set_ef_search(1.0).unwrap();

    let widened = index.search(&[0.0, 0.0], 5.0, Some(64.0)).unwrap();
    assert_eq!(
        widened.ids(),
        vec![0, 1, 2, 3, 4],
        "a beam wider than the corpus must return the exact answer"
    );
    assert_eq!(
        index.ef_search(),
        1,
        "a per-query width must not write to the shared property"
    );

    // Omitting it falls back to that property, and it is validated like `k`.
    assert_eq!(index.search(&[0.0, 0.0], 5.0, None).unwrap().length(), 5);
    assert!(index.search(&[0.0, 0.0], 5.0, Some(-1.0)).is_err());
    assert!(index.search(&[0.0, 0.0], 5.0, Some(f64::NAN)).is_err());
}
