//! C ABI (`vanedb_rs_*`) over the VaneDB core. Mirrors vanedb-cpp's C API.
//! Handle pointers are intentionally non-const and HNSW search takes a per-call
//! ef_search — these match the parallel C++ ABI so a benchmark harness can call
//! both through one uniform FFI. Inputs are valid by contract (the bench controls
//! them); raw-pointer wrappers additionally null-guard handles.
use std::ffi::CStr;
use std::os::raw::c_char;
use std::slice;

use vanedb::distance::distance_fn;
use vanedb::{DistanceMetric, HnswIndex, MmapVectorStore, MmapVectorStoreBuilder, VectorStore};

fn to_metric(m: u32) -> DistanceMetric {
    match m {
        1 => DistanceMetric::Cosine,
        2 => DistanceMetric::Dot,
        _ => DistanceMetric::L2,
    }
}

#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_l2_sq(a: *const f32, b: *const f32, dim: usize) -> f32 {
    distance_fn(DistanceMetric::L2)(slice::from_raw_parts(a, dim), slice::from_raw_parts(b, dim))
}

#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_cosine_distance(a: *const f32, b: *const f32, dim: usize) -> f32 {
    distance_fn(DistanceMetric::Cosine)(slice::from_raw_parts(a, dim), slice::from_raw_parts(b, dim))
}

#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_dot_product(a: *const f32, b: *const f32, dim: usize) -> f32 {
    // Negate to get the raw inner product (+a·b). The core's distance_fn(Dot) returns the
    // negated distance form (-a·b, lower=closer) for search ranking. This C ABI function must
    // return the raw product to match vanedb_cpp_dot_product, which returns +a·b.
    -distance_fn(DistanceMetric::Dot)(slice::from_raw_parts(a, dim), slice::from_raw_parts(b, dim))
}

#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_store_new(dim: usize, metric: u32) -> *mut VectorStore {
    match VectorStore::new(dim, to_metric(metric)) {
        Ok(s) => Box::into_raw(Box::new(s)),
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_store_add(s: *mut VectorStore, id: u64, v: *const f32) -> i32 {
    if s.is_null() { return 1; }
    let store = &*s;
    let vec = slice::from_raw_parts(v, store.dimension());
    match store.add(id, vec) { Ok(()) => 0, Err(_) => 1 }
}

#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_store_search(
    s: *mut VectorStore, q: *const f32, k: usize, out_ids: *mut u64, out_dists: *mut f32,
) -> usize {
    if s.is_null() { return 0; }
    let store = &*s;
    let query = slice::from_raw_parts(q, store.dimension());
    match store.search(query, k) {
        Ok(res) => {
            for (i, r) in res.iter().enumerate() {
                *out_ids.add(i) = r.id;
                *out_dists.add(i) = r.distance;
            }
            res.len()
        }
        Err(_) => 0,
    }
}

#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_store_free(s: *mut VectorStore) {
    if !s.is_null() { drop(Box::from_raw(s)); }
}

#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_hnsw_new(
    dim: usize, metric: u32, capacity: usize, m: usize, ef_construction: usize, seed: u64,
) -> *mut HnswIndex {
    match HnswIndex::builder(dim, to_metric(metric))
        .capacity(capacity).m(m).ef_construction(ef_construction).seed(seed).build()
    {
        Ok(h) => Box::into_raw(Box::new(h)),
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_hnsw_add(h: *mut HnswIndex, id: u64, v: *const f32) -> i32 {
    if h.is_null() { return 1; }
    let idx = &*h;
    let vec = slice::from_raw_parts(v, idx.dimension());
    match idx.add(id, vec) { Ok(()) => 0, Err(_) => 1 }
}

#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_hnsw_search(
    h: *mut HnswIndex, q: *const f32, k: usize, ef_search: usize,
    out_ids: *mut u64, out_dists: *mut f32,
) -> usize {
    if h.is_null() { return 0; }
    let idx = &*h;
    idx.set_ef_search(ef_search);
    let query = slice::from_raw_parts(q, idx.dimension());
    match idx.search(query, k) {
        Ok(res) => {
            for (i, r) in res.iter().enumerate() {
                *out_ids.add(i) = r.id;
                *out_dists.add(i) = r.distance;
            }
            res.len()
        }
        Err(_) => 0,
    }
}

#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_hnsw_save(h: *mut HnswIndex, path: *const c_char) -> i32 {
    if h.is_null() { return 1; }
    let idx = &*h;
    match CStr::from_ptr(path).to_str() {
        Ok(p) => match idx.save(p) { Ok(()) => 0, Err(_) => 1 },
        Err(_) => 1,
    }
}

#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_hnsw_load(path: *const c_char) -> *mut HnswIndex {
    match CStr::from_ptr(path).to_str() {
        Ok(p) => match HnswIndex::load(p) {
            Ok(h) => Box::into_raw(Box::new(h)),
            Err(_) => std::ptr::null_mut(),
        },
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_hnsw_free(h: *mut HnswIndex) {
    if !h.is_null() { drop(Box::from_raw(h)); }
}
