//! C ABI (`vanedb_rs_*`) over the VaneDB core. Mirrors vanedb-cpp's C API.
//! Handle pointers are intentionally non-const and HNSW search takes a per-call
//! ef_search — these match the parallel C++ ABI so a benchmark harness can call
//! both through one uniform FFI. Inputs are valid by contract (the bench controls
//! them); raw-pointer wrappers additionally null-guard handles.
//! `to_metric` maps any unrecognized metric value to L2 (no error).
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

/// # Safety
/// `a` and `b` must each point to at least `dim` valid `f32` values.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_l2_sq(a: *const f32, b: *const f32, dim: usize) -> f32 {
    distance_fn(DistanceMetric::L2)(slice::from_raw_parts(a, dim), slice::from_raw_parts(b, dim))
}

/// # Safety
/// `a` and `b` must each point to at least `dim` valid `f32` values.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_cosine_distance(
    a: *const f32,
    b: *const f32,
    dim: usize,
) -> f32 {
    distance_fn(DistanceMetric::Cosine)(
        slice::from_raw_parts(a, dim),
        slice::from_raw_parts(b, dim),
    )
}

/// # Safety
/// `a` and `b` must each point to at least `dim` valid `f32` values.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_dot_product(a: *const f32, b: *const f32, dim: usize) -> f32 {
    // Negate to get the raw inner product (+a·b). The core's distance_fn(Dot) returns the
    // negated distance form (-a·b, lower=closer) for search ranking. This C ABI function must
    // return the raw product to match vanedb_cpp_dot_product, which returns +a·b.
    -distance_fn(DistanceMetric::Dot)(slice::from_raw_parts(a, dim), slice::from_raw_parts(b, dim))
}

/// # Safety
/// Safe to call with any arguments; returns an owning handle (or null on error)
/// that must eventually be freed with `vanedb_rs_store_free`.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_store_new(dim: usize, metric: u32) -> *mut VectorStore {
    match VectorStore::new(dim, to_metric(metric)) {
        Ok(s) => Box::into_raw(Box::new(s)),
        Err(_) => std::ptr::null_mut(),
    }
}

/// # Safety
/// `s` must be a live handle from `vanedb_rs_store_new` (or null), and
/// `v` must point to at least `dim` valid `f32` values (where `dim` matches the store).
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_store_add(s: *mut VectorStore, id: u64, v: *const f32) -> i32 {
    if s.is_null() {
        return 1;
    }
    let store = &*s;
    let vec = slice::from_raw_parts(v, store.dimension());
    match store.add(id, vec) {
        Ok(()) => 0,
        Err(_) => 1,
    }
}

/// # Safety
/// `s` must be a live handle from `vanedb_rs_store_new` (or null); `ids` must point to
/// `n` valid `u64`s and `vecs` to `n * dim` valid `f32`s (both may be null when `n` is 0).
/// All-or-nothing: on error (duplicate id, length mismatch) the store is unchanged.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_store_add_batch(
    s: *mut VectorStore,
    ids: *const u64,
    vecs: *const f32,
    n: usize,
) -> i32 {
    if s.is_null() {
        return 1;
    }
    let store = &*s;
    let (id_slice, vec_slice): (&[u64], &[f32]) = if n == 0 {
        (&[], &[])
    } else {
        (
            slice::from_raw_parts(ids, n),
            slice::from_raw_parts(vecs, n * store.dimension()),
        )
    };
    match store.add_batch(id_slice, vec_slice) {
        Ok(()) => 0,
        Err(_) => 1,
    }
}

/// # Safety
/// `s` must be a live handle from `vanedb_rs_store_new` (or null); `q` must point to
/// `dim` valid `f32`s; `out_ids` and `out_dists` must each have room for `k` elements.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_store_search(
    s: *mut VectorStore,
    q: *const f32,
    k: usize,
    out_ids: *mut u64,
    out_dists: *mut f32,
) -> usize {
    if s.is_null() {
        return 0;
    }
    let store = &*s;
    let query = slice::from_raw_parts(q, store.dimension());
    match store.search(query, k) {
        Ok(res) => {
            let n = res.len().min(k);
            for (i, r) in res.iter().take(k).enumerate() {
                *out_ids.add(i) = r.id;
                *out_dists.add(i) = r.distance;
            }
            n
        }
        Err(_) => 0,
    }
}

/// # Safety
/// The handle must have come from `vanedb_rs_store_new` and not been freed already
/// (or be null, which is a no-op).
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_store_free(s: *mut VectorStore) {
    if !s.is_null() {
        drop(Box::from_raw(s));
    }
}

/// # Safety
/// Safe to call with any arguments; returns an owning handle (or null on error)
/// that must eventually be freed with `vanedb_rs_hnsw_free`.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_hnsw_new(
    dim: usize,
    metric: u32,
    capacity: usize,
    m: usize,
    ef_construction: usize,
    seed: u64,
) -> *mut HnswIndex {
    match HnswIndex::builder(dim, to_metric(metric))
        .capacity(capacity)
        .m(m)
        .ef_construction(ef_construction)
        .seed(seed)
        .build()
    {
        Ok(h) => Box::into_raw(Box::new(h)),
        Err(_) => std::ptr::null_mut(),
    }
}

/// # Safety
/// `h` must be a live handle from `vanedb_rs_hnsw_new` (or null), and
/// `v` must point to at least `dim` valid `f32` values (where `dim` matches the index).
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_hnsw_add(h: *mut HnswIndex, id: u64, v: *const f32) -> i32 {
    if h.is_null() {
        return 1;
    }
    let idx = &*h;
    let vec = slice::from_raw_parts(v, idx.dimension());
    match idx.add(id, vec) {
        Ok(()) => 0,
        Err(_) => 1,
    }
}

/// # Safety
/// `h` must be a live handle from `vanedb_rs_hnsw_new` (or null); `ids` must point to
/// `n` valid `u64`s and `vecs` to `n * dim` valid `f32`s (both may be null when `n` is 0).
/// All-or-nothing: on error (duplicate id, capacity, length mismatch) the index is unchanged.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_hnsw_add_batch(
    h: *mut HnswIndex,
    ids: *const u64,
    vecs: *const f32,
    n: usize,
) -> i32 {
    if h.is_null() {
        return 1;
    }
    let idx = &*h;
    let (id_slice, vec_slice): (&[u64], &[f32]) = if n == 0 {
        (&[], &[])
    } else {
        (
            slice::from_raw_parts(ids, n),
            slice::from_raw_parts(vecs, n * idx.dimension()),
        )
    };
    match idx.add_batch(id_slice, vec_slice) {
        Ok(()) => 0,
        Err(_) => 1,
    }
}

/// # Safety
/// `h` must be a live handle from `vanedb_rs_hnsw_new` (or null); `q` must point to
/// `dim` valid `f32`s; `out_ids` and `out_dists` must each have room for `k` elements.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_hnsw_search(
    h: *mut HnswIndex,
    q: *const f32,
    k: usize,
    ef_search: usize,
    out_ids: *mut u64,
    out_dists: *mut f32,
) -> usize {
    if h.is_null() {
        return 0;
    }
    let idx = &*h;
    idx.set_ef_search(ef_search);
    let query = slice::from_raw_parts(q, idx.dimension());
    match idx.search(query, k) {
        Ok(res) => {
            let n = res.len().min(k);
            for (i, r) in res.iter().take(k).enumerate() {
                *out_ids.add(i) = r.id;
                *out_dists.add(i) = r.distance;
            }
            n
        }
        Err(_) => 0,
    }
}

/// # Safety
/// `h` must be a live handle from `vanedb_rs_hnsw_new` (or null);
/// `path` must be a valid NUL-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_hnsw_save(h: *mut HnswIndex, path: *const c_char) -> i32 {
    if h.is_null() {
        return 1;
    }
    if path.is_null() {
        return 1;
    }
    let idx = &*h;
    match CStr::from_ptr(path).to_str() {
        Ok(p) => match idx.save(p) {
            Ok(()) => 0,
            Err(_) => 1,
        },
        Err(_) => 1,
    }
}

/// # Safety
/// `path` must be a valid NUL-terminated C string. Returns an owning handle (or null)
/// that must be freed with `vanedb_rs_hnsw_free`.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_hnsw_load(path: *const c_char) -> *mut HnswIndex {
    if path.is_null() {
        return std::ptr::null_mut();
    }
    match CStr::from_ptr(path).to_str() {
        Ok(p) => match HnswIndex::load(p) {
            Ok(h) => Box::into_raw(Box::new(h)),
            Err(_) => std::ptr::null_mut(),
        },
        Err(_) => std::ptr::null_mut(),
    }
}

/// # Safety
/// The handle must have come from `vanedb_rs_hnsw_new` or `vanedb_rs_hnsw_load`
/// and not been freed already (or be null, which is a no-op).
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_hnsw_free(h: *mut HnswIndex) {
    if !h.is_null() {
        drop(Box::from_raw(h));
    }
}

/// # Safety
/// `path` must be a valid NUL-terminated C string; `ids` must point to `n` valid `u64`s;
/// `vecs` must point to `n * dim` valid `f32`s.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_mmap_build(
    path: *const c_char,
    dim: usize,
    metric: u32,
    ids: *const u64,
    vecs: *const f32,
    n: usize,
) -> i32 {
    if path.is_null() {
        return 1;
    }
    let p = match CStr::from_ptr(path).to_str() {
        Ok(s) => s,
        Err(_) => return 1,
    };
    let mut b = match MmapVectorStoreBuilder::new(dim, to_metric(metric)) {
        Ok(b) => b,
        Err(_) => return 1,
    };
    let id_slice = slice::from_raw_parts(ids, n);
    for (i, &id) in id_slice.iter().enumerate() {
        let v = slice::from_raw_parts(vecs.add(i * dim), dim);
        if b.add(id, v).is_err() {
            return 1;
        }
    }
    match b.save(p) {
        Ok(()) => 0,
        Err(_) => 1,
    }
}

/// # Safety
/// `path` must be a valid NUL-terminated C string. Returns an owning handle (or null)
/// that must be freed with `vanedb_rs_mmap_free`.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_mmap_open(path: *const c_char) -> *mut MmapVectorStore {
    if path.is_null() {
        return std::ptr::null_mut();
    }
    match CStr::from_ptr(path).to_str() {
        Ok(p) => match MmapVectorStore::open(p) {
            Ok(m) => Box::into_raw(Box::new(m)),
            Err(_) => std::ptr::null_mut(),
        },
        Err(_) => std::ptr::null_mut(),
    }
}

/// # Safety
/// `m` must be a live handle from `vanedb_rs_mmap_open` (or null); `q` must point to
/// `dim` valid `f32`s; `out_ids` and `out_dists` must each have room for `k` elements.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_mmap_search(
    m: *mut MmapVectorStore,
    q: *const f32,
    k: usize,
    out_ids: *mut u64,
    out_dists: *mut f32,
) -> usize {
    if m.is_null() {
        return 0;
    }
    let store = &*m;
    let query = slice::from_raw_parts(q, store.dimension());
    match store.search(query, k) {
        Ok(res) => {
            let n = res.len().min(k);
            for (i, r) in res.iter().take(k).enumerate() {
                *out_ids.add(i) = r.id;
                *out_dists.add(i) = r.distance;
            }
            n
        }
        Err(_) => 0,
    }
}

/// # Safety
/// The handle must have come from `vanedb_rs_mmap_open` and not been freed already
/// (or be null, which is a no-op).
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_mmap_free(m: *mut MmapVectorStore) {
    if !m.is_null() {
        drop(Box::from_raw(m));
    }
}
