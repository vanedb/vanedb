//! C ABI (`vanedb_rs_*`) over the VaneDB core. Mirrors vanedb-cpp's C API.
//! Handle pointers are intentionally non-const and HNSW search takes a per-call
//! ef_search — these match the parallel C++ ABI so a benchmark harness can call
//! both through one uniform FFI. Stored vectors and queries must contain only
//! finite values; raw-pointer wrappers additionally null-guard handles.
//! `to_metric` maps any unrecognized metric value to L2 (no error).
use std::ffi::CStr;
use std::os::raw::c_char;
use std::slice;

use vanedb::distance::distance_fn;
use vanedb::{DiskStore, DiskStoreBuilder, Index, Metric, Store};

// cbindgen emits one opaque typedef per exported type name. These aliases
// give the C header namespaced names without renaming the Rust types.
#[allow(non_camel_case_types)]
pub type vanedb_rs_store = Store;
#[allow(non_camel_case_types)]
pub type vanedb_rs_index = Index;
#[allow(non_camel_case_types)]
pub type vanedb_rs_disk = DiskStore;

fn to_metric(m: u32) -> Metric {
    match m {
        1 => Metric::Cosine,
        2 => Metric::Dot,
        _ => Metric::L2,
    }
}

/// `n * dim`, or `None` on overflow.
///
/// The product is a slice length handed to `from_raw_parts`. Unchecked it
/// panics in debug and wraps in release, producing a slice length unrelated to
/// the caller's buffer -- reachable on 32-bit targets such as wasm32, where
/// `n = 100_000, dim = 50_000` wraps.
fn elements(n: usize, dim: usize) -> Option<usize> {
    n.checked_mul(dim)
}

/// Runs `body`, returning `fallback` if it panics.
///
/// A panic unwinding out of an `extern "C"` function aborts the process, taking
/// the embedding application with it. Every entry point routes through here so
/// a bug surfaces as this ABI's ordinary failure value — null, 1, 0 or NaN —
/// instead. vanedb-cpp wraps every entry point in try/catch for the same reason.
fn guard<T>(fallback: T, body: impl FnOnce() -> T) -> T {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)) {
        Ok(value) => value,
        Err(_) => fallback,
    }
}

/// # Safety
/// `a` and `b` must each point to at least `dim` valid `f32` values.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_l2_sq(a: *const f32, b: *const f32, dim: usize) -> f32 {
    guard(f32::NAN, || {
        distance_fn(Metric::L2)(slice::from_raw_parts(a, dim), slice::from_raw_parts(b, dim))
    })
}

/// # Safety
/// `a` and `b` must each point to at least `dim` valid `f32` values.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_cosine_distance(
    a: *const f32,
    b: *const f32,
    dim: usize,
) -> f32 {
    guard(f32::NAN, || {
        distance_fn(Metric::Cosine)(slice::from_raw_parts(a, dim), slice::from_raw_parts(b, dim))
    })
}

/// # Safety
/// `a` and `b` must each point to at least `dim` valid `f32` values.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_dot_product(a: *const f32, b: *const f32, dim: usize) -> f32 {
    guard(f32::NAN, || {
        // Negate to get the raw inner product (+a·b). The core's distance_fn(Dot) returns the
        // negated distance form (-a·b, lower=closer) for search ranking. This C ABI function must
        // return the raw product to match vanedb_cpp_dot_product, which returns +a·b.
        -distance_fn(Metric::Dot)(slice::from_raw_parts(a, dim), slice::from_raw_parts(b, dim))
    })
}

/// # Safety
/// Safe to call with any arguments; returns an owning handle (or null on error)
/// that must eventually be freed with `vanedb_rs_store_free`.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_store_new(dim: usize, metric: u32) -> *mut vanedb_rs_store {
    guard(std::ptr::null_mut(), || {
        match Store::new(dim, to_metric(metric)) {
            Ok(s) => Box::into_raw(Box::new(s)),
            Err(_) => std::ptr::null_mut(),
        }
    })
}

/// # Safety
/// `s` must be a live handle from `vanedb_rs_store_new` (or null), and
/// `v` must point to at least `dim` valid `f32` values (where `dim` matches the store).
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_store_add(
    s: *mut vanedb_rs_store,
    id: u64,
    v: *const f32,
) -> i32 {
    guard(1, || {
        if s.is_null() {
            return 1;
        }
        if v.is_null() {
            return 1;
        }
        let store = &*s;
        let vec = slice::from_raw_parts(v, store.dimension());
        match store.add(id, vec) {
            Ok(()) => 0,
            Err(_) => 1,
        }
    })
}

/// # Safety
/// `s` must be a live handle from `vanedb_rs_store_new` (or null); `ids` must point to
/// `n` valid `u64`s and `vecs` to `n * dim` valid `f32`s (both may be null when `n` is 0).
/// All-or-nothing: on error (duplicate id, length mismatch) the store is unchanged.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_store_add_batch(
    s: *mut vanedb_rs_store,
    ids: *const u64,
    vecs: *const f32,
    n: usize,
) -> i32 {
    guard(1, || {
        if s.is_null() {
            return 1;
        }
        if n != 0 && (ids.is_null() || vecs.is_null()) {
            return 1;
        }
        let store = &*s;
        let (id_slice, vec_slice): (&[u64], &[f32]) = if n == 0 {
            (&[], &[])
        } else {
            let Some(len) = elements(n, store.dimension()) else {
                return 1;
            };
            (
                slice::from_raw_parts(ids, n),
                slice::from_raw_parts(vecs, len),
            )
        };
        match store.add_batch(id_slice, vec_slice) {
            Ok(()) => 0,
            Err(_) => 1,
        }
    })
}

/// # Safety
/// `s` must be a live handle from `vanedb_rs_store_new` (or null); `q` must point to
/// `dim` valid `f32`s; `out_ids` and `out_dists` must each have room for `k` elements.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_store_search(
    s: *mut vanedb_rs_store,
    q: *const f32,
    k: usize,
    out_ids: *mut u64,
    out_dists: *mut f32,
) -> usize {
    guard(0, || {
        if s.is_null() {
            return 0;
        }
        if q.is_null() || out_ids.is_null() || out_dists.is_null() {
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
    })
}

/// # Safety
/// The handle must have come from `vanedb_rs_store_new` and not been freed already
/// (or be null, which is a no-op).
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_store_free(s: *mut vanedb_rs_store) {
    guard((), || {
        if !s.is_null() {
            drop(Box::from_raw(s));
        }
    })
}

/// # Safety
/// Safe to call with any arguments; returns an owning handle (or null on error)
/// that must eventually be freed with `vanedb_rs_index_free`.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_index_new(
    dim: usize,
    metric: u32,
    capacity: usize,
    m: usize,
    ef_construction: usize,
    seed: u64,
) -> *mut vanedb_rs_index {
    guard(std::ptr::null_mut(), || {
        match Index::builder(dim, to_metric(metric))
            .capacity(capacity)
            .m(m)
            .ef_construction(ef_construction)
            .seed(seed)
            .build()
        {
            Ok(h) => Box::into_raw(Box::new(h)),
            Err(_) => std::ptr::null_mut(),
        }
    })
}

/// # Safety
/// `h` must be a live handle from `vanedb_rs_index_new` (or null), and
/// `v` must point to at least `dim` valid `f32` values (where `dim` matches the index).
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_index_add(
    h: *mut vanedb_rs_index,
    id: u64,
    v: *const f32,
) -> i32 {
    guard(1, || {
        if h.is_null() {
            return 1;
        }
        if v.is_null() {
            return 1;
        }
        let idx = &*h;
        let vec = slice::from_raw_parts(v, idx.dimension());
        match idx.add(id, vec) {
            Ok(()) => 0,
            Err(_) => 1,
        }
    })
}

/// # Safety
/// `h` must be a live handle from `vanedb_rs_index_new` (or null); `ids` must point to
/// `n` valid `u64`s and `vecs` to `n * dim` valid `f32`s (both may be null when `n` is 0).
/// All-or-nothing: on error (duplicate id, capacity, length mismatch) the index is unchanged.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_index_add_batch(
    h: *mut vanedb_rs_index,
    ids: *const u64,
    vecs: *const f32,
    n: usize,
) -> i32 {
    guard(1, || {
        if h.is_null() {
            return 1;
        }
        if n != 0 && (ids.is_null() || vecs.is_null()) {
            return 1;
        }
        let idx = &*h;
        let (id_slice, vec_slice): (&[u64], &[f32]) = if n == 0 {
            (&[], &[])
        } else {
            let Some(len) = elements(n, idx.dimension()) else {
                return 1;
            };
            (
                slice::from_raw_parts(ids, n),
                slice::from_raw_parts(vecs, len),
            )
        };
        match idx.add_batch(id_slice, vec_slice) {
            Ok(()) => 0,
            Err(_) => 1,
        }
    })
}

/// # Safety
/// `h` must be a live handle from `vanedb_rs_index_new` (or null); `q` must point to
/// `dim` valid `f32`s; `out_ids` and `out_dists` must each have room for `k` elements.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_index_search(
    h: *mut vanedb_rs_index,
    q: *const f32,
    k: usize,
    ef_search: usize,
    out_ids: *mut u64,
    out_dists: *mut f32,
) -> usize {
    guard(0, || {
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
    })
}

/// # Safety
/// `h` must be a live handle from `vanedb_rs_index_new` (or null);
/// `path` must be a valid NUL-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_index_save(h: *mut vanedb_rs_index, path: *const c_char) -> i32 {
    guard(1, || {
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
    })
}

/// # Safety
/// `path` must be a valid NUL-terminated C string. Returns an owning handle (or null)
/// that must be freed with `vanedb_rs_index_free`.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_index_load(path: *const c_char) -> *mut vanedb_rs_index {
    guard(std::ptr::null_mut(), || {
        if path.is_null() {
            return std::ptr::null_mut();
        }
        match CStr::from_ptr(path).to_str() {
            Ok(p) => match Index::load(p) {
                Ok(h) => Box::into_raw(Box::new(h)),
                Err(_) => std::ptr::null_mut(),
            },
            Err(_) => std::ptr::null_mut(),
        }
    })
}

/// # Safety
/// The handle must have come from `vanedb_rs_index_new` or `vanedb_rs_index_load`
/// and not been freed already (or be null, which is a no-op).
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_index_free(h: *mut vanedb_rs_index) {
    guard((), || {
        if !h.is_null() {
            drop(Box::from_raw(h));
        }
    })
}

/// # Safety
/// `path` must be a valid NUL-terminated C string; `ids` must point to `n` valid `u64`s
/// and `vecs` to `n * dim` valid `f32`s (both may be null when `n` is 0).
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_disk_build(
    path: *const c_char,
    dim: usize,
    metric: u32,
    ids: *const u64,
    vecs: *const f32,
    n: usize,
) -> i32 {
    guard(1, || {
        if path.is_null() {
            return 1;
        }
        let p = match CStr::from_ptr(path).to_str() {
            Ok(s) => s,
            Err(_) => return 1,
        };
        let mut b = match DiskStoreBuilder::new(dim, to_metric(metric)) {
            Ok(b) => b,
            Err(_) => return 1,
        };
        let id_slice: &[u64] = if n == 0 {
            &[]
        } else {
            slice::from_raw_parts(ids, n)
        };
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
    })
}

/// # Safety
/// `path` must be a valid NUL-terminated C string. Returns an owning handle (or null)
/// that must be freed with `vanedb_rs_disk_free`.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_disk_open(path: *const c_char) -> *mut vanedb_rs_disk {
    guard(std::ptr::null_mut(), || {
        if path.is_null() {
            return std::ptr::null_mut();
        }
        match CStr::from_ptr(path).to_str() {
            Ok(p) => match DiskStore::open(p) {
                Ok(m) => Box::into_raw(Box::new(m)),
                Err(_) => std::ptr::null_mut(),
            },
            Err(_) => std::ptr::null_mut(),
        }
    })
}

/// # Safety
/// `m` must be a live handle from `vanedb_rs_disk_open` (or null); `q` must point to
/// `dim` valid `f32`s; `out_ids` and `out_dists` must each have room for `k` elements.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_disk_search(
    m: *mut vanedb_rs_disk,
    q: *const f32,
    k: usize,
    out_ids: *mut u64,
    out_dists: *mut f32,
) -> usize {
    guard(0, || {
        if m.is_null() {
            return 0;
        }
        if q.is_null() || out_ids.is_null() || out_dists.is_null() {
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
    })
}

/// # Safety
/// The handle must have come from `vanedb_rs_disk_open` and not been freed already
/// (or be null, which is a no-op).
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_disk_free(m: *mut vanedb_rs_disk) {
    guard((), || {
        if !m.is_null() {
            drop(Box::from_raw(m));
        }
    })
}

#[cfg(test)]
mod tests {
    use super::guard;

    /// The guard is what stands between an engine bug and a dead host process,
    /// so it is tested directly rather than through a contrived engine panic —
    /// an entry-point test that never actually panics would pass with the
    /// guard removed and prove nothing.
    #[test]
    fn a_panic_becomes_the_fallback() {
        assert_eq!(guard(1i32, || panic!("engine bug")), 1);
        assert_eq!(guard(0usize, || panic!("engine bug")), 0);
        assert!(guard(f32::NAN, || panic!("engine bug")).is_nan());
        assert!(guard(std::ptr::null_mut::<u8>(), || panic!("engine bug")).is_null());
    }

    /// Null data pointers must return this ABI's failure value, not dereference.
    /// The handle was always checked; the buffers were not, and the C++ ABI
    /// rejects null buffers explicitly.
    #[test]
    fn null_data_pointers_are_rejected() {
        unsafe {
            let s = super::vanedb_rs_store_new(4, 0);
            assert!(!s.is_null());
            assert_eq!(super::vanedb_rs_store_add(s, 1, std::ptr::null()), 1);
            assert_eq!(
                super::vanedb_rs_store_add_batch(s, std::ptr::null(), std::ptr::null(), 3),
                1
            );
            let (mut ids, mut ds) = ([0u64; 2], [0f32; 2]);
            assert_eq!(
                super::vanedb_rs_store_search(
                    s,
                    std::ptr::null(),
                    2,
                    ids.as_mut_ptr(),
                    ds.as_mut_ptr()
                ),
                0
            );
            let v = [1.0f32, 0.0, 0.0, 0.0];
            assert_eq!(
                super::vanedb_rs_store_search(
                    s,
                    v.as_ptr(),
                    2,
                    std::ptr::null_mut(),
                    ds.as_mut_ptr()
                ),
                0
            );
            // A well-formed call still works.
            assert_eq!(super::vanedb_rs_store_add(s, 1, v.as_ptr()), 0);
            super::vanedb_rs_store_free(s);
        }
    }

    #[test]
    fn element_count_refuses_to_wrap() {
        // The realistic 32-bit case: wasm32 with a large n and dim.
        assert_eq!(super::elements(100_000, 8), Some(800_000));
        assert_eq!(super::elements(usize::MAX, 2), None);
        assert_eq!(super::elements(usize::MAX / 2 + 1, 2), None);
        assert_eq!(super::elements(0, usize::MAX), Some(0));
    }

    #[test]
    fn a_normal_return_passes_through_untouched() {
        assert_eq!(guard(1i32, || 0i32), 0);
        assert_eq!(guard(0usize, || 7usize), 7);
    }
}
