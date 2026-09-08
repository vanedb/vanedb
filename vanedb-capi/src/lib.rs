//! C ABI (`vanedb_rs_*`) over the VaneDB core. Mirrors vanedb-cpp's C API.
//! Handle pointers are intentionally non-const and HNSW search takes a per-call
//! ef_search — these match the parallel C++ ABI so a benchmark harness can call
//! both through one uniform FFI. Stored vectors and queries must contain only
//! finite values; raw-pointer wrappers additionally null-guard handles.
//! An unrecognized metric value is rejected: constructors return null.
use std::cell::RefCell;
use std::ffi::CStr;
use std::os::raw::c_char;
use std::ptr;
use std::slice;

use vanedb::distance::distance_fn;
use vanedb::{ApproxIndex, DiskIndex, DiskIndexBuilder, FlatIndex, Metric};

// cbindgen emits one opaque typedef per exported type name. These aliases
// give the C header namespaced names without renaming the Rust types.
#[allow(non_camel_case_types)]
pub type vanedb_rs_store = FlatIndex;
#[allow(non_camel_case_types)]
pub type vanedb_rs_index = ApproxIndex;
#[allow(non_camel_case_types)]
pub type vanedb_rs_disk = DiskIndex;

fn from_metric(m: Metric) -> u32 {
    match m {
        Metric::Cosine => 1,
        Metric::Dot => 2,
        // Metric is #[non_exhaustive]; a variant this ABI does not define
        // cannot reach a handle built through it.
        _ => 0,
    }
}

/// `None` for a value this ABI does not define.
///
/// Mapping an unknown value to L2 would defeat `Metric`'s `#[non_exhaustive]`
/// across the boundary: a caller built against a newer header would get
/// silently wrong distances rather than a refusal.
fn to_metric(m: u32) -> Option<Metric> {
    match m {
        0 => Some(Metric::L2),
        1 => Some(Metric::Cosine),
        2 => Some(Metric::Dot),
        _ => None,
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

/// Success.
pub const VANEDB_RS_OK: u32 = 0;
/// A required handle, buffer or path argument was null. This ABI's own misuse
/// code: it has no `VaneError` counterpart because the call never reached the
/// core.
pub const VANEDB_RS_NULL_ARGUMENT: u32 = 1;
/// The vector's length does not match the handle's dimension.
pub const VANEDB_RS_DIMENSION_MISMATCH: u32 = 2;
/// A batch's id count and vector count disagree.
pub const VANEDB_RS_BATCH_LENGTH_MISMATCH: u32 = 3;
/// A dimension of zero was requested.
pub const VANEDB_RS_ZERO_DIMENSION: u32 = 4;
/// No vector is stored under that id.
pub const VANEDB_RS_NOT_FOUND: u32 = 5;
/// That id is already stored. Retrying will not help; upsert instead.
pub const VANEDB_RS_DUPLICATE_ID: u32 = 6;
/// `k` was zero.
pub const VANEDB_RS_INVALID_K: u32 = 7;
/// A vector or query contained a NaN or an infinity.
pub const VANEDB_RS_NON_FINITE_VALUE: u32 = 8;
/// A construction parameter was out of range — including a metric value this
/// ABI does not define.
pub const VANEDB_RS_INVALID_PARAMETER: u32 = 9;
/// The file does not exist. Distinct from `CORRUPT`: building it is sensible.
pub const VANEDB_RS_FILE_NOT_FOUND: u32 = 10;
/// The file exists but its bytes are wrong. Retrying will not help.
pub const VANEDB_RS_CORRUPT: u32 = 11;
/// An I/O failure. Retrying may help.
pub const VANEDB_RS_IO: u32 = 12;
/// A compute backend was unavailable.
pub const VANEDB_RS_BACKEND: u32 = 13;
/// A panic was caught at the boundary. A bug in the engine; report it.
pub const VANEDB_RS_PANIC: u32 = 14;
/// A `VaneError` variant added after this header was generated. `VaneError` is
/// `#[non_exhaustive]`, so a caller built against an older header must treat
/// any unrecognized code as a failure rather than as success.
pub const VANEDB_RS_UNKNOWN: u32 = 15;

thread_local! {
    static LAST_ERROR: std::cell::Cell<u32> = const { std::cell::Cell::new(VANEDB_RS_OK) };
    static LAST_MESSAGE: std::cell::RefCell<std::ffi::CString> =
        RefCell::new(std::ffi::CString::default());
}

fn code_for(error: &vanedb::VaneError) -> u32 {
    use vanedb::VaneError as E;
    match error {
        E::DimensionMismatch { .. } => VANEDB_RS_DIMENSION_MISMATCH,
        E::BatchLengthMismatch { .. } => VANEDB_RS_BATCH_LENGTH_MISMATCH,
        E::ZeroDimension => VANEDB_RS_ZERO_DIMENSION,
        E::NotFound { .. } => VANEDB_RS_NOT_FOUND,
        E::DuplicateId { .. } => VANEDB_RS_DUPLICATE_ID,
        E::InvalidK => VANEDB_RS_INVALID_K,
        E::NonFiniteValue { .. } => VANEDB_RS_NON_FINITE_VALUE,
        E::InvalidParameter(_) => VANEDB_RS_INVALID_PARAMETER,
        E::FileNotFound { .. } => VANEDB_RS_FILE_NOT_FOUND,
        E::Corrupt { .. } => VANEDB_RS_CORRUPT,
        E::Io { .. } => VANEDB_RS_IO,
        E::Backend { .. } => VANEDB_RS_BACKEND,
        // `VaneError` is #[non_exhaustive]; a variant this ABI predates must
        // still read as a failure.
        _ => VANEDB_RS_UNKNOWN,
    }
}

fn set_code(code: u32, message: &str) {
    LAST_ERROR.with(|slot| slot.set(code));
    // A NUL inside a Display string would be a bug in the core, not the
    // caller's problem: truncate rather than lose the report entirely.
    let owned = std::ffi::CString::new(message).unwrap_or_else(|e| {
        let mut bytes = e.into_vec();
        bytes.truncate(bytes.iter().position(|&b| b == 0).unwrap_or(0));
        std::ffi::CString::new(bytes).unwrap_or_default()
    });
    LAST_MESSAGE.with(|slot| *slot.borrow_mut() = owned);
}

/// Records `error` and yields `fallback`, so a failing arm stays one
/// expression: `Err(e) => fail(e, 1)`.
fn fail<T>(error: vanedb::VaneError, fallback: T) -> T {
    set_code(code_for(&error), &error.to_string());
    fallback
}

/// Records a null argument and yields `fallback`.
fn null_arg<T>(fallback: T) -> T {
    set_code(VANEDB_RS_NULL_ARGUMENT, "null argument");
    fallback
}

/// Records a path argument that is not valid UTF-8. The call never reached
/// the core, so there is no `VaneError` to map.
fn bad_path<T>(fallback: T) -> T {
    set_code(VANEDB_RS_INVALID_PARAMETER, "path is not valid UTF-8");
    fallback
}

/// Records a metric value this ABI does not define.
fn bad_metric<T>(fallback: T) -> T {
    set_code(
        VANEDB_RS_INVALID_PARAMETER,
        "unrecognized metric value: expected 0 (L2), 1 (Cosine) or 2 (Dot)",
    );
    fallback
}

/// The code from the calling thread's most recent `vanedb_rs_*` call, or
/// `VANEDB_RS_OK` if it succeeded.
///
/// The status functions' `1` and the searches' `0` cannot say *why*, and a
/// search returning `0` is otherwise indistinguishable from an empty store.
/// This is what a caller branches on: retry an `IO`, abort a `CORRUPT`, skip a
/// `DUPLICATE_ID`. It is thread-local, so a value set on one thread is never
/// observed on another, and reading it does not clear it.
///
/// This function does not itself count as a call: it leaves the code in place.
#[no_mangle]
pub extern "C" fn vanedb_rs_last_error() -> u32 {
    LAST_ERROR.with(|slot| slot.get())
}

/// The message for the calling thread's most recent failure, NUL-terminated
/// and never null (the empty string after a success).
///
/// This carries the detail the code cannot — which field mismatched, what was
/// wrong with the file.
///
/// The pointer is owned by the library and stays valid only until **either**
/// the next `vanedb_rs_*` call on this thread, **or** this thread exits — the
/// buffer lives in thread-local storage and is dropped with it. Handing the
/// pointer to another thread that outlives this one is a use-after-free, and
/// so is stashing it across a join. Copy the bytes if they need to outlive
/// either event.
///
/// # Safety
/// The returned pointer must not be freed by the caller. It must not be used
/// after another `vanedb_rs_*` call on the same thread, and must not be used
/// after that thread has exited — including by a thread that joined it.
#[no_mangle]
pub extern "C" fn vanedb_rs_last_error_message() -> *const c_char {
    LAST_MESSAGE.with(|slot| slot.borrow().as_ptr())
}

/// This library's version, as a NUL-terminated static string.
///
/// A consumer cannot otherwise check that the shared object it loaded matches
/// the header it compiled against.
#[no_mangle]
pub extern "C" fn vanedb_rs_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
}

/// Runs `body`, returning `fallback` if it panics.
///
/// A panic unwinding out of an `extern "C"` function aborts the process, taking
/// the embedding application with it. Every entry point routes through here so
/// a bug surfaces as this ABI's ordinary failure value — null, 1, 0 or NaN —
/// instead. vanedb-cpp wraps every entry point in try/catch for the same reason.
///
/// Entering also clears the thread's error code, which is what makes
/// `vanedb_rs_last_error` mean "the most recent call" rather than "the most
/// recent failure". Every failing path inside `body` sets it again.
fn guard<T>(fallback: T, body: impl FnOnce() -> T) -> T {
    set_code(VANEDB_RS_OK, "");
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)) {
        Ok(value) => value,
        Err(_) => {
            set_code(VANEDB_RS_PANIC, "a panic was caught at the C ABI boundary");
            fallback
        }
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
        let Some(metric) = to_metric(metric) else {
            return bad_metric(std::ptr::null_mut());
        };
        match FlatIndex::new(dim, metric) {
            Ok(s) => Box::into_raw(Box::new(s)),
            Err(e) => fail(e, std::ptr::null_mut()),
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
            return null_arg(1);
        }
        if v.is_null() {
            return null_arg(1);
        }
        let store = &*s;
        let vec = slice::from_raw_parts(v, store.dimension());
        match store.add(id, vec) {
            Ok(()) => 0,
            Err(e) => fail(e, 1),
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
            return null_arg(1);
        }
        if n != 0 && (ids.is_null() || vecs.is_null()) {
            return null_arg(1);
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
            Err(e) => fail(e, 1),
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
            return null_arg(0);
        }
        if q.is_null() || out_ids.is_null() || out_dists.is_null() {
            return null_arg(0);
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
            Err(e) => fail(e, 0),
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
        let Some(metric) = to_metric(metric) else {
            return bad_metric(std::ptr::null_mut());
        };
        match ApproxIndex::builder(dim, metric)
            .capacity(capacity)
            .m(m)
            .ef_construction(ef_construction)
            .seed(seed)
            .build()
        {
            Ok(h) => Box::into_raw(Box::new(h)),
            Err(e) => fail(e, std::ptr::null_mut()),
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
            return null_arg(1);
        }
        if v.is_null() {
            return null_arg(1);
        }
        let idx = &*h;
        let vec = slice::from_raw_parts(v, idx.dimension());
        match idx.add(id, vec) {
            Ok(()) => 0,
            Err(e) => fail(e, 1),
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
            return null_arg(1);
        }
        if n != 0 && (ids.is_null() || vecs.is_null()) {
            return null_arg(1);
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
            Err(e) => fail(e, 1),
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
            return null_arg(0);
        }
        if q.is_null() || out_ids.is_null() || out_dists.is_null() {
            return null_arg(0);
        }
        let idx = &*h;
        let query = slice::from_raw_parts(q, idx.dimension());
        // Per-call, not a store: mutating the handle here made one caller's
        // beam width visible to every other user of the index, and `save`
        // then wrote it into the file.
        //
        // `ef_search` is required and has no default a C caller can name, so 0
        // is the idiom for "whatever the index is set to". Passing it through
        // would clamp the beam to `k` -- for k=10 that is 5x narrower than the
        // stored default of 50, returning plausible results at silently
        // degraded recall. Resolving it here reads the handle without mutating
        // it, so it stays a per-call parameter.
        let params = if ef_search == 0 {
            vanedb::SearchParams::new()
        } else {
            vanedb::SearchParams::new().ef_search(ef_search)
        };
        match idx.search_with(query, k, &params) {
            Ok(res) => {
                let n = res.len().min(k);
                for (i, r) in res.iter().take(k).enumerate() {
                    *out_ids.add(i) = r.id;
                    *out_dists.add(i) = r.distance;
                }
                n
            }
            Err(e) => fail(e, 0),
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
            return null_arg(1);
        }
        if path.is_null() {
            return null_arg(1);
        }
        let idx = &*h;
        match CStr::from_ptr(path).to_str() {
            Ok(p) => match idx.save(p) {
                Ok(()) => 0,
                Err(e) => fail(e, 1),
            },
            Err(_) => bad_path(1),
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
            return null_arg(std::ptr::null_mut());
        }
        match CStr::from_ptr(path).to_str() {
            Ok(p) => match ApproxIndex::load(p) {
                Ok(h) => Box::into_raw(Box::new(h)),
                Err(e) => fail(e, std::ptr::null_mut()),
            },
            Err(_) => bad_path(std::ptr::null_mut()),
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
            return null_arg(1);
        }
        let p = match CStr::from_ptr(path).to_str() {
            Ok(s) => s,
            Err(_) => return bad_path(1),
        };
        let Some(metric) = to_metric(metric) else {
            return bad_metric(1);
        };
        let mut b = match DiskIndexBuilder::new(dim, metric) {
            Ok(b) => b,
            Err(e) => return fail(e, 1),
        };
        if n != 0 && (ids.is_null() || vecs.is_null()) {
            return null_arg(1);
        }
        let Some(total) = elements(n, dim) else {
            return 1;
        };
        let id_slice: &[u64] = if n == 0 {
            &[]
        } else {
            slice::from_raw_parts(ids, n)
        };
        let vec_slice: &[f32] = if total == 0 {
            &[]
        } else {
            slice::from_raw_parts(vecs, total)
        };
        for (i, &id) in id_slice.iter().enumerate() {
            if b.add(id, &vec_slice[i * dim..(i + 1) * dim]).is_err() {
                return 1;
            }
        }
        match b.save(p) {
            Ok(()) => 0,
            Err(e) => fail(e, 1),
        }
    })
}

/// # Safety
/// `path` must be a valid NUL-terminated C string. Returns an owning handle (or null)
/// that must be freed with `vanedb_rs_disk_free`.
/// The underlying file must not be modified or truncated from the start of
/// this call until the handle is freed. Replacing its path with a newly built
/// file is allowed; modifying the mapped file in place is not.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_disk_open(path: *const c_char) -> *mut vanedb_rs_disk {
    guard(std::ptr::null_mut(), || {
        if path.is_null() {
            return null_arg(std::ptr::null_mut());
        }
        match CStr::from_ptr(path).to_str() {
            // The caller guarantees the mapped file remains immutable.
            Ok(p) => match unsafe { DiskIndex::open(p) } {
                Ok(m) => Box::into_raw(Box::new(m)),
                Err(e) => fail(e, std::ptr::null_mut()),
            },
            Err(_) => bad_path(std::ptr::null_mut()),
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
            return null_arg(0);
        }
        if q.is_null() || out_ids.is_null() || out_dists.is_null() {
            return null_arg(0);
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
            Err(e) => fail(e, 0),
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

// ---------------------------------------------------------------------------
// Introspection and mutation.
//
// Each function is a thin wrapper over the Rust method of the same name. Null
// handles and buffers return this ABI's failure value rather than
// dereferencing, and `guard()` keeps a panic from unwinding across the
// boundary.
// ---------------------------------------------------------------------------

/// Number of vectors in the store, or 0 if `s` is null.
///
/// # Safety
/// `s` must be a live handle from `vanedb_rs_store_new`, or null.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_store_len(s: *const vanedb_rs_store) -> usize {
    guard(0, || if s.is_null() { 0 } else { (*s).len() })
}

/// Vector dimension of the store, or 0 if `s` is null.
///
/// # Safety
/// `s` must be a live handle from `vanedb_rs_store_new`, or null.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_store_dimension(s: *const vanedb_rs_store) -> usize {
    guard(0, || if s.is_null() { 0 } else { (*s).dimension() })
}

/// Metric the store was built with: 0 = L2, 1 = cosine, 2 = dot.
///
/// Returns 0 for a null handle, which is indistinguishable from L2 — check the
/// handle before trusting it, as with every other accessor here.
///
/// # Safety
/// `s` must be a live handle from `vanedb_rs_store_new`, or null.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_store_metric(s: *const vanedb_rs_store) -> u32 {
    guard(0, || {
        if s.is_null() {
            0
        } else {
            from_metric((*s).metric())
        }
    })
}

/// Metric the index was built with: 0 = L2, 1 = cosine, 2 = dot.
///
/// A loaded index reads this from the file, so it is the only way a caller can
/// confirm their query convention matches what was stored. Returns 0 for null.
///
/// # Safety
/// `h` must be a live handle from `vanedb_rs_index_new`/`_load`, or null.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_index_metric(h: *const vanedb_rs_index) -> u32 {
    guard(0, || {
        if h.is_null() {
            0
        } else {
            from_metric((*h).metric())
        }
    })
}

/// Metric the mapped file was written with: 0 = L2, 1 = cosine, 2 = dot.
///
/// Returns 0 for null.
///
/// # Safety
/// `d` must be a live handle from `vanedb_rs_disk_open`, or null.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_disk_metric(d: *const vanedb_rs_disk) -> u32 {
    guard(0, || {
        if d.is_null() {
            0
        } else {
            from_metric((*d).metric())
        }
    })
}

/// Whether `id` is present. False if `s` is null.
///
/// # Safety
/// `s` must be a live handle from `vanedb_rs_store_new`, or null.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_store_contains(s: *const vanedb_rs_store, id: u64) -> bool {
    guard(false, || {
        if s.is_null() {
            false
        } else {
            (*s).contains(id)
        }
    })
}

/// Copies the vector stored under `id` into `out`. Returns 0 on success, 1 if
/// absent or on error.
///
/// # Safety
/// `s` must be a live handle from `vanedb_rs_store_new` (or null); `out` must
/// have room for `vanedb_rs_store_dimension(s)` floats.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_store_get(
    s: *const vanedb_rs_store,
    id: u64,
    out: *mut f32,
) -> i32 {
    guard(1, || {
        if s.is_null() || out.is_null() {
            return null_arg(1);
        }
        match (*s).get(id) {
            Ok(v) => {
                ptr::copy_nonoverlapping(v.as_ptr(), out, v.len());
                0
            }
            Err(e) => fail(e, 1),
        }
    })
}

/// Removes `id`. Returns 0 on success, 1 if absent or on error.
///
/// # Safety
/// `s` must be a live handle from `vanedb_rs_store_new`, or null.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_store_remove(s: *const vanedb_rs_store, id: u64) -> i32 {
    guard(1, || {
        if s.is_null() {
            return null_arg(1);
        }
        if (*s).remove(id).is_ok() {
            0
        } else {
            1
        }
    })
}

/// Number of live vectors in the index, or 0 if `h` is null.
///
/// Excludes tombstones; see `vanedb_rs_index_tombstones`.
///
/// # Safety
/// `h` must be a live handle from `vanedb_rs_index_new`, or null.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_index_len(h: *const vanedb_rs_index) -> usize {
    guard(0, || if h.is_null() { 0 } else { (*h).len() })
}

/// Vector dimension of the index, or 0 if `h` is null.
///
/// # Safety
/// `h` must be a live handle from `vanedb_rs_index_new`, or null.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_index_dimension(h: *const vanedb_rs_index) -> usize {
    guard(0, || if h.is_null() { 0 } else { (*h).dimension() })
}

/// Whether `id` is present and not deleted. False if `h` is null.
///
/// # Safety
/// `h` must be a live handle from `vanedb_rs_index_new`, or null.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_index_contains(h: *const vanedb_rs_index, id: u64) -> bool {
    guard(false, || {
        if h.is_null() {
            false
        } else {
            (*h).contains(id)
        }
    })
}

/// Copies the vector stored under `id` into `out`. Returns 0 on success, 1 if
/// absent or on error.
///
/// # Safety
/// `h` must be a live handle from `vanedb_rs_index_new` (or null); `out` must
/// have room for `vanedb_rs_index_dimension(h)` floats.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_index_get_vector(
    h: *const vanedb_rs_index,
    id: u64,
    out: *mut f32,
) -> i32 {
    guard(1, || {
        if h.is_null() || out.is_null() {
            return null_arg(1);
        }
        match (*h).get_vector(id) {
            Ok(v) => {
                ptr::copy_nonoverlapping(v.as_ptr(), out, v.len());
                0
            }
            Err(e) => fail(e, 1),
        }
    })
}

/// Replaces the vector under `id`, inserting it if absent. Returns 0 on
/// success, 1 on error.
///
/// # Safety
/// `h` must be a live handle from `vanedb_rs_index_new` (or null); `v` must
/// point to `vanedb_rs_index_dimension(h)` valid floats.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_index_upsert(
    h: *const vanedb_rs_index,
    id: u64,
    v: *const f32,
) -> i32 {
    guard(1, || {
        if h.is_null() || v.is_null() {
            return null_arg(1);
        }
        let idx = &*h;
        let vec = slice::from_raw_parts(v, idx.dimension());
        if idx.upsert(id, vec).is_ok() {
            0
        } else {
            1
        }
    })
}

/// Tombstones `id`. Returns 0 on success, 1 if absent or on error.
///
/// The vector stops being returned by searches immediately; its graph links
/// are retained until `vanedb_rs_index_compact`.
///
/// # Safety
/// `h` must be a live handle from `vanedb_rs_index_new`, or null.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_index_remove(h: *const vanedb_rs_index, id: u64) -> i32 {
    guard(1, || {
        if h.is_null() {
            return null_arg(1);
        }
        if (*h).remove(id).is_ok() {
            0
        } else {
            1
        }
    })
}

/// Number of tombstoned slots, or 0 if `h` is null.
///
/// # Safety
/// `h` must be a live handle from `vanedb_rs_index_new`, or null.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_index_tombstones(h: *const vanedb_rs_index) -> usize {
    guard(0, || if h.is_null() { 0 } else { (*h).tombstones() })
}

/// Rebuilds the graph from live vectors, clearing all tombstones. Returns 0 on
/// success, 1 on error.
///
/// # Safety
/// `h` must be a live handle from `vanedb_rs_index_new`, or null.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_index_compact(h: *const vanedb_rs_index) -> i32 {
    guard(1, || {
        if h.is_null() {
            return null_arg(1);
        }
        if (*h).compact().is_ok() {
            0
        } else {
            1
        }
    })
}

/// Number of vectors in the mapped file, or 0 if `d` is null.
///
/// # Safety
/// `d` must be a live handle from `vanedb_rs_disk_open`, or null.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_disk_len(d: *const vanedb_rs_disk) -> usize {
    guard(0, || if d.is_null() { 0 } else { (*d).size() })
}

/// Vector dimension of the mapped file, or 0 if `d` is null.
///
/// # Safety
/// `d` must be a live handle from `vanedb_rs_disk_open`, or null.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_disk_dimension(d: *const vanedb_rs_disk) -> usize {
    guard(0, || if d.is_null() { 0 } else { (*d).dimension() })
}

/// Whether `id` is present. False if `d` is null.
///
/// # Safety
/// `d` must be a live handle from `vanedb_rs_disk_open`, or null.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_disk_contains(d: *const vanedb_rs_disk, id: u64) -> bool {
    guard(false, || {
        if d.is_null() {
            false
        } else {
            (*d).contains(id)
        }
    })
}

/// Copies the vector stored under `id` into `out`. Returns 0 on success, 1 if
/// absent or on error.
///
/// # Safety
/// `d` must be a live handle from `vanedb_rs_disk_open` (or null); `out` must
/// have room for `vanedb_rs_disk_dimension(d)` floats.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_disk_get(
    d: *const vanedb_rs_disk,
    id: u64,
    out: *mut f32,
) -> i32 {
    guard(1, || {
        if d.is_null() || out.is_null() {
            return null_arg(1);
        }
        match (*d).get(id) {
            Ok(v) => {
                ptr::copy_nonoverlapping(v.as_ptr(), out, v.len());
                0
            }
            Err(e) => fail(e, 1),
        }
    })
}

/// Reads the vector stored under `id` into `out`, which must have room for
/// `dim` floats. Returns 0, or 1 if absent.
///
/// The same operation as `vanedb_rs_index_get_vector`, under the spelling the
/// store and disk handles use. Both exist so swapping index type does not mean
/// renaming call sites (#85).
///
/// # Safety
/// `h` must be a live handle from `vanedb_rs_index_new` (or null); `out` must
/// point to at least `dim` writable `f32`s.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_index_get(
    h: *const vanedb_rs_index,
    id: u64,
    out: *mut f32,
) -> i32 {
    vanedb_rs_index_get_vector(h, id, out)
}

/// The same operation as `vanedb_rs_store_get`, under the spelling the graph
/// handle uses (#85).
///
/// # Safety
/// `s` must be a live handle from `vanedb_rs_store_new` (or null); `out` must
/// point to at least `dim` writable `f32`s.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_store_get_vector(
    s: *const vanedb_rs_store,
    id: u64,
    out: *mut f32,
) -> i32 {
    vanedb_rs_store_get(s, id, out)
}

/// The same operation as `vanedb_rs_disk_get`, under the spelling the graph
/// handle uses (#85).
///
/// # Safety
/// `d` must be a live handle from `vanedb_rs_disk_open` (or null); `out` must
/// point to at least `dim` writable `f32`s.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_disk_get_vector(
    d: *const vanedb_rs_disk,
    id: u64,
    out: *mut f32,
) -> i32 {
    vanedb_rs_disk_get(d, id, out)
}

/// The graph's `M`, or 0 for a null handle.
///
/// A handle from `vanedb_rs_index_load` read this from the file, and a caller
/// who did not build it has no other way to know what graph they are searching.
/// The same argument covers `ef_construction`, `seed` and `capacity`.
///
/// # Safety
/// `h` must be a live handle from `vanedb_rs_index_new`/`_load`, or null.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_index_m(h: *const vanedb_rs_index) -> usize {
    guard(0, || if h.is_null() { null_arg(0) } else { (*h).m() })
}

/// The graph's `ef_construction`, or 0 for a null handle.
///
/// # Safety
/// `h` must be a live handle from `vanedb_rs_index_new`/`_load`, or null.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_index_ef_construction(h: *const vanedb_rs_index) -> usize {
    guard(0, || {
        if h.is_null() {
            null_arg(0)
        } else {
            (*h).ef_construction()
        }
    })
}

/// The seed the graph was built with, or 0 for a null handle.
///
/// # Safety
/// `h` must be a live handle from `vanedb_rs_index_new`/`_load`, or null.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_index_seed(h: *const vanedb_rs_index) -> u64 {
    guard(0, || {
        if h.is_null() {
            null_arg(0)
        } else {
            (*h).seed()
        }
    })
}

/// The capacity the graph was built with, or 0 for a null handle.
///
/// This is the build-time hint, not a limit: the index grows past it, so this
/// may be smaller than `vanedb_rs_index_len`.
///
/// # Safety
/// `h` must be a live handle from `vanedb_rs_index_new`/`_load`, or null.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_index_capacity(h: *const vanedb_rs_index) -> usize {
    guard(0, || {
        if h.is_null() {
            null_arg(0)
        } else {
            (*h).capacity()
        }
    })
}

/// The handle's stored `ef_search` — the beam a search gets when it passes 0.
///
/// There was previously no way to read it back, so a caller could not tell what
/// a `0` would resolve to.
///
/// # Safety
/// `h` must be a live handle from `vanedb_rs_index_new`/`_load`, or null.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_index_ef_search(h: *const vanedb_rs_index) -> usize {
    guard(0, || {
        if h.is_null() {
            null_arg(0)
        } else {
            (*h).get_ef_search()
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

    /// Null data pointers return this ABI's failure value rather than being
    /// dereferenced, matching the C++ ABI.
    /// The introspection and mutation surface, end to end through the ABI:
    /// an embedder can read a vector back, delete one, and ask how many the
    /// handle holds.
    #[test]
    fn the_introspection_surface_round_trips() {
        unsafe {
            let s = super::vanedb_rs_store_new(4, 0);
            assert_eq!(super::vanedb_rs_store_dimension(s), 4);
            assert_eq!(super::vanedb_rs_store_len(s), 0);

            let v = [1.0f32, 2.0, 3.0, 4.0];
            assert_eq!(super::vanedb_rs_store_add(s, 7, v.as_ptr()), 0);
            assert_eq!(super::vanedb_rs_store_len(s), 1);
            assert!(super::vanedb_rs_store_contains(s, 7));
            assert!(!super::vanedb_rs_store_contains(s, 8));

            let mut out = [0.0f32; 4];
            assert_eq!(super::vanedb_rs_store_get(s, 7, out.as_mut_ptr()), 0);
            assert_eq!(out, v);
            assert_eq!(super::vanedb_rs_store_get(s, 8, out.as_mut_ptr()), 1);

            assert_eq!(super::vanedb_rs_store_remove(s, 7), 0);
            assert_eq!(super::vanedb_rs_store_remove(s, 7), 1);
            assert_eq!(super::vanedb_rs_store_len(s), 0);
            super::vanedb_rs_store_free(s);

            let h = super::vanedb_rs_index_new(4, 0, 64, 16, 200, 42);
            assert_eq!(super::vanedb_rs_index_dimension(h), 4);
            assert_eq!(super::vanedb_rs_index_add(h, 1, v.as_ptr()), 0);
            assert_eq!(super::vanedb_rs_index_len(h), 1);
            assert!(super::vanedb_rs_index_contains(h, 1));
            assert_eq!(super::vanedb_rs_index_get_vector(h, 1, out.as_mut_ptr()), 0);
            assert_eq!(out, v);

            // upsert replaces in place; len does not grow.
            let w = [9.0f32, 9.0, 9.0, 9.0];
            assert_eq!(super::vanedb_rs_index_upsert(h, 1, w.as_ptr()), 0);
            assert_eq!(super::vanedb_rs_index_len(h), 1);
            assert_eq!(super::vanedb_rs_index_get_vector(h, 1, out.as_mut_ptr()), 0);
            assert_eq!(out, w);

            // upsert tombstones the slot it replaces, so it costs space
            // until a compact: a loop of upserts grows the index.
            assert_eq!(super::vanedb_rs_index_tombstones(h), 1);

            assert_eq!(super::vanedb_rs_index_remove(h, 1), 0);
            assert_eq!(super::vanedb_rs_index_len(h), 0);
            assert_eq!(
                super::vanedb_rs_index_remove(h, 1),
                1,
                "removing twice must fail"
            );
            assert_eq!(super::vanedb_rs_index_tombstones(h), 2);
            assert_eq!(super::vanedb_rs_index_compact(h), 0);
            assert_eq!(super::vanedb_rs_index_tombstones(h), 0);
            assert_eq!(super::vanedb_rs_index_len(h), 0);
            super::vanedb_rs_index_free(h);
        }
    }

    /// A null handle or output pointer returns the failure value rather than
    /// dereferencing, as everywhere else in this ABI.
    #[test]
    fn the_introspection_surface_rejects_nulls() {
        unsafe {
            let mut out = [0.0f32; 4];
            assert_eq!(super::vanedb_rs_store_len(std::ptr::null()), 0);
            assert_eq!(super::vanedb_rs_store_dimension(std::ptr::null()), 0);
            assert!(!super::vanedb_rs_store_contains(std::ptr::null(), 1));
            assert_eq!(
                super::vanedb_rs_store_get(std::ptr::null(), 1, out.as_mut_ptr()),
                1
            );
            assert_eq!(super::vanedb_rs_store_remove(std::ptr::null(), 1), 1);

            assert_eq!(super::vanedb_rs_index_len(std::ptr::null()), 0);
            assert_eq!(super::vanedb_rs_index_dimension(std::ptr::null()), 0);
            assert!(!super::vanedb_rs_index_contains(std::ptr::null(), 1));
            assert_eq!(
                super::vanedb_rs_index_get_vector(std::ptr::null(), 1, out.as_mut_ptr()),
                1
            );
            assert_eq!(super::vanedb_rs_index_remove(std::ptr::null(), 1), 1);
            assert_eq!(super::vanedb_rs_index_tombstones(std::ptr::null()), 0);
            assert_eq!(super::vanedb_rs_index_compact(std::ptr::null()), 1);
            assert_eq!(
                super::vanedb_rs_index_upsert(std::ptr::null(), 1, out.as_ptr()),
                1
            );

            assert_eq!(super::vanedb_rs_disk_len(std::ptr::null()), 0);
            assert_eq!(super::vanedb_rs_disk_dimension(std::ptr::null()), 0);
            assert!(!super::vanedb_rs_disk_contains(std::ptr::null(), 1));
            assert_eq!(
                super::vanedb_rs_disk_get(std::ptr::null(), 1, out.as_mut_ptr()),
                1
            );

            // A live handle with a null output buffer must also be rejected.
            let s = super::vanedb_rs_store_new(4, 0);
            assert_eq!(super::vanedb_rs_store_get(s, 1, std::ptr::null_mut()), 1);
            super::vanedb_rs_store_free(s);
        }
    }

    /// Buffer pointers are checked on every entry point, not just the handle.
    /// Reading through a null query is undefined behaviour, which `guard()`
    /// cannot catch — it intercepts panics, not UB.
    #[test]
    fn null_data_pointers_are_rejected_by_index_and_disk() {
        unsafe {
            let h = super::vanedb_rs_index_new(4, 0, 16, 16, 200, 42);
            assert!(!h.is_null());
            let (mut ids, mut ds) = ([0u64; 2], [0f32; 2]);
            assert_eq!(
                super::vanedb_rs_index_search(
                    h,
                    std::ptr::null(),
                    2,
                    50,
                    ids.as_mut_ptr(),
                    ds.as_mut_ptr()
                ),
                0
            );
            let q = [0.0f32; 4];
            assert_eq!(
                super::vanedb_rs_index_search(
                    h,
                    q.as_ptr(),
                    2,
                    50,
                    std::ptr::null_mut(),
                    ds.as_mut_ptr()
                ),
                0
            );
            assert_eq!(
                super::vanedb_rs_index_search(
                    h,
                    q.as_ptr(),
                    2,
                    50,
                    ids.as_mut_ptr(),
                    std::ptr::null_mut()
                ),
                0
            );
            super::vanedb_rs_index_free(h);

            let path = std::ffi::CString::new(
                std::env::temp_dir()
                    .join("vanedb_capi_null.disk")
                    .to_str()
                    .unwrap(),
            )
            .unwrap();
            let ids2 = [1u64];
            assert_eq!(
                super::vanedb_rs_disk_build(
                    path.as_ptr(),
                    4,
                    0,
                    ids2.as_ptr(),
                    std::ptr::null(),
                    1
                ),
                1
            );
            assert_eq!(
                super::vanedb_rs_disk_build(path.as_ptr(), 4, 0, std::ptr::null(), q.as_ptr(), 1),
                1
            );
            // n * dim overflowing must fail rather than wrap into a bogus slice.
            assert_eq!(
                super::vanedb_rs_disk_build(
                    path.as_ptr(),
                    usize::MAX,
                    0,
                    ids2.as_ptr(),
                    q.as_ptr(),
                    3
                ),
                1
            );
        }
    }

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
