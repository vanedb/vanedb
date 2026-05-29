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
