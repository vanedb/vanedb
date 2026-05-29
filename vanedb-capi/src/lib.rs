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
