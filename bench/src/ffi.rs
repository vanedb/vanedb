//! Both implementations are reached through their C ABI as non-inlined calls.
use std::os::raw::c_char;

// --- C++ side: declared here, resolved from libvanedb_cpp_capi.a ---
extern "C" {
    pub fn vanedb_cpp_l2_sq(a: *const f32, b: *const f32, dim: usize) -> f32;
    pub fn vanedb_cpp_cosine_distance(a: *const f32, b: *const f32, dim: usize) -> f32;
    pub fn vanedb_cpp_dot_product(a: *const f32, b: *const f32, dim: usize) -> f32;

    pub fn vanedb_cpp_store_new(dim: usize, metric: u32) -> *mut std::ffi::c_void;
    pub fn vanedb_cpp_store_add(s: *mut std::ffi::c_void, id: u64, v: *const f32) -> i32;
    pub fn vanedb_cpp_store_search(
        s: *mut std::ffi::c_void,
        q: *const f32,
        k: usize,
        out_ids: *mut u64,
        out_dists: *mut f32,
    ) -> usize;
    pub fn vanedb_cpp_store_free(s: *mut std::ffi::c_void);

    pub fn vanedb_cpp_hnsw_new(
        dim: usize,
        metric: u32,
        capacity: usize,
        m: usize,
        ef_construction: usize,
        seed: u64,
    ) -> *mut std::ffi::c_void;
    pub fn vanedb_cpp_hnsw_add(h: *mut std::ffi::c_void, id: u64, v: *const f32) -> i32;
    pub fn vanedb_cpp_hnsw_search(
        h: *mut std::ffi::c_void,
        q: *const f32,
        k: usize,
        ef: usize,
        out_ids: *mut u64,
        out_dists: *mut f32,
    ) -> usize;
    pub fn vanedb_cpp_hnsw_save(h: *mut std::ffi::c_void, path: *const c_char) -> i32;
    pub fn vanedb_cpp_hnsw_load(path: *const c_char) -> *mut std::ffi::c_void;
    pub fn vanedb_cpp_hnsw_free(h: *mut std::ffi::c_void);

    pub fn vanedb_cpp_mmap_build(
        path: *const c_char,
        dim: usize,
        metric: u32,
        ids: *const u64,
        vecs: *const f32,
        n: usize,
    ) -> i32;
    pub fn vanedb_cpp_mmap_open(path: *const c_char) -> *mut std::ffi::c_void;
    pub fn vanedb_cpp_mmap_search(
        m: *mut std::ffi::c_void,
        q: *const f32,
        k: usize,
        out_ids: *mut u64,
        out_dists: *mut f32,
    ) -> usize;
    pub fn vanedb_cpp_mmap_free(m: *mut std::ffi::c_void);
}

// --- Rust side: re-exported from the vanedb-capi crate (same #[no_mangle] symbols) ---
pub use vanedb_capi::{
    vanedb_rs_cosine_distance, vanedb_rs_dot_product, vanedb_rs_hnsw_add, vanedb_rs_hnsw_free,
    vanedb_rs_hnsw_load, vanedb_rs_hnsw_new, vanedb_rs_hnsw_save, vanedb_rs_hnsw_search,
    vanedb_rs_l2_sq, vanedb_rs_mmap_build, vanedb_rs_mmap_free, vanedb_rs_mmap_open,
    vanedb_rs_mmap_search, vanedb_rs_store_add, vanedb_rs_store_free, vanedb_rs_store_new,
    vanedb_rs_store_search,
};
