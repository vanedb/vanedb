use std::path::Path;

use super::{BuildParams, BuildStats, Engine, MetricKind};
use crate::measure::{current_rss_bytes, InstantTimer};

#[link(name = "hnsw_bridge")]
extern "C" {
    fn hnsw_bridge_create(
        dim: usize,
        max_elements: usize,
        m: usize,
        ef_construction: usize,
        metric: i32,
    ) -> *mut HnswBridge;
    fn hnsw_bridge_free(index: *mut HnswBridge);
    fn hnsw_bridge_add(index: *mut HnswBridge, id: u64, vector: *const f32) -> i32;
    fn hnsw_bridge_mark_deleted(index: *mut HnswBridge, id: u64) -> i32;
    fn hnsw_bridge_set_ef(index: *mut HnswBridge, ef: usize);
    fn hnsw_bridge_search(
        index: *mut HnswBridge,
        query: *const f32,
        k: usize,
        out_ids: *mut u64,
    ) -> i32;
    fn hnsw_bridge_save(index: *mut HnswBridge, path: *const libc::c_char) -> i32;
}

enum HnswBridge {}

pub struct HnswlibEngine {
    ptr: *mut HnswBridge,
    dim: usize,
}

// The bridge is used from one thread in the harness.
unsafe impl Send for HnswlibEngine {}

impl HnswlibEngine {
    pub fn new() -> Self {
        Self {
            ptr: std::ptr::null_mut(),
            dim: 0,
        }
    }
}

impl Default for HnswlibEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for HnswlibEngine {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { hnsw_bridge_free(self.ptr) };
            self.ptr = std::ptr::null_mut();
        }
    }
}

impl Engine for HnswlibEngine {
    fn name(&self) -> &'static str {
        "hnswlib"
    }

    fn version(&self) -> String {
        "0.8.0".to_string()
    }

    fn supports_delete(&self) -> bool {
        true
    }

    fn supports_save(&self) -> bool {
        true
    }

    fn build(
        &mut self,
        vectors: &[f32],
        ids: &[u64],
        dim: usize,
        metric: MetricKind,
        params: &BuildParams,
    ) -> Result<BuildStats, String> {
        if !self.ptr.is_null() {
            unsafe { hnsw_bridge_free(self.ptr) };
            self.ptr = std::ptr::null_mut();
        }
        let metric_i = match metric {
            MetricKind::L2 => 0,
            MetricKind::Cosine => 1,
        };
        let rss_before = current_rss_bytes();
        let timer = InstantTimer::start();
        let ptr = unsafe {
            hnsw_bridge_create(
                dim,
                ids.len().max(1),
                params.m,
                params.ef_construction,
                metric_i,
            )
        };
        if ptr.is_null() {
            return Err("hnswlib: create failed".into());
        }
        for (i, &id) in ids.iter().enumerate() {
            let row = &vectors[i * dim..(i + 1) * dim];
            let rc = unsafe { hnsw_bridge_add(ptr, id, row.as_ptr()) };
            if rc != 0 {
                unsafe { hnsw_bridge_free(ptr) };
                return Err(format!("hnswlib: add failed at id {id}"));
            }
        }
        unsafe { hnsw_bridge_set_ef(ptr, params.ef_search) };
        let build_secs = timer.elapsed_secs();
        let rss_after = current_rss_bytes();
        self.ptr = ptr;
        self.dim = dim;
        let mut notes = vec![format!(
            "vendored hnswlib v0.8.0; M={} efC={}",
            params.m, params.ef_construction
        )];
        if metric == MetricKind::Cosine {
            notes.push(
                "cosine via InnerProductSpace on L2-normalized vectors (hnswlib convention)".into(),
            );
        }
        let _ = self.dim;
        Ok(BuildStats {
            build_secs,
            peak_rss_bytes: rss_delta(rss_before, rss_after),
            notes,
        })
    }

    fn search(&self, query: &[f32], k: usize, ef: usize) -> Result<Vec<u64>, String> {
        if self.ptr.is_null() {
            return Err("hnswlib: not built".into());
        }
        unsafe { hnsw_bridge_set_ef(self.ptr, ef) };
        let mut out = vec![0u64; k];
        let n = unsafe { hnsw_bridge_search(self.ptr, query.as_ptr(), k, out.as_mut_ptr()) };
        if n < 0 {
            return Err("hnswlib: search failed".into());
        }
        out.truncate(n as usize);
        Ok(out)
    }

    fn save(&self, path: &Path) -> Result<u64, String> {
        if self.ptr.is_null() {
            return Err("hnswlib: not built".into());
        }
        let c_path =
            std::ffi::CString::new(path.to_string_lossy().as_bytes()).map_err(|e| e.to_string())?;
        let rc = unsafe { hnsw_bridge_save(self.ptr, c_path.as_ptr()) };
        if rc != 0 {
            return Err("hnswlib: save failed".into());
        }
        std::fs::metadata(path)
            .map(|m| m.len())
            .map_err(|e| e.to_string())
    }

    fn remove(&mut self, id: u64) -> Result<(), String> {
        if self.ptr.is_null() {
            return Err("hnswlib: not built".into());
        }
        let rc = unsafe { hnsw_bridge_mark_deleted(self.ptr, id) };
        if rc != 0 {
            return Err("hnswlib: markDelete failed".into());
        }
        Ok(())
    }
}

fn rss_delta(before: Option<u64>, after: Option<u64>) -> Option<u64> {
    match (before, after) {
        (Some(b), Some(a)) if a >= b => Some(a - b),
        (_, after) => after,
    }
}
