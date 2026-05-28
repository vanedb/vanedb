//! EXPERIMENTAL / UNIMPLEMENTED: this is a stub. Every method returns an error.
//! There is no NVIDIA hardware in CI to validate a real implementation, so CUDA
//! is intentionally unimplemented. Metal (`gpu::metal`) is the supported GPU path.
//! Mirrors the dormant `cuda_distance.cuh` in the C++ implementation — neither
//! has a working CUDA backend today.

use crate::error::{Result, VaneError};
use crate::gpu::GpuMetric;
use crate::store::SearchResult;

/// Unimplemented CUDA compute stub. Construction always fails — see the module
/// docs. Requires NVIDIA GPU + CUDA toolkit and a real implementation first.
pub struct CudaCompute {
    _private: (),
}

/// Handle to vectors in CUDA device memory.
pub struct CudaBuffer {
    _private: (),
}

impl CudaCompute {
    /// Initialize CUDA compute. Returns error if no CUDA device is available.
    pub fn new() -> Result<Self> {
        Err(VaneError::Io(
            "CUDA support requires NVIDIA GPU and CUDA toolkit".to_string(),
        ))
    }

    /// Upload vectors to GPU memory.
    pub fn upload(&self, _vectors: &[f32], _n: usize, _dim: usize) -> Result<CudaBuffer> {
        Err(VaneError::Io("CUDA not available".to_string()))
    }

    /// Compute distances from query to all vectors.
    pub fn distances(
        &self,
        _query: &[f32],
        _buffer: &CudaBuffer,
        _metric: GpuMetric,
    ) -> Result<Vec<f32>> {
        Err(VaneError::Io("CUDA not available".to_string()))
    }

    /// Search for k nearest neighbors using GPU.
    pub fn search(
        &self,
        _query: &[f32],
        _ids: &[u64],
        _buffer: &CudaBuffer,
        _k: usize,
        _metric: GpuMetric,
    ) -> Result<Vec<SearchResult>> {
        Err(VaneError::Io("CUDA not available".to_string()))
    }
}
