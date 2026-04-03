use crate::error::{Result, VaneError};
use crate::gpu::GpuMetric;
use crate::store::SearchResult;

/// CUDA GPU compute for distance calculations.
/// Requires NVIDIA GPU with CUDA toolkit installed.
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
