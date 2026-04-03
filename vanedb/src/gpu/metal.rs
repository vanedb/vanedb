use crate::error::{Result, VaneError};
use crate::gpu::GpuMetric;
use crate::store::SearchResult;

use metal::*;
use objc::rc::autoreleasepool;

/// MSL (Metal Shading Language) compute kernels for distance computation.
/// All kernels work on float4 vectors (dim must be divisible by 4).
const MSL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void l2(
    device const float4* q [[buffer(0)]],
    device const float4* v [[buffer(1)]],
    device float* r [[buffer(2)]],
    constant uint& d4 [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    float4 s = 0;
    uint o = i * d4;
    for (uint j = 0; j < d4; ++j) {
        float4 x = q[j] - v[o + j];
        s += x * x;
    }
    r[i] = s.x + s.y + s.z + s.w;
}

kernel void dp(
    device const float4* q [[buffer(0)]],
    device const float4* v [[buffer(1)]],
    device float* r [[buffer(2)]],
    constant uint& d4 [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    float4 s = 0;
    uint o = i * d4;
    for (uint j = 0; j < d4; ++j) {
        s += q[j] * v[o + j];
    }
    r[i] = -(s.x + s.y + s.z + s.w);
}

kernel void cs(
    device const float4* q [[buffer(0)]],
    device const float4* v [[buffer(1)]],
    device float* r [[buffer(2)]],
    constant uint& d4 [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    float4 d = 0, nq = 0, nv = 0;
    uint o = i * d4;
    for (uint j = 0; j < d4; ++j) {
        float4 a = q[j], b = v[o + j];
        d += a * b;
        nq += a * a;
        nv += b * b;
    }
    float dot = d.x + d.y + d.z + d.w;
    float na = nq.x + nq.y + nq.z + nq.w;
    float nb = nv.x + nv.y + nv.z + nv.w;
    float dn = na * nb;
    r[i] = 1.0f - clamp((dn < 1e-12f) ? 0.0f : dot * rsqrt(dn), -1.0f, 1.0f);
}
"#;

/// Handle to vectors uploaded to GPU memory.
pub struct GpuBuffer {
    buffer: Buffer,
    n: usize,
    dim: usize,
}

impl GpuBuffer {
    pub fn n(&self) -> usize {
        self.n
    }
    pub fn dim(&self) -> usize {
        self.dim
    }
}

/// Metal GPU compute for distance calculations.
pub struct MetalCompute {
    device: Device,
    queue: CommandQueue,
    l2_pipeline: ComputePipelineState,
    dot_pipeline: ComputePipelineState,
    cos_pipeline: ComputePipelineState,
}

impl MetalCompute {
    /// Initialize Metal compute. Returns error if no Metal device is available.
    pub fn new() -> Result<Self> {
        let device = Device::system_default()
            .ok_or_else(|| VaneError::Io("no Metal device available".to_string()))?;
        let queue = device.new_command_queue();

        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(MSL_SOURCE, &options)
            .map_err(|e| VaneError::Io(format!("MSL compile error: {e}")))?;

        let l2_fn = library
            .get_function("l2", None)
            .map_err(|e| VaneError::Io(format!("get l2 function: {e}")))?;
        let dp_fn = library
            .get_function("dp", None)
            .map_err(|e| VaneError::Io(format!("get dp function: {e}")))?;
        let cs_fn = library
            .get_function("cs", None)
            .map_err(|e| VaneError::Io(format!("get cs function: {e}")))?;

        let l2_pipeline = device
            .new_compute_pipeline_state_with_function(&l2_fn)
            .map_err(|e| VaneError::Io(format!("l2 pipeline: {e}")))?;
        let dot_pipeline = device
            .new_compute_pipeline_state_with_function(&dp_fn)
            .map_err(|e| VaneError::Io(format!("dot pipeline: {e}")))?;
        let cos_pipeline = device
            .new_compute_pipeline_state_with_function(&cs_fn)
            .map_err(|e| VaneError::Io(format!("cos pipeline: {e}")))?;

        Ok(Self {
            device,
            queue,
            l2_pipeline,
            dot_pipeline,
            cos_pipeline,
        })
    }

    /// Upload vectors to GPU memory. Vectors is a flat array of n * dim floats.
    /// Dimension must be divisible by 4.
    pub fn upload(&self, vectors: &[f32], n: usize, dim: usize) -> Result<GpuBuffer> {
        if !dim.is_multiple_of(4) {
            return Err(VaneError::InvalidParameter("GPU requires dim % 4 == 0"));
        }
        if vectors.len() != n * dim {
            return Err(VaneError::DimensionMismatch {
                expected: n * dim,
                got: vectors.len(),
            });
        }
        let buffer = self.device.new_buffer_with_data(
            vectors.as_ptr() as *const _,
            std::mem::size_of_val(vectors) as NSUInteger,
            MTLResourceOptions::StorageModeShared,
        );
        Ok(GpuBuffer { buffer, n, dim })
    }

    /// Compute distances from query to all vectors in the buffer.
    pub fn distances(
        &self,
        query: &[f32],
        buffer: &GpuBuffer,
        metric: GpuMetric,
    ) -> Result<Vec<f32>> {
        if query.len() != buffer.dim {
            return Err(VaneError::DimensionMismatch {
                expected: buffer.dim,
                got: query.len(),
            });
        }

        let pipeline = match metric {
            GpuMetric::L2 => &self.l2_pipeline,
            GpuMetric::Dot => &self.dot_pipeline,
            GpuMetric::Cosine => &self.cos_pipeline,
        };

        let n = buffer.n;
        let dim = buffer.dim;

        let result = autoreleasepool(|| {
            let query_buf = self.device.new_buffer_with_data(
                query.as_ptr() as *const _,
                (dim * std::mem::size_of::<f32>()) as NSUInteger,
                MTLResourceOptions::StorageModeShared,
            );
            let result_buf = self.device.new_buffer(
                (n * std::mem::size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let d4: u32 = (dim / 4) as u32;
            let dim_buf = self.device.new_buffer_with_data(
                &d4 as *const u32 as *const _,
                std::mem::size_of::<u32>() as NSUInteger,
                MTLResourceOptions::StorageModeShared,
            );

            let command_buffer = self.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(&query_buf), 0);
            encoder.set_buffer(1, Some(&buffer.buffer), 0);
            encoder.set_buffer(2, Some(&result_buf), 0);
            encoder.set_buffer(3, Some(&dim_buf), 0);

            let grid = MTLSize::new(n as u64, 1, 1);
            let max_threads = pipeline.max_total_threads_per_threadgroup();
            let group = MTLSize::new(max_threads.min(n as u64), 1, 1);
            encoder.dispatch_threads(grid, group);
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            let ptr = result_buf.contents() as *const f32;
            unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec()
        });

        Ok(result)
    }

    /// Search for k nearest neighbors using GPU distance computation.
    pub fn search(
        &self,
        query: &[f32],
        ids: &[u64],
        buffer: &GpuBuffer,
        k: usize,
        metric: GpuMetric,
    ) -> Result<Vec<SearchResult>> {
        if k == 0 {
            return Err(VaneError::InvalidK);
        }
        let dists = self.distances(query, buffer, metric)?;
        let mut results: Vec<SearchResult> = ids
            .iter()
            .zip(dists.iter())
            .map(|(&id, &d)| SearchResult::new(id, d))
            .collect();
        results.sort();
        results.truncate(k);
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::{self, DistanceMetric};

    #[test]
    fn metal_init() {
        let gpu = MetalCompute::new();
        assert!(gpu.is_ok(), "Metal should be available on macOS");
    }

    #[test]
    fn metal_l2_matches_cpu() {
        let gpu = MetalCompute::new().unwrap();
        let dim = 128;
        let n = 100;
        let vectors: Vec<f32> = (0..n * dim).map(|i| (i as f32 * 0.01).sin()).collect();
        let query: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.02).cos()).collect();

        let buf = gpu.upload(&vectors, n, dim).unwrap();
        let gpu_dists = gpu.distances(&query, &buf, GpuMetric::L2).unwrap();

        let cpu_dist = distance::distance_fn(DistanceMetric::L2);
        for i in 0..n {
            let cpu_d = cpu_dist(&query, &vectors[i * dim..(i + 1) * dim]);
            assert!(
                (gpu_dists[i] - cpu_d).abs() < 1e-3,
                "vector {i}: gpu={} cpu={cpu_d}",
                gpu_dists[i]
            );
        }
    }

    #[test]
    fn metal_cosine_matches_cpu() {
        let gpu = MetalCompute::new().unwrap();
        let dim = 128;
        let n = 100;
        let vectors: Vec<f32> = (0..n * dim)
            .map(|i| (i as f32 * 0.01).sin() + 0.1)
            .collect();
        let query: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.02).cos() + 0.1).collect();

        let buf = gpu.upload(&vectors, n, dim).unwrap();
        let gpu_dists = gpu.distances(&query, &buf, GpuMetric::Cosine).unwrap();

        let cpu_dist = distance::distance_fn(DistanceMetric::Cosine);
        for i in 0..n {
            let cpu_d = cpu_dist(&query, &vectors[i * dim..(i + 1) * dim]);
            assert!(
                (gpu_dists[i] - cpu_d).abs() < 1e-3,
                "vector {i}: gpu={} cpu={cpu_d}",
                gpu_dists[i]
            );
        }
    }

    #[test]
    fn metal_dot_matches_cpu() {
        let gpu = MetalCompute::new().unwrap();
        let dim = 128;
        let n = 100;
        let vectors: Vec<f32> = (0..n * dim).map(|i| (i as f32 * 0.01).sin()).collect();
        let query: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.02).cos()).collect();

        let buf = gpu.upload(&vectors, n, dim).unwrap();
        let gpu_dists = gpu.distances(&query, &buf, GpuMetric::Dot).unwrap();

        let cpu_dist = distance::distance_fn(DistanceMetric::Dot);
        for i in 0..n {
            let cpu_d = cpu_dist(&query, &vectors[i * dim..(i + 1) * dim]);
            assert!(
                (gpu_dists[i] - cpu_d).abs() < 1e-3,
                "vector {i}: gpu={} cpu={cpu_d}",
                gpu_dists[i]
            );
        }
    }

    #[test]
    fn metal_search_returns_sorted() {
        let gpu = MetalCompute::new().unwrap();
        let dim = 128;
        let n = 50;
        let vectors: Vec<f32> = (0..n * dim).map(|i| (i as f32 * 0.01).sin()).collect();
        let ids: Vec<u64> = (0..n as u64).collect();
        let query: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.02).cos()).collect();

        let buf = gpu.upload(&vectors, n, dim).unwrap();
        let results = gpu.search(&query, &ids, &buf, 5, GpuMetric::L2).unwrap();
        assert_eq!(results.len(), 5);
        for w in results.windows(2) {
            assert!(w[0].distance <= w[1].distance);
        }
    }

    #[test]
    fn metal_rejects_dim_not_divisible_by_4() {
        let gpu = MetalCompute::new().unwrap();
        let vectors = vec![0.0f32; 300]; // 100 vectors of dim 3
        assert!(gpu.upload(&vectors, 100, 3).is_err());
    }
}
