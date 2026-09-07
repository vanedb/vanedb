use crate::error::{Result, VaneError};
use crate::flat::{topk, SearchResult};
use crate::gpu::GpuMetric;
use crate::validation::validate_finite;

use metal::foreign_types::{ForeignType, ForeignTypeRef};
use metal::*;
use objc::rc::autoreleasepool;
use objc::{msg_send, sel, sel_impl};

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
    float dn = sqrt(na) * sqrt(nb);
    float sim = (dn > 0.0f && isfinite(dn)) ? dot / dn : 0.0f;
    r[i] = 1.0f - clamp(sim, -1.0f, 1.0f);
}
"#;

/// Handle to vectors uploaded to GPU memory.
pub struct GpuBuffer {
    buffer: Buffer,
    n: usize,
    dim: usize,
    scalar_fallback: bool,
}

impl GpuBuffer {
    /// Number of uploaded vectors.
    pub fn n(&self) -> usize {
        self.n
    }
    /// Components per uploaded vector.
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

#[allow(unexpected_cfgs)] // objc 0.2 macros probe the legacy cargo-clippy cfg.
impl MetalCompute {
    /// Initialize Metal compute on macOS 10.14 or newer.
    /// Returns an error if the device or required resources are unavailable.
    pub fn new() -> Result<Self> {
        let device = Device::system_default()
            .ok_or_else(|| VaneError::backend("no Metal device available"))?;
        let has_buffer_limit: objc::runtime::BOOL =
            unsafe { msg_send![&*device, respondsToSelector:sel!(maxBufferLength)] };
        if has_buffer_limit == objc::runtime::NO {
            return Err(VaneError::backend("Metal requires macOS 10.14 or newer"));
        }
        // These Metal selectors may return nil; metal-rs's convenience APIs
        // construct non-null wrappers without checking. Own only non-null results.
        let queue: *mut MTLCommandQueue = unsafe { msg_send![&*device, newCommandQueue] };
        if queue.is_null() {
            return Err(VaneError::backend("Metal command queue allocation failed"));
        }
        let queue = unsafe { CommandQueue::from_ptr(queue) };

        let options = CompileOptions::new();
        // Preserve the finite/overflow checks in cosine; fast math assumes
        // infinities cannot occur even when finite inputs overflow a norm.
        options.set_fast_math_enabled(false);
        let library = device
            .new_library_with_source(MSL_SOURCE, &options)
            .map_err(|e| VaneError::backend(format!("MSL compile error: {e}")))?;

        let l2_fn = library
            .get_function("l2", None)
            .map_err(|e| VaneError::backend(format!("get l2 function: {e}")))?;
        let dp_fn = library
            .get_function("dp", None)
            .map_err(|e| VaneError::backend(format!("get dp function: {e}")))?;
        let cs_fn = library
            .get_function("cs", None)
            .map_err(|e| VaneError::backend(format!("get cs function: {e}")))?;

        let l2_pipeline = device
            .new_compute_pipeline_state_with_function(&l2_fn)
            .map_err(|e| VaneError::backend(format!("l2 pipeline: {e}")))?;
        let dot_pipeline = device
            .new_compute_pipeline_state_with_function(&dp_fn)
            .map_err(|e| VaneError::backend(format!("dot pipeline: {e}")))?;
        let cos_pipeline = device
            .new_compute_pipeline_state_with_function(&cs_fn)
            .map_err(|e| VaneError::backend(format!("cos pipeline: {e}")))?;

        Ok(Self {
            device,
            queue,
            l2_pipeline,
            dot_pipeline,
            cos_pipeline,
        })
    }

    fn new_buffer(&self, bytes: usize) -> Result<Buffer> {
        // Keep an empty upload representable, but never dispatch it.
        let bytes = bytes.max(std::mem::size_of::<f32>());
        if bytes as u64 > self.device.max_buffer_length() {
            return Err(VaneError::InvalidParameter(
                "buffer exceeds the Metal device limit",
            ));
        }
        // newBuffer returns an owned (+1) nullable Objective-C object.
        let raw: *mut MTLBuffer = unsafe {
            msg_send![&*self.device, newBufferWithLength:bytes as NSUInteger
                options:MTLResourceOptions::StorageModeShared]
        };
        if raw.is_null() {
            return Err(VaneError::backend("Metal buffer allocation failed"));
        }
        let buffer = unsafe { Buffer::from_ptr(raw) };
        if buffer.contents().is_null() || buffer.length() < bytes as u64 {
            return Err(VaneError::backend("Metal shared buffer is not accessible"));
        }
        Ok(buffer)
    }

    fn copy_buffer(&self, values: &[f32]) -> Result<Buffer> {
        let buffer = self.new_buffer(std::mem::size_of_val(values))?;
        // Shared Metal storage is aligned and was allocated for this slice;
        // the new allocation cannot overlap the caller's source.
        unsafe {
            std::ptr::copy_nonoverlapping(values.as_ptr(), buffer.contents().cast(), values.len());
        }
        Ok(buffer)
    }

    /// Upload `n` finite vectors from a flat array of exactly `n * dim` floats.
    /// Dimension must be nonzero and divisible by 4; sizes must fit the device
    /// and the kernels' 32-bit addressing. Empty uploads are supported.
    pub fn upload(&self, vectors: &[f32], n: usize, dim: usize) -> Result<GpuBuffer> {
        if dim == 0 {
            return Err(VaneError::ZeroDimension);
        }
        if dim % 4 != 0 {
            return Err(VaneError::InvalidParameter("GPU requires dim % 4 == 0"));
        }
        let expected = n
            .checked_mul(dim)
            .ok_or(VaneError::InvalidParameter("GPU n * dim overflows usize"))?;
        expected
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or(VaneError::InvalidParameter(
                "GPU buffer byte size overflows usize",
            ))?;
        if n > u32::MAX as usize || dim / 4 > u32::MAX as usize || expected / 4 > u32::MAX as usize
        {
            return Err(VaneError::InvalidParameter(
                "GPU shape exceeds 32-bit shader addressing",
            ));
        }
        if vectors.len() != expected {
            return Err(VaneError::DimensionMismatch {
                expected,
                got: vectors.len(),
            });
        }
        let scalar_fallback = classify_upload(vectors)?;
        let buffer = self.copy_buffer(vectors)?;
        Ok(GpuBuffer {
            buffer,
            n,
            dim,
            scalar_fallback,
        })
    }

    /// Compute distances from a finite query to every uploaded vector.
    /// The buffer must belong to the same Metal device. Empty buffers return
    /// an empty result. Tiny nonzero components use CPU distance kernels,
    /// preserving contributions that Metal hardware would flush to zero.
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

        validate_finite(query, "query")?;
        if buffer.buffer.device().registry_id() != self.device.registry_id() {
            return Err(VaneError::InvalidParameter(
                "buffer belongs to another Metal device",
            ));
        }
        if buffer.n == 0 {
            return Ok(Vec::new());
        }
        if buffer.scalar_fallback || needs_scalar(query) {
            // Upload validated the product and copied these immutable values
            // into shared storage. The owning buffer lives through this scan.
            let values = unsafe {
                std::slice::from_raw_parts(
                    buffer.buffer.contents().cast::<f32>(),
                    buffer.n * buffer.dim,
                )
            };
            let distance: crate::distance::DistanceFn = match metric {
                GpuMetric::L2 => crate::distance::l2_squared,
                GpuMetric::Cosine => crate::distance::cosine_distance,
                GpuMetric::Dot => crate::distance::dot_distance,
            };
            return Ok(values
                .chunks_exact(buffer.dim)
                .map(|vector| distance(query, vector))
                .collect());
        }

        let pipeline = match metric {
            GpuMetric::L2 => &self.l2_pipeline,
            GpuMetric::Dot => &self.dot_pipeline,
            GpuMetric::Cosine => &self.cos_pipeline,
        };

        let n = buffer.n;
        let dim = buffer.dim;

        autoreleasepool(|| {
            let query_buf = self.copy_buffer(query)?;
            // n * 4 cannot overflow: upload validated n * dim * 4, and dim >= 4.
            let result_buf = self.new_buffer(n * std::mem::size_of::<f32>())?;
            let d4: u32 = (dim / 4) as u32;

            // Both selectors return autoreleased, nullable objects. Check before
            // borrowing; the enclosing pool outlives encoding and completion.
            let raw: *mut MTLCommandBuffer = unsafe { msg_send![&*self.queue, commandBuffer] };
            if raw.is_null() {
                return Err(VaneError::backend("Metal command buffer allocation failed"));
            }
            let command_buffer = unsafe { CommandBufferRef::from_ptr(raw) };
            let raw: *mut MTLComputeCommandEncoder =
                unsafe { msg_send![command_buffer, computeCommandEncoder] };
            if raw.is_null() {
                return Err(VaneError::backend(
                    "Metal compute encoder allocation failed",
                ));
            }
            let encoder = unsafe { ComputeCommandEncoderRef::from_ptr(raw) };
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(&query_buf), 0);
            encoder.set_buffer(1, Some(&buffer.buffer), 0);
            encoder.set_buffer(2, Some(&result_buf), 0);
            encoder.set_bytes(
                3,
                std::mem::size_of_val(&d4) as NSUInteger,
                (&d4 as *const u32).cast(),
            );

            let grid = MTLSize::new(n as u64, 1, 1);
            let max_threads = pipeline.max_total_threads_per_threadgroup();
            let group = MTLSize::new(max_threads.min(n as u64), 1, 1);
            encoder.dispatch_threads(grid, group);
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            if command_buffer.status() != MTLCommandBufferStatus::Completed {
                return Err(VaneError::backend(format!(
                    "Metal command failed: {:?}",
                    command_buffer.status()
                )));
            }
            let ptr = result_buf.contents() as *const f32;
            // Allocation and accessibility were checked above. Completion means
            // every result has been written and is visible in shared storage.
            Ok(unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec())
        })
    }

    /// Search for `k` nearest neighbors, with exactly one ID per uploaded row.
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
        if ids.len() != buffer.n {
            return Err(VaneError::InvalidParameter(
                "ids length must match uploaded vector count",
            ));
        }
        let dists = self.distances(query, buffer, metric)?;
        Ok(topk::select(
            ids.iter()
                .zip(dists)
                .map(|(&id, distance)| SearchResult::new(id, distance)),
            k,
        ))
    }
}

fn classify_upload(values: &[f32]) -> Result<bool> {
    let threshold = (f32::MIN_POSITIVE.sqrt() / f32::EPSILON).to_bits();
    let mut invalid = 0_u32;
    let mut tiny = 0_u32;
    for value in values {
        // Positive f32 bit patterns have the same order as their magnitudes.
        // Bitwise reductions avoid per-element exits and permit SIMD scanning.
        let magnitude = value.to_bits() & 0x7fff_ffff;
        invalid |= u32::from(magnitude >= f32::INFINITY.to_bits());
        tiny |= u32::from((magnitude != 0) & (magnitude < threshold));
    }
    if invalid != 0 {
        return Err(VaneError::NonFiniteValue {
            input: "vector batch",
        });
    }
    Ok(tiny != 0)
}

fn needs_scalar(values: &[f32]) -> bool {
    // Below 2^-40, adjacent f32 values can have a difference whose square is
    // subnormal, even when the components' own squares are normal. This also
    // covers subnormal dot/norm terms before Metal flushes them to zero.
    let threshold = f32::MIN_POSITIVE.sqrt() / f32::EPSILON;
    values
        .iter()
        .any(|value| *value != 0.0 && value.abs() < threshold)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::{self, Metric};

    #[test]
    fn upload_classification_preserves_zero_and_threshold_boundaries() {
        let threshold = f32::MIN_POSITIVE.sqrt() / f32::EPSILON;
        let below = f32::from_bits(threshold.to_bits() - 1);
        let above = f32::from_bits(threshold.to_bits() + 1);
        assert!(!classify_upload(&[]).unwrap());
        assert!(!classify_upload(&[
            0.0,
            -0.0,
            threshold,
            -threshold,
            above,
            -above,
            f32::MAX,
            -f32::MAX,
        ])
        .unwrap());
        for value in [below, -below, f32::from_bits(1), -f32::from_bits(1)] {
            assert!(classify_upload(&[0.0, value, -0.0]).unwrap());
        }
    }

    #[test]
    fn upload_classification_never_skips_invalid_values_after_tiny_values() {
        for bits in [0x7f80_0000, 0x7f80_0001, 0x7fc0_0000, 0x7fff_ffff] {
            for sign in [0, 0x8000_0000] {
                // Cover vectorized blocks and scalar tails, with a tiny value
                // found before each positive/negative infinity or NaN payload.
                for len in [2, 7, 8, 9, 31, 32, 33, 65] {
                    let mut values = vec![1.0; len];
                    values[0] = f32::from_bits(1);
                    values[len - 1] = f32::from_bits(bits | sign);
                    assert!(matches!(
                        classify_upload(&values),
                        Err(VaneError::NonFiniteValue {
                            input: "vector batch"
                        })
                    ));
                }
            }
        }
    }

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

        let cpu_dist = distance::distance_fn(Metric::L2);
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

        let cpu_dist = distance::distance_fn(Metric::Cosine);
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

        let cpu_dist = distance::distance_fn(Metric::Dot);
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

    #[test]
    fn metal_upload_rejects_size_overflow() {
        let gpu = MetalCompute::new().unwrap();
        assert!(matches!(
            gpu.upload(&[], usize::MAX, 4),
            Err(VaneError::InvalidParameter(_))
        ));
    }

    #[test]
    fn metal_rejects_non_finite_inputs() {
        let gpu = MetalCompute::new().unwrap();
        let buffer = gpu.upload(&[0.0; 4], 1, 4).unwrap();
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(matches!(
                gpu.upload(&[value; 4], 1, 4),
                Err(VaneError::NonFiniteValue { .. })
            ));
            assert!(matches!(
                gpu.distances(&[value; 4], &buffer, GpuMetric::L2),
                Err(VaneError::NonFiniteValue { .. })
            ));
        }
    }

    #[test]
    fn metal_search_rejects_mismatched_ids() {
        let gpu = MetalCompute::new().unwrap();
        let buffer = gpu.upload(&[0.0; 8], 2, 4).unwrap();
        for ids in [&[1][..], &[1, 2, 3][..]] {
            assert!(gpu
                .search(&[0.0; 4], ids, &buffer, 2, GpuMetric::L2)
                .is_err());
        }
    }

    #[test]
    fn metal_cosine_preserves_degenerate_scale_semantics() {
        let gpu = MetalCompute::new().unwrap();
        for value in [0.0, 1e-20, 1e-19, 1e-10, 1.0, 1e10, 1e20] {
            let vector = [value; 4];
            let buffer = gpu.upload(&vector, 1, 4).unwrap();
            let actual = gpu.distances(&vector, &buffer, GpuMetric::Cosine).unwrap()[0];
            let expected = distance::scalar::cosine_distance(&vector, &vector);
            assert!(
                actual.is_finite() && (actual - expected).abs() < 1e-5,
                "scale {value}: Metal {actual}, CPU {expected}"
            );
        }
    }

    #[test]
    fn metal_empty_upload_is_searchable_and_still_validates_queries() {
        let gpu = MetalCompute::new().unwrap();
        assert!(matches!(
            gpu.upload(&[], 0, 0),
            Err(VaneError::ZeroDimension)
        ));
        let buffer = gpu.upload(&[], 0, 4).unwrap();
        assert_eq!((buffer.n(), buffer.dim()), (0, 4));
        for metric in [GpuMetric::L2, GpuMetric::Cosine, GpuMetric::Dot] {
            assert!(gpu
                .distances(&[0.0; 4], &buffer, metric)
                .unwrap()
                .is_empty());
            assert!(gpu
                .search(&[0.0; 4], &[], &buffer, usize::MAX, metric)
                .unwrap()
                .is_empty());
        }
        assert!(gpu.distances(&[], &buffer, GpuMetric::L2).is_err());
        assert!(gpu
            .distances(&[f32::NAN; 4], &buffer, GpuMetric::L2)
            .is_err());
        assert!(matches!(
            gpu.search(&[0.0; 4], &[], &buffer, 0, GpuMetric::L2),
            Err(VaneError::InvalidK)
        ));
    }

    #[test]
    fn metal_rejects_device_and_shader_size_limits_without_allocating() {
        let gpu = MetalCompute::new().unwrap();
        assert!(gpu
            .new_buffer(gpu.device.max_buffer_length() as usize + 1)
            .is_err());
        assert!(gpu.upload(&[], 1, usize::MAX - 3).is_err());
        assert!(gpu.upload(&[], u32::MAX as usize + 1, 4).is_err());
        assert!(gpu.upload(&[], 0, (u32::MAX as usize + 1) * 4).is_err());
    }

    #[test]
    fn metal_tiny_inputs_keep_cpu_rankings_for_every_metric() {
        let gpu = MetalCompute::new().unwrap();
        for dim in [4, 256] {
            let mut vectors = vec![1e-20; dim];
            vectors.resize(dim * 2, 0.0);
            let query = vec![1e-20; dim];
            let buffer = gpu.upload(&vectors, 2, dim).unwrap();
            for metric in [GpuMetric::L2, GpuMetric::Cosine, GpuMetric::Dot] {
                let hits = gpu.search(&query, &[99, 1], &buffer, 2, metric).unwrap();
                assert_eq!(hits[0].id, 99, "{metric:?} dim {dim}: {hits:?}");
                assert!(hits[0].distance < hits[1].distance);
            }
        }
    }

    #[test]
    fn metal_l2_preserves_subnormal_squared_differences() {
        let gpu = MetalCompute::new().unwrap();
        let value = 1e-15_f32;
        let next = f32::from_bits(value.to_bits() + 1);
        let vectors = [value, value, value, value, next, next, next, next];
        let buffer = gpu.upload(&vectors, 2, 4).unwrap();
        let hits = gpu
            .search(&[value; 4], &[99, 1], &buffer, 2, GpuMetric::L2)
            .unwrap();
        assert_eq!(hits[0].id, 99, "{hits:?}");
        assert_eq!(hits[0].distance, 0.0);
        assert_eq!(
            hits[1].distance,
            distance::l2_squared(&[value; 4], &[next; 4])
        );
        assert!(hits[1].distance > 0.0);
    }

    #[test]
    fn metal_topk_keeps_ties_and_finite_first_order() {
        let gpu = MetalCompute::new().unwrap();
        let buffer = gpu.upload(&[0.0; 16], 4, 4).unwrap();
        for k in [1, 2, usize::MAX] {
            let hits = gpu
                .search(&[0.0; 4], &[4, 1, 3, 2], &buffer, k, GpuMetric::L2)
                .unwrap();
            assert_eq!(
                hits.iter().map(|hit| hit.id).collect::<Vec<_>>(),
                (1..=k.min(4) as u64).collect::<Vec<_>>()
            );
        }
        let buffer = gpu
            .upload(&[1e20, 1e20, 1e20, 1e20, 0.0, 0.0, 0.0, 0.0], 2, 4)
            .unwrap();
        let hits = gpu
            .search(&[1e20; 4], &[1, 2], &buffer, 2, GpuMetric::Dot)
            .unwrap();
        assert_eq!(hits[0].id, 2);
        assert!(hits[0].distance.is_finite());
        assert_eq!(hits[1].distance, f32::NEG_INFINITY);
    }

    #[test]
    fn metal_buffer_can_be_shared_by_compute_instances_on_the_same_device() {
        let first = MetalCompute::new().unwrap();
        let second = MetalCompute::new().unwrap();
        assert_eq!(first.device.registry_id(), second.device.registry_id());
        let buffer = first.upload(&[1.0; 4], 1, 4).unwrap();
        drop(first);
        assert_eq!(
            second.distances(&[1.0; 4], &buffer, GpuMetric::L2).unwrap(),
            [0.0]
        );
    }
}
