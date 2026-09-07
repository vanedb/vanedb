//! Optional GPU distance computation; enabled with `gpu-metal` on macOS.
//!
//! This is a standalone scan API. Upload vectors and call it explicitly;
//! enabling the feature does not move index operations to the GPU.

#[cfg(feature = "gpu-metal")]
/// Apple Metal distance kernels and device buffers.
pub mod metal;

#[cfg(feature = "gpu-metal")]
pub use self::metal::MetalCompute;

use crate::distance::Metric;

/// GPU distance metric (maps from [`Metric`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum GpuMetric {
    /// Squared Euclidean distance.
    L2,
    /// One minus cosine similarity.
    Cosine,
    /// Negative dot product.
    Dot,
}

impl From<Metric> for GpuMetric {
    fn from(m: Metric) -> Self {
        match m {
            Metric::L2 => GpuMetric::L2,
            Metric::Cosine => GpuMetric::Cosine,
            Metric::Dot => GpuMetric::Dot,
        }
    }
}
