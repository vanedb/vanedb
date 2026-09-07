//! GPU backends. Metal is the only one; a CUDA path can be added when there
//! is hardware in CI to validate it, which is additive rather than breaking.
//!
//! These are a standalone parallel-scan API: no index uses them internally.
//! The caller supplies a flat corpus, uploads it, and searches the handle.

/// Metal kernels, on Apple platforms.
#[cfg(feature = "gpu-metal")]
pub mod metal;

#[cfg(feature = "gpu-metal")]
pub use self::metal::MetalCompute;

use crate::distance::Metric;

/// GPU distance metric (maps from [`Metric`]).
///
/// `#[non_exhaustive]`, like [`Metric`]: matching needs a `_` arm, so adding a
/// metric is not a breaking change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum GpuMetric {
    /// Squared Euclidean distance.
    L2,
    /// Cosine distance, `1 - cos(a, b)`.
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
