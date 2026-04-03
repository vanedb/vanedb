#[cfg(feature = "gpu-cuda")]
pub mod cuda;
#[cfg(feature = "gpu-metal")]
pub mod metal;

#[cfg(feature = "gpu-metal")]
pub use self::metal::MetalCompute;

use crate::distance::DistanceMetric;

/// GPU distance metric (maps from DistanceMetric).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuMetric {
    L2,
    Cosine,
    Dot,
}

impl From<DistanceMetric> for GpuMetric {
    fn from(m: DistanceMetric) -> Self {
        match m {
            DistanceMetric::L2 => GpuMetric::L2,
            DistanceMetric::Cosine => GpuMetric::Cosine,
            DistanceMetric::Dot => GpuMetric::Dot,
        }
    }
}
