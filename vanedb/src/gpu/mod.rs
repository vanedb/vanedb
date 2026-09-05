#[cfg(feature = "gpu-cuda")]
pub mod cuda;
#[cfg(feature = "gpu-metal")]
pub mod metal;

#[cfg(feature = "gpu-metal")]
pub use self::metal::MetalCompute;

use crate::distance::Metric;

/// GPU distance metric (maps from Metric).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuMetric {
    L2,
    Cosine,
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
