pub mod scalar;

/// Distance metric for vector comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    /// Squared Euclidean distance
    L2,
    /// Cosine distance (1 - cosine similarity)
    Cosine,
    /// Negative dot product (higher similarity = lower distance)
    Dot,
}

/// Function type for distance computation.
pub type DistanceFn = fn(&[f32], &[f32]) -> f32;

/// Zero-norm threshold for cosine distance.
pub const COSINE_EPSILON: f32 = 1e-12;

/// Returns the distance function for the given metric.
/// Automatically selects SIMD implementation when available.
pub fn distance_fn(metric: DistanceMetric) -> DistanceFn {
    match metric {
        DistanceMetric::L2 => l2_squared,
        DistanceMetric::Cosine => cosine_distance,
        DistanceMetric::Dot => dot_distance,
    }
}

fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    scalar::l2_squared(a, b)
}

fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    scalar::cosine_distance(a, b)
}

fn dot_distance(a: &[f32], b: &[f32]) -> f32 {
    scalar::dot_distance(a, b)
}
