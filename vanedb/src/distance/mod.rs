//! Distance metrics and their SIMD implementations.
//!
//! Every kernel compares `a.len().min(b.len())` elements: mismatched
//! lengths truncate to the shorter slice rather than reading past its
//! end. The index types reject a dimension mismatch before they get
//! here, so this governs only direct callers of [`distance_fn`].

/// AVX2 kernels, compiled on x86-64.
#[cfg(target_arch = "x86_64")]
pub mod avx2;
/// NEON kernels, compiled on AArch64.
#[cfg(target_arch = "aarch64")]
pub mod neon;
/// Portable kernels, always available.
pub mod scalar;

/// Distance metric for vector comparison.
///
/// `#[non_exhaustive]`: matching on a `Metric` needs a `_` arm, so a new
/// metric can be added without a breaking release.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Metric {
    /// Squared Euclidean distance
    L2,
    /// Cosine distance (1 - cosine similarity).
    ///
    /// A zero vector has no direction, so the angle to it is undefined. This
    /// crate reports 1.0 — the maximum distance — whenever either *computed*
    /// squared norm is zero or is not finite, including a zero vector's
    /// distance to itself. Ranking it as maximally distant keeps it out of
    /// results rather than making it a NaN that sorts unpredictably. Both
    /// engines share this policy.
    ///
    /// "Computed" is load-bearing at both ends of the range, and neither end
    /// is an error:
    ///
    /// - **Overflow.** Components around 1e19 and up square past `f32::MAX`,
    ///   so the norm is infinite.
    /// - **Underflow.** Components below roughly 3.7e-23 square to zero, so
    ///   the norm is zero even though the vector is not. Such a vector is 1.0
    ///   from everything, itself included — a plausible-looking input with no
    ///   warning attached. Rescale before indexing if your embeddings live
    ///   down there. `conformance/cosine_scale_invariance.tsv` pins both ends.
    Cosine,
    /// Negative dot product (higher similarity = lower distance).
    ///
    /// Dot similarity depends on magnitude: a longer vector scores better
    /// than a shorter one pointing the same way,
    /// so a vector need not be its own nearest neighbour. Normalise, or keep
    /// magnitudes comparable, if you want similarity-search semantics.
    ///
    /// Magnitudes large enough to overflow the inner product give a distance
    /// of negative infinity, and the shared result order places every
    /// non-finite distance *after* the finite ones — so an overflowing pair
    /// ranks last rather than first. Both engines do this deliberately: a
    /// saturated score carries no ranking information, and letting it win
    /// would put an arbitrary vector at the top of every result set.
    Dot,
}

/// Function type for distance computation. Compares the first
/// `a.len().min(b.len())` elements; see the module documentation.
pub type DistanceFn = fn(&[f32], &[f32]) -> f32;

/// Returns the distance function for the given metric.
/// Automatically selects SIMD implementation when available.
pub fn distance_fn(metric: Metric) -> DistanceFn {
    match metric {
        Metric::L2 => l2_squared,
        Metric::Cosine => cosine_distance,
        Metric::Dot => dot_distance,
    }
}

pub(crate) fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        neon::l2_squared(a, b)
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { avx2::l2_squared(a, b) }
        } else {
            scalar::l2_squared(a, b)
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        scalar::l2_squared(a, b)
    }
}

pub(crate) fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        neon::cosine_distance(a, b)
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { avx2::cosine_distance(a, b) }
        } else {
            scalar::cosine_distance(a, b)
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        scalar::cosine_distance(a, b)
    }
}

pub(crate) fn dot_distance(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        neon::dot_distance(a, b)
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { avx2::dot_distance(a, b) }
        } else {
            scalar::dot_distance(a, b)
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        scalar::dot_distance(a, b)
    }
}
