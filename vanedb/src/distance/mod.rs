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
    /// crate reports 1.0 whenever either *computed* squared norm is zero or is
    /// not finite, including a zero vector's distance to itself. 1.0 is the
    /// orthogonal distance, not the maximum: `1 - cos` spans `[0, 2]`, and an
    /// antiparallel pair is 2.0. So a degenerate vector ranks as unrelated
    /// rather than as maximally distant, and will sort ahead of genuinely
    /// opposed vectors — the alternative, a NaN, sorts unpredictably. Both
    /// engines share this policy.
    ///
    /// "Computed" is load-bearing at both ends of the range, and neither end
    /// is an error:
    ///
    /// - **Overflow.** A vector whose magnitude reaches roughly 1.8e19
    ///   (`sqrt(f32::MAX)`) has a squared norm past `f32::MAX`, so the norm is
    ///   infinite. It is the norm that decides, not any one component: 128
    ///   components of 1e19 each are individually under the bound and still
    ///   sum past it.
    /// - **Underflow.** A component at or below roughly 2.6e-23 (`2^-75`)
    ///   squares to zero — not `sqrt(f32::MIN_POSITIVE_SUBNORMAL)` ≈ 3.7e-23,
    ///   because round-to-nearest rounds a square in `(2^-150, 2^-149)` *up* to
    ///   the minimum subnormal rather than down to zero. The interval is open
    ///   at the bottom: `(2^-75)^2` is exactly `2^-150`, the midpoint between
    ///   zero and that subnormal, and a tie rounds to the even significand,
    ///   which is zero. The norm reaches zero
    ///   only when *every* component is under that bound — one ordinary
    ///   component is enough to keep it usable — and such a vector is then 1.0
    ///   from everything, itself included: a plausible-looking input with no
    ///   warning attached. Rescale before indexing if your embeddings live
    ///   down there. `vanedb/tests/fixtures/conformance/cosine_scale_invariance.tsv`
    ///   pins both ends.
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
