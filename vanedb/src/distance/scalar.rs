//! Portable reference kernels. The SIMD paths must agree with these.
//!
//! `zip` stops at the shorter iterator, so a length mismatch truncates.
//! That is the contract the SIMD kernels bound themselves to.

/// Squared Euclidean distance. The square root is skipped: it does not
/// change the ordering, and every caller here only ranks.
pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

/// Cosine distance, `1 - cos(a, b)`, clamped to `[0, 2]`. Returns `1.0`
/// when a computed squared norm is zero — including by underflow — or
/// overflows to infinity, so a degenerate input ranks as orthogonal rather
/// than as NaN. See [`Metric::Cosine`](crate::Metric::Cosine).
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    // Normalise by each vector's own norm. `norm_a * norm_b` grows with the
    // fourth power of magnitude, so a fixed epsilon on that product classified
    // ordinary small vectors as zero, and the product overflowed to infinity
    // for large ones — both returned 1.0 for identical inputs (#40).
    //
    // Policy, shared with vanedb-cpp: a vector with no usable direction is
    // 1.0 away from everything, including itself. Three ways to have none —
    // a zero vector, a squared norm that overflowed f32, and one that
    // *underflowed* it, which needs every component small enough that its
    // square reaches zero. Finite inputs never yield NaN. `Metric::Cosine`
    // carries the bound and the rounding behind it; it is not repeated here,
    // because the figure was wrong once and had to be chased through six
    // files.
    // Multiplying the roots rather than rooting the product keeps the
    // denominator in range for both tiny and huge vectors.
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if !(denom > 0.0 && denom.is_finite()) {
        return 1.0;
    }
    let sim = dot / denom;
    1.0 - sim.clamp(-1.0, 1.0)
}

/// Negated dot product, so that lower still means nearer as it does for
/// the other metrics.
pub fn dot_distance(a: &[f32], b: &[f32]) -> f32 {
    -a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_identical_vectors() {
        let a = vec![1.0, 2.0, 3.0];
        assert_eq!(l2_squared(&a, &a), 0.0);
    }

    #[test]
    fn l2_known_result() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert_eq!(l2_squared(&a, &b), 2.0);
    }

    #[test]
    fn l2_single_dimension() {
        let a = vec![3.0];
        let b = vec![7.0];
        assert_eq!(l2_squared(&a, &b), 16.0);
    }

    #[test]
    fn cosine_identical_vectors() {
        let a = vec![1.0, 2.0, 3.0];
        assert!(cosine_distance(&a, &a).abs() < 1e-6);
    }

    #[test]
    fn cosine_orthogonal_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!((cosine_distance(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_opposite_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        assert!((cosine_distance(&a, &b) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_zero_vector_returns_one() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];
        assert_eq!(cosine_distance(&a, &b), 1.0);
    }

    #[test]
    fn dot_known_result() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert_eq!(dot_distance(&a, &b), -32.0);
    }

    #[test]
    fn dot_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert_eq!(dot_distance(&a, &b), 0.0);
    }
}
