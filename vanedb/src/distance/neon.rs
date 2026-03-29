#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use crate::distance::COSINE_EPSILON;

#[cfg(target_arch = "aarch64")]
pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let mut i = 0;

    // SAFETY: NEON is always available on aarch64.
    // Pointer arithmetic stays within slice bounds (i + 4 <= n).
    let mut sum = unsafe {
        let mut acc = vdupq_n_f32(0.0);
        while i + 4 <= n {
            let va = vld1q_f32(a.as_ptr().add(i));
            let vb = vld1q_f32(b.as_ptr().add(i));
            let d = vsubq_f32(va, vb);
            acc = vmlaq_f32(acc, d, d);
            i += 4;
        }
        vaddvq_f32(acc)
    };

    // Scalar remainder
    while i < n {
        let d = a[i] - b[i];
        sum += d * d;
        i += 1;
    }
    sum
}

#[cfg(target_arch = "aarch64")]
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let mut i = 0;

    let (mut dot, mut norm_a, mut norm_b) = unsafe {
        let mut vdot = vdupq_n_f32(0.0);
        let mut vna = vdupq_n_f32(0.0);
        let mut vnb = vdupq_n_f32(0.0);
        while i + 4 <= n {
            let va = vld1q_f32(a.as_ptr().add(i));
            let vb = vld1q_f32(b.as_ptr().add(i));
            vdot = vmlaq_f32(vdot, va, vb);
            vna = vmlaq_f32(vna, va, va);
            vnb = vmlaq_f32(vnb, vb, vb);
            i += 4;
        }
        (vaddvq_f32(vdot), vaddvq_f32(vna), vaddvq_f32(vnb))
    };

    while i < n {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
        i += 1;
    }

    let denom = norm_a * norm_b;
    if denom < COSINE_EPSILON {
        return 1.0;
    }
    let sim = dot / denom.sqrt();
    1.0 - sim.clamp(-1.0, 1.0)
}

#[cfg(target_arch = "aarch64")]
pub fn dot_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let mut i = 0;

    let mut sum = unsafe {
        let mut acc = vdupq_n_f32(0.0);
        while i + 4 <= n {
            let va = vld1q_f32(a.as_ptr().add(i));
            let vb = vld1q_f32(b.as_ptr().add(i));
            acc = vmlaq_f32(acc, va, vb);
            i += 4;
        }
        vaddvq_f32(acc)
    };

    while i < n {
        sum += a[i] * b[i];
        i += 1;
    }
    -sum
}

#[cfg(test)]
#[cfg(target_arch = "aarch64")]
mod tests {
    use super::*;

    #[test]
    fn neon_l2_matches_scalar() {
        // 33 elements: tests both SIMD loop (8 iterations) and scalar remainder (1 element)
        let a: Vec<f32> = (0..33).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..33).map(|i| (33 - i) as f32 * 0.1).collect();
        let neon_result = l2_squared(&a, &b);
        let scalar_result = crate::distance::scalar::l2_squared(&a, &b);
        assert!(
            (neon_result - scalar_result).abs() < 1e-4,
            "neon={neon_result}, scalar={scalar_result}"
        );
    }

    #[test]
    fn neon_cosine_matches_scalar() {
        let a: Vec<f32> = (0..33).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..33).map(|i| (33 - i) as f32 * 0.1).collect();
        let neon_result = cosine_distance(&a, &b);
        let scalar_result = crate::distance::scalar::cosine_distance(&a, &b);
        assert!(
            (neon_result - scalar_result).abs() < 1e-4,
            "neon={neon_result}, scalar={scalar_result}"
        );
    }

    #[test]
    fn neon_dot_matches_scalar() {
        let a: Vec<f32> = (0..33).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..33).map(|i| (33 - i) as f32 * 0.1).collect();
        let neon_result = dot_distance(&a, &b);
        let scalar_result = crate::distance::scalar::dot_distance(&a, &b);
        assert!(
            (neon_result - scalar_result).abs() < 1e-4,
            "neon={neon_result}, scalar={scalar_result}"
        );
    }

    #[test]
    fn neon_l2_small_vector() {
        // 3 elements -- all scalar remainder, no SIMD loop iterations
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert_eq!(l2_squared(&a, &b), 27.0);
    }
}
