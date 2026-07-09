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
        // Four independent accumulators hide the fused multiply-add latency
        // (~4 cycles); a single-accumulator loop is latency-bound at one
        // vector per FMA latency regardless of ALU width. Mirrors
        // vanedb-cpp src/core/distance.h.
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);
        while i + 16 <= n {
            let p = a.as_ptr().add(i);
            let q = b.as_ptr().add(i);
            let d0 = vsubq_f32(vld1q_f32(p), vld1q_f32(q));
            let d1 = vsubq_f32(vld1q_f32(p.add(4)), vld1q_f32(q.add(4)));
            let d2 = vsubq_f32(vld1q_f32(p.add(8)), vld1q_f32(q.add(8)));
            let d3 = vsubq_f32(vld1q_f32(p.add(12)), vld1q_f32(q.add(12)));
            acc0 = vmlaq_f32(acc0, d0, d0);
            acc1 = vmlaq_f32(acc1, d1, d1);
            acc2 = vmlaq_f32(acc2, d2, d2);
            acc3 = vmlaq_f32(acc3, d3, d3);
            i += 16;
        }
        let mut acc = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
        while i + 4 <= n {
            let d = vsubq_f32(vld1q_f32(a.as_ptr().add(i)), vld1q_f32(b.as_ptr().add(i)));
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
        // Two-way unroll on top of the three naturally independent chains.
        let mut vdot0 = vdupq_n_f32(0.0);
        let mut vna0 = vdupq_n_f32(0.0);
        let mut vnb0 = vdupq_n_f32(0.0);
        let mut vdot1 = vdupq_n_f32(0.0);
        let mut vna1 = vdupq_n_f32(0.0);
        let mut vnb1 = vdupq_n_f32(0.0);
        while i + 8 <= n {
            let p = a.as_ptr().add(i);
            let q = b.as_ptr().add(i);
            let va0 = vld1q_f32(p);
            let vb0 = vld1q_f32(q);
            let va1 = vld1q_f32(p.add(4));
            let vb1 = vld1q_f32(q.add(4));
            vdot0 = vmlaq_f32(vdot0, va0, vb0);
            vna0 = vmlaq_f32(vna0, va0, va0);
            vnb0 = vmlaq_f32(vnb0, vb0, vb0);
            vdot1 = vmlaq_f32(vdot1, va1, vb1);
            vna1 = vmlaq_f32(vna1, va1, va1);
            vnb1 = vmlaq_f32(vnb1, vb1, vb1);
            i += 8;
        }
        let mut vdot = vaddq_f32(vdot0, vdot1);
        let mut vna = vaddq_f32(vna0, vna1);
        let mut vnb = vaddq_f32(vnb0, vnb1);
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
        // Same latency-hiding unroll as l2_squared.
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);
        while i + 16 <= n {
            let p = a.as_ptr().add(i);
            let q = b.as_ptr().add(i);
            acc0 = vmlaq_f32(acc0, vld1q_f32(p), vld1q_f32(q));
            acc1 = vmlaq_f32(acc1, vld1q_f32(p.add(4)), vld1q_f32(q.add(4)));
            acc2 = vmlaq_f32(acc2, vld1q_f32(p.add(8)), vld1q_f32(q.add(8)));
            acc3 = vmlaq_f32(acc3, vld1q_f32(p.add(12)), vld1q_f32(q.add(12)));
            i += 16;
        }
        let mut acc = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
        while i + 4 <= n {
            acc = vmlaq_f32(
                acc,
                vld1q_f32(a.as_ptr().add(i)),
                vld1q_f32(b.as_ptr().add(i)),
            );
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
