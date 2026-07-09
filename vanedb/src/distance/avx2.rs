#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::distance::COSINE_EPSILON;

/// Horizontal sum of 8 floats in a 256-bit register.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hsum_avx2(v: __m256) -> f32 {
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps(v, 1);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sum64 = _mm_add_ps(sum128, shuf);
    _mm_cvtss_f32(_mm_add_ss(sum64, _mm_movehl_ps(shuf, sum64)))
}

/// Squared L2 distance using AVX2+FMA intrinsics.
///
/// # Safety
/// Caller must ensure the CPU supports AVX2 and FMA (use `is_x86_feature_detected!`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let mut i = 0;
    // Four independent accumulators hide the FMA latency; a single-acc
    // loop is latency-bound. Mirrors vanedb-cpp src/core/distance.h.
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();

    while i + 32 <= n {
        let p = a.as_ptr().add(i);
        let q = b.as_ptr().add(i);
        let d0 = _mm256_sub_ps(_mm256_loadu_ps(p), _mm256_loadu_ps(q));
        let d1 = _mm256_sub_ps(_mm256_loadu_ps(p.add(8)), _mm256_loadu_ps(q.add(8)));
        let d2 = _mm256_sub_ps(_mm256_loadu_ps(p.add(16)), _mm256_loadu_ps(q.add(16)));
        let d3 = _mm256_sub_ps(_mm256_loadu_ps(p.add(24)), _mm256_loadu_ps(q.add(24)));
        acc0 = _mm256_fmadd_ps(d0, d0, acc0);
        acc1 = _mm256_fmadd_ps(d1, d1, acc1);
        acc2 = _mm256_fmadd_ps(d2, d2, acc2);
        acc3 = _mm256_fmadd_ps(d3, d3, acc3);
        i += 32;
    }
    let mut acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));

    while i + 8 <= n {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        let d = _mm256_sub_ps(va, vb);
        acc = _mm256_fmadd_ps(d, d, acc);
        i += 8;
    }

    let mut sum = hsum_avx2(acc);
    while i < n {
        let d = a[i] - b[i];
        sum += d * d;
        i += 1;
    }
    sum
}

/// Cosine distance using AVX2+FMA intrinsics.
///
/// # Safety
/// Caller must ensure the CPU supports AVX2 and FMA (use `is_x86_feature_detected!`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let mut i = 0;
    // Two-way unroll on top of the three naturally independent chains.
    let mut vdot0 = _mm256_setzero_ps();
    let mut vna0 = _mm256_setzero_ps();
    let mut vnb0 = _mm256_setzero_ps();
    let mut vdot1 = _mm256_setzero_ps();
    let mut vna1 = _mm256_setzero_ps();
    let mut vnb1 = _mm256_setzero_ps();

    while i + 16 <= n {
        let p = a.as_ptr().add(i);
        let q = b.as_ptr().add(i);
        let va0 = _mm256_loadu_ps(p);
        let vb0 = _mm256_loadu_ps(q);
        let va1 = _mm256_loadu_ps(p.add(8));
        let vb1 = _mm256_loadu_ps(q.add(8));
        vdot0 = _mm256_fmadd_ps(va0, vb0, vdot0);
        vna0 = _mm256_fmadd_ps(va0, va0, vna0);
        vnb0 = _mm256_fmadd_ps(vb0, vb0, vnb0);
        vdot1 = _mm256_fmadd_ps(va1, vb1, vdot1);
        vna1 = _mm256_fmadd_ps(va1, va1, vna1);
        vnb1 = _mm256_fmadd_ps(vb1, vb1, vnb1);
        i += 16;
    }
    let mut vdot = _mm256_add_ps(vdot0, vdot1);
    let mut vna = _mm256_add_ps(vna0, vna1);
    let mut vnb = _mm256_add_ps(vnb0, vnb1);

    while i + 8 <= n {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        vdot = _mm256_fmadd_ps(va, vb, vdot);
        vna = _mm256_fmadd_ps(va, va, vna);
        vnb = _mm256_fmadd_ps(vb, vb, vnb);
        i += 8;
    }

    let mut dot = hsum_avx2(vdot);
    let mut norm_a = hsum_avx2(vna);
    let mut norm_b = hsum_avx2(vnb);

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

/// Negative dot product distance using AVX2+FMA intrinsics.
///
/// # Safety
/// Caller must ensure the CPU supports AVX2 and FMA (use `is_x86_feature_detected!`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let mut i = 0;
    // Same latency-hiding unroll as l2_squared.
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();

    while i + 32 <= n {
        let p = a.as_ptr().add(i);
        let q = b.as_ptr().add(i);
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(p), _mm256_loadu_ps(q), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(p.add(8)), _mm256_loadu_ps(q.add(8)), acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(p.add(16)), _mm256_loadu_ps(q.add(16)), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(p.add(24)), _mm256_loadu_ps(q.add(24)), acc3);
        i += 32;
    }
    let mut acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));

    while i + 8 <= n {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        acc = _mm256_fmadd_ps(va, vb, acc);
        i += 8;
    }

    let mut sum = hsum_avx2(acc);
    while i < n {
        sum += a[i] * b[i];
        i += 1;
    }
    -sum
}

#[cfg(test)]
#[cfg(target_arch = "x86_64")]
mod tests {
    use super::*;

    #[test]
    fn avx2_l2_matches_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return; // skip on CPUs without AVX2+FMA
        }
        let a: Vec<f32> = (0..33).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..33).map(|i| (33 - i) as f32 * 0.1).collect();
        let avx2_result = unsafe { l2_squared(&a, &b) };
        let scalar_result = crate::distance::scalar::l2_squared(&a, &b);
        assert!(
            (avx2_result - scalar_result).abs() < 1e-4,
            "avx2={avx2_result}, scalar={scalar_result}"
        );
    }

    #[test]
    fn avx2_cosine_matches_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        let a: Vec<f32> = (0..33).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..33).map(|i| (33 - i) as f32 * 0.1).collect();
        let avx2_result = unsafe { cosine_distance(&a, &b) };
        let scalar_result = crate::distance::scalar::cosine_distance(&a, &b);
        assert!(
            (avx2_result - scalar_result).abs() < 1e-4,
            "avx2={avx2_result}, scalar={scalar_result}"
        );
    }

    #[test]
    fn avx2_dot_matches_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        let a: Vec<f32> = (0..33).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..33).map(|i| (33 - i) as f32 * 0.1).collect();
        let avx2_result = unsafe { dot_distance(&a, &b) };
        let scalar_result = crate::distance::scalar::dot_distance(&a, &b);
        assert!(
            (avx2_result - scalar_result).abs() < 1e-4,
            "avx2={avx2_result}, scalar={scalar_result}"
        );
    }
}
