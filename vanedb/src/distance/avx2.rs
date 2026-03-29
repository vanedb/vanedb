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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let mut i = 0;
    let mut acc = _mm256_setzero_ps();

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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let mut i = 0;
    let mut vdot = _mm256_setzero_ps();
    let mut vna = _mm256_setzero_ps();
    let mut vnb = _mm256_setzero_ps();

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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let mut i = 0;
    let mut acc = _mm256_setzero_ps();

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
