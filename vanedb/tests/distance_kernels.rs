//! The dispatched kernel must agree with the scalar reference at every length.
//!
//! A single dimension cannot reach every loop tier: the widest unrolled loop
//! consumes its multiple and the scalar remainder takes the rest, so a length
//! like 33 skips the intermediate loops entirely. Sweeping the lengths is what
//! makes the "keep NEON, AVX2 and scalar in sync" invariant checkable.

use vanedb::distance::{distance_fn, scalar};
use vanedb::Metric;

type Kernel = fn(&[f32], &[f32]) -> f32;

const METRICS: [(&str, Metric, Kernel); 3] = [
    ("l2", Metric::L2, scalar::l2_squared),
    ("cosine", Metric::Cosine, scalar::cosine_distance),
    ("dot", Metric::Dot, scalar::dot_distance),
];

/// Values with mixed signs and magnitudes, so a dropped lane changes the sum.
fn ramp(n: usize, phase: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32) * 0.37 + phase).sin() * 3.0)
        .collect()
}

#[test]
fn dispatched_kernels_match_scalar_at_every_length() {
    // 0..=80 crosses every tier boundary of both the NEON (16/4) and the
    // AVX2 (32/8) kernels, for the metrics that halve those widths too.
    for n in 0..=80 {
        let a = ramp(n, 0.0);
        let b = ramp(n, 1.7);
        for (name, metric, reference) in METRICS {
            let got = distance_fn(metric)(&a, &b);
            let want = reference(&a, &b);
            assert!(
                (got - want).abs() < 1e-4,
                "{name} n={n}: dispatched={got}, scalar={want}"
            );
        }
    }
}

#[test]
fn mismatched_lengths_truncate_to_the_shorter_slice() {
    // These are safe public functions, so every input has to be defined
    // behaviour. The scalar reference zips, which truncates; the SIMD paths
    // took their trip count from `a` alone and read past the end of `b`.
    for (long, short) in [(64, 4), (33, 1), (16, 15), (40, 8), (8, 0)] {
        let a = ramp(long, 0.0);
        let b = ramp(short, 1.7);
        for (name, metric, reference) in METRICS {
            let got = distance_fn(metric)(&a, &b);
            let want = reference(&a[..short], &b);
            assert!(
                (got - want).abs() < 1e-4,
                "{name} a={long} b={short}: dispatched={got}, truncated-scalar={want}"
            );
            // And symmetrically, with the short slice first.
            let got = distance_fn(metric)(&b, &a);
            let want = reference(&b, &a[..short]);
            assert!(
                (got - want).abs() < 1e-4,
                "{name} a={short} b={long}: dispatched={got}, truncated-scalar={want}"
            );
        }
    }
}
