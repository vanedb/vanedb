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

/// Relative bound: an absolute epsilon shrinks in headroom as `n` grows,
/// and the AVX2 tiers accumulate in a different order from NEON's.
fn close(got: f32, want: f32) -> bool {
    (got - want).abs() <= 1e-5 * want.abs().max(1.0)
}

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
                close(got, want),
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
                close(got, want),
                "{name} a={long} b={short}: dispatched={got}, truncated-scalar={want}"
            );
            // And symmetrically, with the short slice first.
            let got = distance_fn(metric)(&b, &a);
            let want = reference(&b, &a[..short]);
            assert!(
                close(got, want),
                "{name} a={short} b={long}: dispatched={got}, truncated-scalar={want}"
            );
        }
    }
}

#[test]
fn dot_recovers_cancelling_overflow_on_every_kernel() {
    for n in [2, 8, 16, 32, 33, 64, 80] {
        let a = vec![3e38; n];
        for grouped in [false, true] {
            let b: Vec<f32> = (0..n)
                .map(|i| {
                    if i >= n / 2 * 2 {
                        0.0
                    } else if if grouped { i < n / 2 } else { i % 2 == 0 } {
                        2.0
                    } else {
                        -2.0
                    }
                })
                .collect();
            for (name, got) in [
                ("scalar", scalar::dot_distance(&a, &b)),
                ("dispatched", distance_fn(Metric::Dot)(&a, &b)),
            ] {
                assert_eq!(got, 0.0, "{name} n={n} grouped={grouped}");
            }
        }
        for sign in [-1.0, 1.0] {
            let b = vec![2.0 * sign; n];
            assert_eq!(scalar::dot_distance(&a, &b), f32::NEG_INFINITY);
            assert_eq!(distance_fn(Metric::Dot)(&a, &b), f32::NEG_INFINITY);
        }
    }
    // Each product is finite; overflow is only in some reduction orders.
    for b in [
        [2., 2., 2., 2., -2., -2., -2., -2.],
        [2., -2., 2., -2., 2., -2., 2., -2.],
    ] {
        assert_eq!(scalar::dot_distance(&[1e38; 8], &b), 0.0);
        assert_eq!(distance_fn(Metric::Dot)(&[1e38; 8], &b), 0.0);
    }
}
