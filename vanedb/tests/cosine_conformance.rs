//! Cross-engine cosine cases from `conformance/cosine_scale_invariance.tsv`.
//!
//! Cosine distance is scale-invariant, but the zero-vector guard compared
//! `norm_a * norm_b` against a fixed epsilon. That product scales with the
//! fourth power of magnitude, so ordinary small vectors were classified as
//! zero (returning 1.0 for identical inputs) and large ones overflowed the
//! product to infinity, returning 1.0 as well (#40).

use vanedb::distance::{distance_fn, Metric};

const TOLERANCE: f32 = 1e-5;

fn cases() -> Vec<(f32, String, f32)> {
    let raw = include_str!("../../conformance/cosine_scale_invariance.tsv");
    raw.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(|line| {
            let mut field = line.split('\t');
            let scale: f32 = field.next().unwrap().parse().unwrap();
            let relation = field.next().unwrap().to_string();
            let expected: f32 = field.next().unwrap().parse().unwrap();
            (scale, relation, expected)
        })
        .collect()
}

fn vectors(scale: f32, relation: &str) -> (Vec<f32>, Vec<f32>) {
    match relation {
        "identical" => (vec![scale, 2.0 * scale], vec![scale, 2.0 * scale]),
        "opposite" => (vec![scale, 2.0 * scale], vec![-scale, -2.0 * scale]),
        "orthogonal" => (vec![scale, 0.0], vec![0.0, scale]),
        "zero" => (vec![0.0, 0.0], vec![scale, 2.0 * scale]),
        other => panic!("unknown relation: {other}"),
    }
}

#[test]
fn cosine_distance_matches_the_shared_cases() {
    let cosine = distance_fn(Metric::Cosine);
    let mut failures = Vec::new();

    for (scale, relation, expected) in cases() {
        let (a, b) = vectors(scale, &relation);
        let got = cosine(&a, &b);
        if (got - expected).abs() > TOLERANCE {
            failures.push(format!(
                "scale={scale:e} {relation}: expected {expected}, got {got}"
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "cosine cases failed:\n{}",
        failures.join("\n")
    );
}

/// The dispatcher picks a SIMD path at runtime; the scalar reference defines
/// the contract, so the two must agree on every shared case.
#[test]
fn dispatched_cosine_agrees_with_the_scalar_reference() {
    let cosine = distance_fn(Metric::Cosine);
    for (scale, relation, _) in cases() {
        let (a, b) = vectors(scale, &relation);
        let dispatched = cosine(&a, &b);
        let scalar = vanedb::distance::scalar::cosine_distance(&a, &b);
        assert!(
            (dispatched - scalar).abs() <= TOLERANCE,
            "scale={scale:e} {relation}: dispatched {dispatched} vs scalar {scalar}"
        );
    }
}

/// Wider vectors exercise the SIMD body rather than only the scalar tail.
#[test]
fn cosine_is_scale_invariant_for_simd_width_vectors() {
    let cosine = distance_fn(Metric::Cosine);
    for scale in [1e-18f32, 1e-4, 1.0, 1e4, 1e15] {
        let a: Vec<f32> = (0..128).map(|i| (i as f32 + 1.0) * scale).collect();
        let got = cosine(&a, &a);
        assert!(
            got.abs() <= TOLERANCE,
            "identical 128-d vectors at scale {scale:e} gave {got}, expected 0"
        );
    }
}

/// A finite input whose squared norm overflows f32 has no usable direction.
/// The documented answer is 1.0 — never NaN, which would otherwise leak a
/// non-finite distance into top-k ordering.
#[test]
fn cosine_returns_one_when_norms_overflow_rather_than_nan() {
    let cosine = distance_fn(Metric::Cosine);
    let a: Vec<f32> = vec![1e20; 128];
    let got = cosine(&a, &a);
    assert!(got.is_finite(), "expected a finite distance, got {got}");
    assert!((got - 1.0).abs() <= TOLERANCE, "expected 1.0, got {got}");
}
