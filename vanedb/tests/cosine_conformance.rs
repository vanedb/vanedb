//! Cross-engine cosine cases from `tests/fixtures/conformance/cosine_scale_invariance.tsv`.
//!
//! Cosine distance is scale-invariant, but the zero-vector guard compared
//! `norm_a * norm_b` against a fixed epsilon. That product scales with the
//! fourth power of magnitude, so ordinary small vectors were classified as
//! zero (returning 1.0 for identical inputs) and large ones overflowed the
//! product to infinity, returning 1.0 as well (#40).

use vanedb::distance::{distance_fn, Metric};

const TOLERANCE: f32 = 1e-5;

fn cases() -> Vec<(f32, String, f32)> {
    let raw = include_str!("fixtures/conformance/cosine_scale_invariance.tsv");
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

/// The underflow end of the "no usable direction" rule is a property of the
/// whole vector, not of any one component.
///
/// The rustdoc quotes a per-component bound (`2^-75`, about 2.6e-23) because
/// that is where a single square reaches zero. It would be easy to read that
/// as "a small component poisons the vector", which is not what the kernel
/// does: the norm is a sum, so one ordinary component keeps it usable no
/// matter how small the rest are. Both directions are asserted here, because
/// prose is what drifted last time.
///
/// The magnitudes are deep inside each region rather than at the boundary.
/// The exact boundary depends on subnormal handling — flush-to-zero would
/// move it — and the rule under test holds either way.
#[test]
fn cosine_underflows_only_when_every_component_is_below_the_bound() {
    let cosine = distance_fn(Metric::Cosine);

    // Every component far below the bound: no usable direction, so 1.0 even
    // against itself. This is the case `cosine_scale_invariance.tsv` pins.
    for all_tiny in [vec![1e-30f32, 1e-30], vec![1e-30; 128], vec![1e-30, 1e-24]] {
        assert!(
            (cosine(&all_tiny, &all_tiny) - 1.0).abs() <= TOLERANCE,
            "a vector with every component under the bound must be 1.0 from \
             itself, got {}",
            cosine(&all_tiny, &all_tiny)
        );
    }

    // One ordinary component is enough. A vector that is mostly negligible is
    // still a direction, and must behave like one.
    for mut mixed in [vec![1e-30f32; 128], vec![1e-30f32, 1e-30]] {
        mixed[0] = 1.0;
        assert!(
            cosine(&mixed, &mixed).abs() <= TOLERANCE,
            "one ordinary component must keep the norm usable, got {}",
            cosine(&mixed, &mixed)
        );
        // And it still ranks against an unrelated vector rather than
        // collapsing to the degenerate answer.
        let mut other = vec![0.0f32; mixed.len()];
        other[0] = -1.0;
        assert!(
            (cosine(&mixed, &other) - 2.0).abs() <= TOLERANCE,
            "an anti-parallel pair must be 2.0 apart, got {}",
            cosine(&mixed, &other)
        );
    }
}

/// The overflow end is a property of the norm too, and the many-component case
/// is the one a per-component reading misses.
#[test]
fn cosine_overflows_on_the_norm_not_on_any_single_component() {
    let cosine = distance_fn(Metric::Cosine);
    // 1e19 is below sqrt(f32::MAX) (~1.845e19), so no single square overflows
    // — but 128 of them sum past f32::MAX, and the vector has no usable
    // direction after all.
    let many = vec![1e19f32; 128];
    assert!(
        (cosine(&many, &many) - 1.0).abs() <= TOLERANCE,
        "a norm that overflows only in the sum must still be 1.0, got {}",
        cosine(&many, &many)
    );
    // The same magnitude in a short vector keeps its direction.
    let few = vec![1e19f32, 0.0];
    assert!(
        cosine(&few, &few).abs() <= TOLERANCE,
        "a component under the bound must not be treated as an overflow, got {}",
        cosine(&few, &few)
    );
}
