use proptest::prelude::*;
use vanedb::distance::{self, DistanceMetric};

fn arb_vector(dim: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(-100.0f32..100.0, dim)
}

proptest! {
    #[test]
    fn l2_is_non_negative(a in arb_vector(128), b in arb_vector(128)) {
        let dist_fn = distance::distance_fn(DistanceMetric::L2);
        let d = dist_fn(&a, &b);
        prop_assert!(d >= 0.0, "L2 distance was negative: {d}");
    }

    #[test]
    fn l2_self_distance_is_zero(a in arb_vector(128)) {
        let dist_fn = distance::distance_fn(DistanceMetric::L2);
        let d = dist_fn(&a, &a);
        prop_assert!(d.abs() < 1e-5, "L2 self-distance was {d}");
    }

    #[test]
    fn l2_is_symmetric(a in arb_vector(64), b in arb_vector(64)) {
        let dist_fn = distance::distance_fn(DistanceMetric::L2);
        let d_ab = dist_fn(&a, &b);
        let d_ba = dist_fn(&b, &a);
        prop_assert!((d_ab - d_ba).abs() < 1e-4,
            "L2 not symmetric: {d_ab} vs {d_ba}");
    }

    #[test]
    fn cosine_is_bounded(a in arb_vector(64), b in arb_vector(64)) {
        let dist_fn = distance::distance_fn(DistanceMetric::Cosine);
        let d = dist_fn(&a, &b);
        prop_assert!((0.0..=2.0).contains(&d),
            "Cosine distance out of [0, 2]: {d}");
    }

    #[test]
    fn cosine_self_distance_near_zero(
        a in prop::collection::vec(0.1f32..100.0, 64)
    ) {
        let dist_fn = distance::distance_fn(DistanceMetric::Cosine);
        let d = dist_fn(&a, &a);
        prop_assert!(d.abs() < 1e-5, "Cosine self-distance was {d}");
    }

    #[test]
    fn dot_is_symmetric(a in arb_vector(64), b in arb_vector(64)) {
        let dist_fn = distance::distance_fn(DistanceMetric::Dot);
        let d_ab = dist_fn(&a, &b);
        let d_ba = dist_fn(&b, &a);
        prop_assert!((d_ab - d_ba).abs() < 1e-3,
            "Dot not symmetric: {d_ab} vs {d_ba}");
    }
}
