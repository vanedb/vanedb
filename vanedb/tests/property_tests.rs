use proptest::prelude::*;
use vanedb::distance::{self, Metric};

fn arb_vector(dim: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(-100.0f32..100.0, dim)
}

proptest! {
    #[test]
    fn l2_is_non_negative(a in arb_vector(128), b in arb_vector(128)) {
        let dist_fn = distance::distance_fn(Metric::L2);
        let d = dist_fn(&a, &b);
        prop_assert!(d >= 0.0, "L2 distance was negative: {d}");
    }

    #[test]
    fn l2_self_distance_is_zero(a in arb_vector(128)) {
        let dist_fn = distance::distance_fn(Metric::L2);
        let d = dist_fn(&a, &a);
        prop_assert!(d.abs() < 1e-5, "L2 self-distance was {d}");
    }

    #[test]
    fn l2_is_symmetric(a in arb_vector(64), b in arb_vector(64)) {
        let dist_fn = distance::distance_fn(Metric::L2);
        let d_ab = dist_fn(&a, &b);
        let d_ba = dist_fn(&b, &a);
        prop_assert!((d_ab - d_ba).abs() < 1e-4,
            "L2 not symmetric: {d_ab} vs {d_ba}");
    }

    #[test]
    fn cosine_is_bounded(a in arb_vector(64), b in arb_vector(64)) {
        let dist_fn = distance::distance_fn(Metric::Cosine);
        let d = dist_fn(&a, &b);
        prop_assert!((0.0..=2.0).contains(&d),
            "Cosine distance out of [0, 2]: {d}");

        // `a` and `b` are drawn independently, so this never generates the
        // identical-vector case — which is the one that breaks without the
        // similarity clamp, because floating-point error pushes the cosine
        // just past 1.0 and the distance just below 0.
        let same = dist_fn(&a, &a);
        prop_assert!((0.0..=2.0).contains(&same),
            "Cosine self-distance out of [0, 2]: {same}");
        let scaled: Vec<f32> = a.iter().map(|x| x * 3.0).collect();
        let parallel = dist_fn(&a, &scaled);
        prop_assert!((0.0..=2.0).contains(&parallel),
            "Cosine distance to a scaled copy out of [0, 2]: {parallel}");
    }

    #[test]
    fn cosine_self_distance_near_zero(
        a in prop::collection::vec(0.1f32..100.0, 64)
    ) {
        let dist_fn = distance::distance_fn(Metric::Cosine);
        let d = dist_fn(&a, &a);
        // Not `d.abs()`. Removing the `clamp(-1.0, 1.0)` in the kernels makes
        // a self-distance go slightly *negative* — 3763 of 14000 cases, worst
        // -1.19e-7 — and `.abs()` hid exactly that sign. The lower bound is
        // what the clamp exists for.
        prop_assert!(
            (0.0..1e-5).contains(&d),
            "Cosine self-distance was {d}; a negative distance means the \
             similarity clamp is gone"
        );
    }

    #[test]
    fn dot_is_symmetric(a in arb_vector(64), b in arb_vector(64)) {
        let dist_fn = distance::distance_fn(Metric::Dot);
        let d_ab = dist_fn(&a, &b);
        let d_ba = dist_fn(&b, &a);
        prop_assert!((d_ab - d_ba).abs() < 1e-3,
            "Dot not symmetric: {d_ab} vs {d_ba}");
    }
}
