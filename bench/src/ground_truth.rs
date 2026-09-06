//! Brute-force top-k (the reference) and recall@k for approximate results.

fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
    for (x, y) in a.iter().zip(b) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = na.sqrt() * nb.sqrt();
    if !(denom > 0.0 && denom.is_finite()) {
        return 1.0;
    }
    1.0 - (dot / denom).clamp(-1.0, 1.0)
}

fn neg_dot(a: &[f32], b: &[f32]) -> f32 {
    -a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>()
}

/// Exact top-k ids for one query against row-major `vectors` (n * dim).
///
/// `metric` uses the C ABI's encoding — 0 = L2, 1 = cosine, 2 = dot — and must
/// be the same value the index under test was built with. Ground truth
/// computed under a different metric ranks a different set, so the recall
/// figure would be meaningless rather than merely wrong.
pub fn brute_force_topk(
    vectors: &[f32],
    ids: &[u64],
    dim: usize,
    query: &[f32],
    k: usize,
    metric: u32,
) -> Vec<u64> {
    let distance: fn(&[f32], &[f32]) -> f32 = match metric {
        0 => l2_sq,
        1 => cosine,
        2 => neg_dot,
        other => panic!("unknown metric {other}; expected 0 (L2), 1 (cosine) or 2 (dot)"),
    };
    let mut scored: Vec<(f32, u64)> = ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (distance(query, &vectors[i * dim..(i + 1) * dim]), id))
        .collect();
    scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    scored.iter().take(k).map(|&(_, id)| id).collect()
}

/// recall@k = |returned ∩ truth| / k, for one query.
pub fn recall_at_k(returned: &[u64], truth: &[u64]) -> f32 {
    if truth.is_empty() {
        return 1.0;
    }
    let hits = returned.iter().filter(|id| truth.contains(id)).count();
    hits as f32 / truth.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn topk_and_recall() {
        let vectors = [0.0, 0.0, 1.0, 1.0, 5.0, 5.0]; // ids 0,1,2
        let ids = [0u64, 1, 2];
        let q = [0.1, 0.1];
        let truth = brute_force_topk(&vectors, &ids, 2, &q, 2, 0);
        assert_eq!(truth, vec![0, 1]); // nearest two
        assert!((recall_at_k(&[0, 1], &truth) - 1.0).abs() < 1e-6);
        assert!((recall_at_k(&[0, 2], &truth) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn each_metric_ranks_by_its_own_distance() {
        // One corpus, three metrics, three different winners, no ties — so
        // every arm is load-bearing. A corpus where two metrics agree proves
        // only one of them, and an exact tie proves nothing at all, since the
        // winner is then decided by sort stability.
        let vectors = [1.1, 0.3, 0.5, 0.0, 8.0, 1.0];
        let ids = [0u64, 1, 2];
        let q = [1.0, 0.0];
        // L2 = 0.10, 0.25, 50: nearest in absolute position.
        assert_eq!(brute_force_topk(&vectors, &ids, 2, &q, 1, 0), vec![0]);
        // cosine = 0.035, 0.0, 0.008: the exactly-parallel one, whatever its length.
        assert_eq!(brute_force_topk(&vectors, &ids, 2, &q, 1, 1), vec![1]);
        // dot = -1.1, -0.5, -8.0: magnitude wins.
        assert_eq!(brute_force_topk(&vectors, &ids, 2, &q, 1, 2), vec![2]);
    }

    #[test]
    fn kernels_match_the_engine_reference() {
        // These are a second copy of the engine's distance functions. If they
        // drift, every recall figure silently measures the wrong thing, and
        // nothing else in this harness would notice.
        let a = [0.5f32, -1.5, 2.0, 0.0, 3.25];
        let b = [1.0f32, 2.0, -0.5, 0.0, 1.5];
        let zero = [0.0f32; 5];
        // Magnitudes where the squared norms are finite but their product is
        // not, and where it underflows to zero. That is the case the engine's
        // cosine is written around (#40): multiplying the roots stays in range
        // where rooting the product does not. Ordinary values cannot tell the
        // two formulations apart.
        let huge = [1.0e15f32; 5];
        let tiny = [1.0e-15f32; 5];
        for (x, y) in [
            (&a, &b),
            (&a, &a),
            (&a, &zero),
            (&zero, &zero),
            (&huge, &huge),
            (&tiny, &tiny),
            (&huge, &tiny),
        ] {
            assert_eq!(l2_sq(x, y), vanedb::distance::scalar::l2_squared(x, y));
            assert_eq!(
                cosine(x, y),
                vanedb::distance::scalar::cosine_distance(x, y)
            );
            assert_eq!(neg_dot(x, y), vanedb::distance::scalar::dot_distance(x, y));
        }
    }
}
