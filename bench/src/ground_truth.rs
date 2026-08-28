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

/// Exact top-k ids for one query against row-major `vectors` (n * dim).
pub fn brute_force_topk(
    vectors: &[f32],
    ids: &[u64],
    dim: usize,
    query: &[f32],
    k: usize,
) -> Vec<u64> {
    let mut scored: Vec<(f32, u64)> = ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (l2_sq(query, &vectors[i * dim..(i + 1) * dim]), id))
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
        let truth = brute_force_topk(&vectors, &ids, 2, &q, 2);
        assert_eq!(truth, vec![0, 1]); // nearest two
        assert!((recall_at_k(&[0, 1], &truth) - 1.0).abs() < 1e-6);
        assert!((recall_at_k(&[0, 2], &truth) - 0.5).abs() < 1e-6);
    }
}
