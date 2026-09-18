//! Exact top-k in f64 (RFC 0003) and recall@k.

#[derive(Clone, Copy, Debug)]
pub enum GtMetric {
    L2,
    Cosine,
}

fn l2_sq_f64(a: &[f32], b: &[f32]) -> f64 {
    let mut sum = 0.0f64;
    for (&x, &y) in a.iter().zip(b) {
        let d = f64::from(x) - f64::from(y);
        sum += d * d;
    }
    sum
}

fn cosine_distance_f64(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        let xf = f64::from(x);
        let yf = f64::from(y);
        dot += xf * yf;
        na += xf * xf;
        nb += yf * yf;
    }
    let denom = na.sqrt() * nb.sqrt();
    if !(denom > 0.0 && denom.is_finite()) {
        return 1.0;
    }
    1.0 - (dot / denom).clamp(-1.0, 1.0)
}

/// Exact top-k ids for one query against row-major `vectors`.
pub fn brute_force_topk_f64(
    vectors: &[f32],
    ids: &[u64],
    dim: usize,
    query: &[f32],
    k: usize,
    metric: GtMetric,
) -> Vec<u64> {
    let distance: fn(&[f32], &[f32]) -> f64 = match metric {
        GtMetric::L2 => l2_sq_f64,
        GtMetric::Cosine => cosine_distance_f64,
    };
    let mut scored: Vec<(f64, u64)> = ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (distance(query, &vectors[i * dim..(i + 1) * dim]), id))
        .collect();
    scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)));
    scored.into_iter().take(k).map(|(_, id)| id).collect()
}

pub fn recall_at_k(returned: &[u64], truth: &[u64]) -> f64 {
    if truth.is_empty() {
        return 1.0;
    }
    let hits = returned.iter().filter(|id| truth.contains(id)).count();
    hits as f64 / truth.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topk_l2() {
        let vectors = [0.0f32, 0.0, 1.0, 1.0, 5.0, 5.0];
        let ids = [0u64, 1, 2];
        let q = [0.1f32, 0.1];
        let truth = brute_force_topk_f64(&vectors, &ids, 2, &q, 2, GtMetric::L2);
        assert_eq!(truth, vec![0, 1]);
        assert!((recall_at_k(&[0, 1], &truth) - 1.0).abs() < 1e-12);
    }
}
