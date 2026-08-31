//! Deterministic vector generation via splitmix64 — no rand dependency, so the
//! harness is reproducible regardless of any implementation's RNG choices.

pub struct Workload {
    pub dim: usize,
    pub vectors: Vec<f32>, // row-major, n * dim
    pub ids: Vec<u64>,
    pub queries: Vec<f32>, // row-major, n_queries * dim
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

fn unit_f32(state: &mut u64) -> f32 {
    // 24-bit mantissa → [0,1)
    (splitmix64(state) >> 40) as f32 / (1u64 << 24) as f32
}

pub fn generate(seed: u64, dim: usize, n: usize, n_queries: usize) -> Workload {
    let mut s = seed;
    let mut vectors = Vec::with_capacity(n * dim);
    for _ in 0..n * dim {
        vectors.push(unit_f32(&mut s));
    }
    let ids = (0..n as u64).collect();
    let mut queries = Vec::with_capacity(n_queries * dim);
    for _ in 0..n_queries * dim {
        queries.push(unit_f32(&mut s));
    }
    Workload {
        dim,
        vectors,
        ids,
        queries,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn deterministic_and_shaped() {
        let a = generate(42, 8, 100, 5);
        let b = generate(42, 8, 100, 5);
        assert_eq!(a.vectors, b.vectors); // same seed => identical
        assert_eq!(a.vectors.len(), 800);
        assert_eq!(a.queries.len(), 40);
        assert_eq!(a.ids.len(), 100);
        let c = generate(43, 8, 100, 5);
        assert_ne!(a.vectors, c.vectors); // different seed => different
    }
}
