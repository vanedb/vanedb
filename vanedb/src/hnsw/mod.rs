use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};

use parking_lot::RwLock;
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::distance::{distance_fn, DistanceFn, DistanceMetric};
use crate::error::{Result, VaneError};
use crate::store::SearchResult;

/// Wrapper for f32 that implements Ord (needed for BinaryHeap).
#[derive(Debug, Clone, Copy, PartialEq)]
struct FloatOrd(f32);

impl Eq for FloatOrd {}

impl PartialOrd for FloatOrd {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FloatOrd {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

const MAX_LEVEL: i32 = 32;
const MIN_LEVEL_RANDOM: f64 = 1e-9;

pub struct HnswIndex {
    pub(super) dim: usize,
    pub(super) metric: DistanceMetric,
    pub(super) dist_fn: DistanceFn,
    pub(super) max_elements: usize,
    pub(super) m: usize,
    pub(super) m_max: usize,
    pub(super) m_max0: usize,
    pub(super) ef_construction: usize,
    pub(super) ef_search: AtomicUsize,
    pub(super) mult: f64,
    pub(super) inner: RwLock<Inner>,
}

pub(super) struct Inner {
    pub(super) vectors: Vec<f32>,
    pub(super) ext_ids: Vec<u64>,
    pub(super) id_map: HashMap<u64, usize>,
    pub(super) levels: Vec<i32>,
    pub(super) neighbors: Vec<Vec<Vec<usize>>>,
    pub(super) entry_point: Option<usize>,
    pub(super) max_level: i32,
    pub(super) count: usize,
    pub(super) rng: StdRng,
}

pub struct HnswIndexBuilder {
    dim: usize,
    metric: DistanceMetric,
    capacity: usize,
    m: usize,
    ef_construction: usize,
    seed: u64,
}

impl HnswIndex {
    pub fn builder(dim: usize, metric: DistanceMetric) -> HnswIndexBuilder {
        HnswIndexBuilder {
            dim,
            metric,
            capacity: 100_000,
            m: 16,
            ef_construction: 200,
            seed: 42,
        }
    }

    pub fn size(&self) -> usize {
        self.inner.read().count
    }

    pub fn is_empty(&self) -> bool {
        self.size() == 0
    }

    pub fn capacity(&self) -> usize {
        self.max_elements
    }

    pub fn dimension(&self) -> usize {
        self.dim
    }

    pub fn metric(&self) -> DistanceMetric {
        self.metric
    }

    pub fn contains(&self, id: u64) -> bool {
        self.inner.read().id_map.contains_key(&id)
    }

    pub fn get_vector(&self, id: u64) -> Result<Vec<f32>> {
        let inner = self.inner.read();
        let &iid = inner.id_map.get(&id).ok_or(VaneError::NotFound { id })?;
        let start = iid * self.dim;
        Ok(inner.vectors[start..start + self.dim].to_vec())
    }

    pub fn set_ef_search(&self, ef: usize) {
        self.ef_search.store(ef, Ordering::Relaxed);
    }

    pub fn get_ef_search(&self) -> usize {
        self.ef_search.load(Ordering::Relaxed)
    }
}

impl HnswIndexBuilder {
    pub fn capacity(mut self, cap: usize) -> Self {
        self.capacity = cap;
        self
    }

    pub fn m(mut self, m: usize) -> Self {
        self.m = m;
        self
    }

    pub fn ef_construction(mut self, ef: usize) -> Self {
        self.ef_construction = ef;
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn build(self) -> Result<HnswIndex> {
        if self.dim == 0 {
            return Err(VaneError::EmptyVector);
        }
        if self.capacity == 0 {
            return Err(VaneError::InvalidParameter("capacity must be > 0"));
        }
        if self.m < 2 {
            return Err(VaneError::InvalidParameter("M must be >= 2"));
        }
        let ef_construction = self.ef_construction.max(self.m);
        let mult = if self.m > 1 {
            1.0 / (self.m as f64).ln()
        } else {
            1.0
        };
        Ok(HnswIndex {
            dim: self.dim,
            metric: self.metric,
            dist_fn: distance_fn(self.metric),
            max_elements: self.capacity,
            m: self.m,
            m_max: self.m,
            m_max0: self.m * 2,
            ef_construction,
            ef_search: AtomicUsize::new(50),
            mult,
            inner: RwLock::new(Inner {
                vectors: vec![0.0; self.capacity * self.dim],
                ext_ids: vec![0; self.capacity],
                id_map: HashMap::new(),
                levels: vec![0; self.capacity],
                neighbors: (0..self.capacity).map(|_| Vec::new()).collect(),
                entry_point: None,
                max_level: -1,
                count: 0,
                rng: StdRng::seed_from_u64(self.seed),
            }),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_defaults() {
        let idx = HnswIndex::builder(128, DistanceMetric::Cosine)
            .build()
            .unwrap();
        assert_eq!(idx.dimension(), 128);
        assert_eq!(idx.capacity(), 100_000);
        assert!(idx.is_empty());
        assert_eq!(idx.size(), 0);
        assert_eq!(idx.get_ef_search(), 50);
    }

    #[test]
    fn builder_custom_params() {
        let idx = HnswIndex::builder(64, DistanceMetric::L2)
            .capacity(1000)
            .m(32)
            .ef_construction(400)
            .seed(123)
            .build()
            .unwrap();
        assert_eq!(idx.capacity(), 1000);
    }

    #[test]
    fn builder_rejects_zero_dim() {
        assert!(HnswIndex::builder(0, DistanceMetric::L2).build().is_err());
    }

    #[test]
    fn builder_rejects_zero_capacity() {
        assert!(HnswIndex::builder(64, DistanceMetric::L2)
            .capacity(0)
            .build()
            .is_err());
    }

    #[test]
    fn builder_rejects_m_below_2() {
        assert!(HnswIndex::builder(64, DistanceMetric::L2)
            .m(1)
            .build()
            .is_err());
    }

    #[test]
    fn set_ef_search() {
        let idx = HnswIndex::builder(64, DistanceMetric::L2).build().unwrap();
        idx.set_ef_search(100);
        assert_eq!(idx.get_ef_search(), 100);
    }
}
