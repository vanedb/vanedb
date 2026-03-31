use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};

use parking_lot::RwLock;
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::distance::{distance_fn, DistanceFn, DistanceMetric};
use crate::error::{Result, VaneError};
use crate::store::SearchResult;

mod persistence;

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

    /// Insert a vector into the HNSW graph.
    pub fn add(&self, id: u64, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dim {
            return Err(VaneError::DimensionMismatch {
                expected: self.dim,
                got: vector.len(),
            });
        }

        let mut inner = self.inner.write();

        if inner.count >= self.max_elements {
            return Err(VaneError::IndexFull);
        }
        if inner.id_map.contains_key(&id) {
            return Err(VaneError::DuplicateId { id });
        }

        let iid = inner.count;
        inner.count += 1;

        // Copy vector data
        let start = iid * self.dim;
        inner.vectors[start..start + self.dim].copy_from_slice(vector);
        inner.ext_ids[iid] = id;
        inner.id_map.insert(id, iid);

        // Generate random level
        let level = Self::get_level(&mut inner.rng, self.mult);
        inner.levels[iid] = level;

        // Allocate neighbor lists for each layer
        inner.neighbors[iid] = (0..=level as usize).map(|_| Vec::new()).collect();

        // First vector: set as entry point and return
        if iid == 0 {
            inner.entry_point = Some(0);
            inner.max_level = level;
            return Ok(());
        }

        let mut cur_ep = inner.entry_point.unwrap();
        let cur_max_level = inner.max_level;

        // Greedy descent through upper layers (above new node's level)
        for lev in (((level + 1) as usize)..=(cur_max_level as usize)).rev() {
            let d = (self.dist_fn)(Self::get_vec(&inner.vectors, cur_ep, self.dim), vector);
            let mut cur_dist = d;

            let mut changed = true;
            while changed {
                changed = false;
                let neighbor_list = inner.neighbors[cur_ep]
                    .get(lev)
                    .cloned()
                    .unwrap_or_default();
                for &nb in &neighbor_list {
                    let nb_dist =
                        (self.dist_fn)(Self::get_vec(&inner.vectors, nb, self.dim), vector);
                    if nb_dist < cur_dist {
                        cur_dist = nb_dist;
                        cur_ep = nb;
                        changed = true;
                    }
                }
            }
        }

        // Insert at layers from min(level, max_level) down to 0
        let insert_from = std::cmp::min(level, cur_max_level) as usize;
        let mut ep_for_layer = cur_ep;

        for lev in (0..=insert_from).rev() {
            let results = Self::search_layer(
                &inner.vectors,
                self.dist_fn,
                self.dim,
                &inner.neighbors,
                vector,
                ep_for_layer,
                self.ef_construction,
                lev,
            );

            let m_for_layer = if lev == 0 { self.m_max0 } else { self.m_max };
            let neighbors_to_add = Self::select_neighbors(
                &inner.vectors,
                self.dist_fn,
                self.dim,
                &results,
                m_for_layer,
            );

            // Set neighbors for the new node at this layer
            if lev < inner.neighbors[iid].len() {
                inner.neighbors[iid][lev] = neighbors_to_add.iter().map(|&(_, n)| n).collect();
            }

            // Add bidirectional links and prune if needed
            for &(_, nb) in &neighbors_to_add {
                // Ensure neighbor has this layer
                if lev < inner.neighbors[nb].len() {
                    inner.neighbors[nb][lev].push(iid);
                    // Prune if over capacity
                    if inner.neighbors[nb][lev].len() > m_for_layer {
                        let nb_vec = Self::get_vec(&inner.vectors, nb, self.dim);
                        let mut candidates: Vec<(f32, usize)> = inner.neighbors[nb][lev]
                            .iter()
                            .map(|&n| {
                                let d = (self.dist_fn)(
                                    nb_vec,
                                    Self::get_vec(&inner.vectors, n, self.dim),
                                );
                                (d, n)
                            })
                            .collect();
                        candidates.sort_by(|a, b| FloatOrd(a.0).cmp(&FloatOrd(b.0)));
                        let pruned = Self::select_neighbors(
                            &inner.vectors,
                            self.dist_fn,
                            self.dim,
                            &candidates,
                            m_for_layer,
                        );
                        inner.neighbors[nb][lev] = pruned.iter().map(|&(_, n)| n).collect();
                    }
                }
            }

            // Use the closest result as entry point for the next layer down
            if !results.is_empty() {
                ep_for_layer = results[0].1;
            }
        }

        // Update entry point if new level is higher
        if level > cur_max_level {
            inner.entry_point = Some(iid);
            inner.max_level = level;
        }

        Ok(())
    }

    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        if query.len() != self.dim {
            return Err(VaneError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        if k == 0 {
            return Err(VaneError::InvalidK);
        }
        let inner = self.inner.read();
        if inner.count == 0 {
            return Ok(Vec::new());
        }

        let mut curr = inner.entry_point.unwrap();
        let mut d = (self.dist_fn)(query, Self::get_vec(&inner.vectors, curr, self.dim));

        // Greedy descent through upper layers
        for l in (1..=inner.max_level).rev() {
            let lu = l as usize;
            let mut changed = true;
            while changed {
                changed = false;
                if lu < inner.neighbors[curr].len() {
                    for &n in &inner.neighbors[curr][lu] {
                        let nd = (self.dist_fn)(query, Self::get_vec(&inner.vectors, n, self.dim));
                        if nd < d {
                            d = nd;
                            curr = n;
                            changed = true;
                        }
                    }
                }
            }
        }

        // Search at layer 0 with ef = max(ef_search, k)
        let ef = self.ef_search.load(Ordering::Relaxed).max(k);
        let top = Self::search_layer(
            &inner.vectors,
            self.dist_fn,
            self.dim,
            &inner.neighbors,
            query,
            curr,
            ef,
            0,
        );

        let mut results: Vec<SearchResult> = top
            .into_iter()
            .take(k)
            .map(|(dist, iid)| SearchResult::new(inner.ext_ids[iid], dist))
            .collect();
        results.sort();
        results.truncate(k);
        Ok(results)
    }

    /// Generate a random level using exponential distribution.
    fn get_level(rng: &mut StdRng, mult: f64) -> i32 {
        use rand::Rng;
        let r: f64 = rng.random::<f64>().max(MIN_LEVEL_RANDOM);
        let level = (-r.ln() * mult) as i32;
        level.min(MAX_LEVEL)
    }

    /// Get a vector slice by internal ID.
    fn get_vec(vectors: &[f32], iid: usize, dim: usize) -> &[f32] {
        let start = iid * dim;
        &vectors[start..start + dim]
    }

    /// Beam search on a single graph layer.
    /// Returns results sorted by distance ascending.
    fn search_layer(
        vectors: &[f32],
        dist_fn: DistanceFn,
        dim: usize,
        neighbors: &[Vec<Vec<usize>>],
        query: &[f32],
        entry: usize,
        ef: usize,
        level: usize,
    ) -> Vec<(f32, usize)> {
        let entry_dist = dist_fn(Self::get_vec(vectors, entry, dim), query);

        // Min-heap of candidates (closest first)
        let mut candidates: BinaryHeap<Reverse<(FloatOrd, usize)>> = BinaryHeap::new();
        candidates.push(Reverse((FloatOrd(entry_dist), entry)));

        // Max-heap of results (farthest first, capped at ef)
        let mut results: BinaryHeap<(FloatOrd, usize)> = BinaryHeap::new();
        results.push((FloatOrd(entry_dist), entry));

        let mut visited = HashSet::new();
        visited.insert(entry);

        while let Some(Reverse((FloatOrd(c_dist), c_id))) = candidates.pop() {
            // Stop if closest candidate is farther than farthest result
            if let Some(&(FloatOrd(f_dist), _)) = results.peek() {
                if c_dist > f_dist {
                    break;
                }
            }

            // Expand candidate's neighbors at this layer
            let nb_list = neighbors[c_id].get(level).cloned().unwrap_or_default();
            for &nb in &nb_list {
                if visited.contains(&nb) {
                    continue;
                }
                visited.insert(nb);

                let nb_dist = dist_fn(Self::get_vec(vectors, nb, dim), query);

                let should_add = if results.len() < ef {
                    true
                } else if let Some(&(FloatOrd(f_dist), _)) = results.peek() {
                    nb_dist < f_dist
                } else {
                    true
                };

                if should_add {
                    candidates.push(Reverse((FloatOrd(nb_dist), nb)));
                    results.push((FloatOrd(nb_dist), nb));
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }

        // Convert to sorted vec (ascending distance)
        let mut result_vec: Vec<(f32, usize)> = results
            .into_iter()
            .map(|(FloatOrd(d), id)| (d, id))
            .collect();
        result_vec.sort_by(|a, b| FloatOrd(a.0).cmp(&FloatOrd(b.0)));
        result_vec
    }

    /// Heuristic neighbor selection (Algorithm 4 from HNSW paper).
    fn select_neighbors(
        vectors: &[f32],
        dist_fn: DistanceFn,
        dim: usize,
        candidates: &[(f32, usize)],
        m: usize,
    ) -> Vec<(f32, usize)> {
        if candidates.len() <= m {
            return candidates.to_vec();
        }

        let mut sorted = candidates.to_vec();
        sorted.sort_by(|a, b| FloatOrd(a.0).cmp(&FloatOrd(b.0)));

        let mut selected: Vec<(f32, usize)> = Vec::with_capacity(m);
        let mut remaining: Vec<(f32, usize)> = Vec::new();

        for &(dist, cid) in &sorted {
            if selected.len() >= m {
                break;
            }

            // Heuristic: include only if not closer to any already-selected neighbor
            let is_diverse = selected.iter().all(|&(_, sid)| {
                let inter_dist = dist_fn(
                    Self::get_vec(vectors, cid, dim),
                    Self::get_vec(vectors, sid, dim),
                );
                inter_dist >= dist
            });

            if is_diverse {
                selected.push((dist, cid));
            } else {
                remaining.push((dist, cid));
            }
        }

        // Fill remaining slots with closest candidates not yet selected
        if selected.len() < m {
            let selected_set: HashSet<usize> = selected.iter().map(|&(_, id)| id).collect();
            for &(dist, cid) in &remaining {
                if selected.len() >= m {
                    break;
                }
                if !selected_set.contains(&cid) {
                    selected.push((dist, cid));
                }
            }
        }

        selected
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

    #[test]
    fn add_single_vector() {
        let idx = HnswIndex::builder(3, DistanceMetric::L2)
            .capacity(100)
            .build()
            .unwrap();
        idx.add(1, &[1.0, 2.0, 3.0]).unwrap();
        assert_eq!(idx.size(), 1);
        assert!(idx.contains(1));
        assert_eq!(idx.get_vector(1).unwrap(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn add_multiple_vectors() {
        let idx = HnswIndex::builder(3, DistanceMetric::L2)
            .capacity(100)
            .build()
            .unwrap();
        for i in 0..50u64 {
            idx.add(i, &[i as f32, 0.0, 0.0]).unwrap();
        }
        assert_eq!(idx.size(), 50);
        for i in 0..50u64 {
            assert!(idx.contains(i));
        }
    }

    #[test]
    fn add_rejects_duplicate() {
        let idx = HnswIndex::builder(3, DistanceMetric::L2)
            .capacity(100)
            .build()
            .unwrap();
        idx.add(1, &[1.0, 2.0, 3.0]).unwrap();
        assert!(idx.add(1, &[4.0, 5.0, 6.0]).is_err());
    }

    #[test]
    fn add_rejects_wrong_dim() {
        let idx = HnswIndex::builder(3, DistanceMetric::L2)
            .capacity(100)
            .build()
            .unwrap();
        assert!(idx.add(1, &[1.0, 2.0]).is_err());
    }

    #[test]
    fn add_rejects_when_full() {
        let idx = HnswIndex::builder(2, DistanceMetric::L2)
            .capacity(2)
            .build()
            .unwrap();
        idx.add(0, &[0.0, 0.0]).unwrap();
        idx.add(1, &[1.0, 1.0]).unwrap();
        assert!(matches!(idx.add(2, &[2.0, 2.0]), Err(VaneError::IndexFull)));
    }

    #[test]
    fn search_finds_exact_match() {
        let idx = HnswIndex::builder(3, DistanceMetric::L2)
            .capacity(100)
            .seed(42)
            .build()
            .unwrap();
        idx.add(1, &[0.0, 0.0, 0.0]).unwrap();
        idx.add(2, &[10.0, 10.0, 10.0]).unwrap();
        idx.add(3, &[20.0, 20.0, 20.0]).unwrap();

        let results = idx.search(&[0.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(results[0].id, 1);
        assert!(results[0].distance < 1e-6);
    }

    #[test]
    fn search_returns_k_results() {
        let idx = HnswIndex::builder(2, DistanceMetric::L2)
            .capacity(100)
            .seed(42)
            .build()
            .unwrap();
        for i in 0..20u64 {
            idx.add(i, &[i as f32, 0.0]).unwrap();
        }
        let results = idx.search(&[5.0, 0.0], 3).unwrap();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn search_empty_index() {
        let idx = HnswIndex::builder(3, DistanceMetric::L2)
            .capacity(100)
            .build()
            .unwrap();
        let results = idx.search(&[1.0, 2.0, 3.0], 5).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn search_wrong_dimension() {
        let idx = HnswIndex::builder(3, DistanceMetric::L2)
            .capacity(100)
            .build()
            .unwrap();
        assert!(idx.search(&[1.0, 2.0], 5).is_err());
    }
}
