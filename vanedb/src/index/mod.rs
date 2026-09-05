//! Approximate nearest-neighbour search over an HNSW graph.

use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};

use parking_lot::RwLock;
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::distance::{distance_fn, DistanceFn, Metric};
use crate::error::{Result, VaneError};
use crate::store::SearchResult;
use crate::validation::{compare_distances, validate_finite};

mod persistence;

// Versioned thread-local visited tracker. `marks[i] == epoch` means visited;
// the epoch is bumped each `search_layer` call, so the per-search work stays
// O(visited) instead of O(N) (which a fresh-bitmap-per-call or HashSet
// becomes at scale). On the rare epoch wrap (every 65k searches with u16
// the buffer is reset once. Buffer is shared across Index instances on
// a thread (monotonic epoch keeps cross-index marks distinct) and is
// retained across calls so we pay the allocation cost at most once.
//
// Mirrors the optimization in vanedb-cpp src/core/index.h.
thread_local! {
    static VISITED: RefCell<VisitedBuffer> = const { RefCell::new(VisitedBuffer::new()) };
}

struct VisitedBuffer {
    marks: Vec<u16>,
    epoch: u16,
}

impl VisitedBuffer {
    const fn new() -> Self {
        Self {
            marks: Vec::new(),
            epoch: 0,
        }
    }

    /// Begin a new search pass over `total` nodes. Returns the epoch tag for
    /// this call; callers compare `marks[i] == ep` to test visited.
    fn begin(&mut self, total: usize) -> u16 {
        if self.marks.len() < total {
            self.marks.resize(total, 0);
        }
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            // Wrap: zero the whole buffer, not just the active range. It is
            // shared across every Index on this thread and never shrunk, so
            // marks above `total` belong to some larger index and would be
            // read as current once the epoch climbs past them again.
            self.marks.fill(0);
            self.epoch = 1;
        }
        self.epoch
    }
}

/// Wrapper for f32 that implements Ord (needed for BinaryHeap).
#[derive(Debug, Clone, Copy)]
struct FloatOrd(f32);

impl PartialEq for FloatOrd {
    fn eq(&self, other: &Self) -> bool {
        compare_distances(self.0, other.0).is_eq()
    }
}

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
            .unwrap_or_else(|| match (self.0.is_nan(), other.0.is_nan()) {
                (false, true) => std::cmp::Ordering::Less,
                (true, false) => std::cmp::Ordering::Greater,
                (true, true) => self.0.total_cmp(&other.0),
                (false, false) => unreachable!("only NaN is unordered"),
            })
    }
}

pub(super) const MAX_LEVEL: i32 = 32;
const MIN_LEVEL_RANDOM: f64 = 1e-9;

/// Approximate k-nearest-neighbour search over a Hierarchical Navigable
/// Small World graph.
///
/// Search is sub-linear in the corpus, at the cost of occasionally missing a
/// true neighbour. Recall is traded against speed at query time with
/// [`set_ef_search`](Self::set_ef_search) and at build time with
/// [`m`](IndexBuilder::m) and
/// [`ef_construction`](IndexBuilder::ef_construction).
///
/// Built through [`Index::builder`]. Mutating methods take `&self`; the
/// index is internally synchronised.
pub struct Index {
    pub(super) dim: usize,
    pub(super) metric: Metric,
    pub(super) dist_fn: DistanceFn,
    pub(super) max_elements: usize,
    pub(super) m: usize,
    pub(super) m_max: usize,
    pub(super) m_max0: usize,
    pub(super) ef_construction: usize,
    pub(super) ef_search: AtomicUsize,
    pub(super) mult: f64,
    /// Original RNG seed; persisted on save and used to deterministically
    /// rewind the RNG to its post-`count`-inserts state on load.
    pub(super) seed: u64,
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

/// Configures an [`Index`] before construction.
///
/// Capacity is fixed once built, because the graph's storage is allocated up
/// front.
pub struct IndexBuilder {
    dim: usize,
    metric: Metric,
    capacity: usize,
    m: usize,
    ef_construction: usize,
    seed: u64,
}

impl std::fmt::Debug for Index {
    /// Identity and size only; the graph sits behind a lock.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Index")
            .field("dim", &self.dim)
            .field("metric", &self.metric)
            .field("size", &self.size())
            .field("m", &self.m)
            .field("ef_construction", &self.ef_construction)
            .finish()
    }
}

impl Index {
    /// Starts configuring an index over vectors of `dim` components.
    pub fn builder(dim: usize, metric: Metric) -> IndexBuilder {
        IndexBuilder {
            dim,
            metric,
            capacity: 100_000,
            m: 16,
            ef_construction: 200,
            seed: 42,
        }
    }

    /// Number of vectors in the graph.
    pub fn size(&self) -> usize {
        self.inner.read().count
    }

    /// Whether the graph holds no vectors.
    pub fn is_empty(&self) -> bool {
        self.size() == 0
    }

    /// Vectors this index can hold; adding beyond it fails with
    /// [`crate::VaneError::IndexFull`].
    pub fn capacity(&self) -> usize {
        self.max_elements
    }

    /// Component count of every vector in this index.
    pub fn dimension(&self) -> usize {
        self.dim
    }

    /// The metric this index ranks by.
    pub fn metric(&self) -> Metric {
        self.metric
    }

    /// Whether a vector is stored under `id`.
    pub fn contains(&self, id: u64) -> bool {
        self.inner.read().id_map.contains_key(&id)
    }

    /// Returns a copy of the vector stored under `id`.
    pub fn get_vector(&self, id: u64) -> Result<Vec<f32>> {
        let inner = self.inner.read();
        let &iid = inner.id_map.get(&id).ok_or(VaneError::NotFound { id })?;
        let start = iid * self.dim;
        Ok(inner.vectors[start..start + self.dim].to_vec())
    }

    /// Sets the search beam width: higher recovers more true neighbours and
    /// costs more time. Applies to subsequent searches.
    pub fn set_ef_search(&self, ef: usize) {
        self.ef_search.store(ef, Ordering::Relaxed);
    }

    /// The current search beam width.
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
        validate_finite(vector, "vector")?;

        let mut inner = self.inner.write();

        if inner.count >= self.max_elements {
            return Err(VaneError::IndexFull);
        }
        if inner.id_map.contains_key(&id) {
            return Err(VaneError::DuplicateId { id });
        }

        self.insert_into(&mut inner, id, vector);
        Ok(())
    }

    /// Insert many vectors under a single lock acquisition. `vectors` is the
    /// row-major concatenation of `ids.len()` vectors of `dimension()` floats.
    /// All-or-nothing: capacity and every id are validated before any insert,
    /// so an error leaves the index unchanged. Levels are drawn from the RNG
    /// in batch order, so the resulting graph is identical to serial `add`.
    pub fn add_batch(&self, ids: &[u64], vectors: &[f32]) -> Result<()> {
        if vectors.len() != ids.len() * self.dim {
            return Err(VaneError::DimensionMismatch {
                expected: ids.len() * self.dim,
                got: vectors.len(),
            });
        }
        validate_finite(vectors, "vector batch")?;

        let mut inner = self.inner.write();

        if inner.count + ids.len() > self.max_elements {
            return Err(VaneError::IndexFull);
        }
        let mut seen = HashSet::with_capacity(ids.len());
        for &id in ids {
            if inner.id_map.contains_key(&id) || !seen.insert(id) {
                return Err(VaneError::DuplicateId { id });
            }
        }

        for (&id, chunk) in ids.iter().zip(vectors.chunks_exact(self.dim)) {
            self.insert_into(&mut inner, id, chunk);
        }
        Ok(())
    }

    /// Graph insertion body shared by `add` and `add_batch`. Caller must hold
    /// the write lock and have already validated dimension, capacity, and id
    /// uniqueness — from here on insertion cannot fail.
    fn insert_into(&self, inner: &mut Inner, id: u64, vector: &[f32]) {
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
            return;
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
                inner.count,
            );

            // New nodes get M links; m_for_layer (2M at level 0) is only the
            // overflow cap for existing nodes' reverse links. Mirrors
            // vanedb-cpp add() / hnswlib semantics.
            let m_for_layer = if lev == 0 { self.m_max0 } else { self.m_max };
            let neighbors_to_add =
                Self::select_neighbors(&inner.vectors, self.dist_fn, self.dim, &results, self.m);

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
                        candidates.sort_by_key(|a| FloatOrd(a.0));
                        // Keep the m_for_layer closest (plain truncate, no
                        // diversity heuristic) — mirrors vanedb-cpp. The
                        // heuristic here re-scanned every overflowing list
                        // pairwise, costing ~36% of build time for no
                        // measurable recall gain (see PR benchmarks).
                        candidates.truncate(m_for_layer);
                        inner.neighbors[nb][lev] = candidates.iter().map(|&(_, n)| n).collect();
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
    }

    /// The `k` nearest vectors to `query`, nearest first.
    ///
    /// Approximate: a true neighbour can be missed. Raise the beam width with
    /// [`set_ef_search`](Self::set_ef_search) to trade speed for recall.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        if query.len() != self.dim {
            return Err(VaneError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        validate_finite(query, "query")?;
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
            inner.count,
        );

        // Sort the whole candidate set before cutting to k: `SearchResult`'s
        // Ord tie-breaks on id, and truncating first would pick among equal
        // distances in heap order instead. vanedb-cpp sorts then takes k for
        // the same reason.
        let mut results: Vec<SearchResult> = top
            .into_iter()
            .map(|(dist, iid)| SearchResult::new(inner.ext_ids[iid], dist))
            .collect();
        results.sort();
        results.truncate(k);
        Ok(results)
    }

    /// Generate a random level using exponential distribution.
    pub(super) fn get_level(rng: &mut StdRng, mult: f64) -> i32 {
        use rand::RngExt;
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
    ///
    /// `total` is the number of live nodes (== `inner.count`) and bounds the
    /// thread-local visited bitmap. Caller must guarantee `entry < total`.
    #[allow(clippy::too_many_arguments)]
    fn search_layer(
        vectors: &[f32],
        dist_fn: DistanceFn,
        dim: usize,
        neighbors: &[Vec<Vec<usize>>],
        query: &[f32],
        entry: usize,
        ef: usize,
        level: usize,
        total: usize,
    ) -> Vec<(f32, usize)> {
        debug_assert!(entry < total, "search_layer: entry out of range");

        VISITED.with_borrow_mut(|vb| {
            let epoch = vb.begin(total);

            let entry_dist = dist_fn(Self::get_vec(vectors, entry, dim), query);

            // Min-heap of candidates (closest first)
            let mut candidates: BinaryHeap<Reverse<(FloatOrd, usize)>> = BinaryHeap::new();
            candidates.push(Reverse((FloatOrd(entry_dist), entry)));

            // Max-heap of results (farthest first, capped at ef)
            let mut results: BinaryHeap<(FloatOrd, usize)> = BinaryHeap::new();
            results.push((FloatOrd(entry_dist), entry));

            vb.marks[entry] = epoch;

            while let Some(Reverse((FloatOrd(c_dist), c_id))) = candidates.pop() {
                // Stop if closest candidate is farther than farthest result
                if let Some(&(FloatOrd(f_dist), _)) = results.peek() {
                    if c_dist > f_dist {
                        break;
                    }
                }

                let Some(nb_list) = neighbors[c_id].get(level) else {
                    continue;
                };
                for &nb in nb_list {
                    if vb.marks[nb] == epoch {
                        continue;
                    }
                    vb.marks[nb] = epoch;

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
            result_vec.sort_by_key(|a| FloatOrd(a.0));
            result_vec
        })
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
        sorted.sort_by_key(|a| FloatOrd(a.0));

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

impl std::fmt::Debug for IndexBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IndexBuilder")
            .field("dim", &self.dim)
            .field("metric", &self.metric)
            .field("capacity", &self.capacity)
            .field("m", &self.m)
            .field("ef_construction", &self.ef_construction)
            .field("seed", &self.seed)
            .finish()
    }
}

impl IndexBuilder {
    /// Vectors the index will be able to hold. Fixed once built.
    pub fn capacity(mut self, cap: usize) -> Self {
        self.capacity = cap;
        self
    }

    /// Links kept per node. Larger graphs recall better and cost more memory
    /// and build time.
    pub fn m(mut self, m: usize) -> Self {
        self.m = m;
        self
    }

    /// Beam width used while building. Larger yields a better-connected graph
    /// and a slower build; it does not affect query cost.
    pub fn ef_construction(mut self, ef: usize) -> Self {
        self.ef_construction = ef;
        self
    }

    /// Seeds the level-assignment RNG. A fixed seed makes construction
    /// reproducible.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Allocates the graph and returns the index.
    pub fn build(self) -> Result<Index> {
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
        // Derived sizes are checked before any allocation: unchecked `m * 2`
        // and `capacity * dim` panicked on overflow, which a fallible builder
        // must not do — and which aborts the host process when the C ABI calls
        // it (#43).
        let m_max0 = self
            .m
            .checked_mul(2)
            .ok_or(VaneError::InvalidParameter("M * 2 overflows usize"))?;
        let vector_len = self
            .capacity
            .checked_mul(self.dim)
            .ok_or(VaneError::InvalidParameter(
                "capacity * dim overflows usize",
            ))?;
        let mut vectors: Vec<f32> = Vec::new();
        vectors
            .try_reserve_exact(vector_len)
            .map_err(|_| VaneError::InvalidParameter("capacity is too large to allocate"))?;
        vectors.resize(vector_len, 0.0);

        Ok(Index {
            dim: self.dim,
            metric: self.metric,
            dist_fn: distance_fn(self.metric),
            max_elements: self.capacity,
            m: self.m,
            m_max: self.m,
            m_max0,
            ef_construction,
            ef_search: AtomicUsize::new(50),
            mult,
            seed: self.seed,
            inner: RwLock::new(Inner {
                vectors,
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
        let idx = Index::builder(128, Metric::Cosine).build().unwrap();
        assert_eq!(idx.dimension(), 128);
        assert_eq!(idx.capacity(), 100_000);
        assert!(idx.is_empty());
        assert_eq!(idx.size(), 0);
        assert_eq!(idx.get_ef_search(), 50);
    }

    #[test]
    fn builder_custom_params() {
        let idx = Index::builder(64, Metric::L2)
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
        assert!(Index::builder(0, Metric::L2).build().is_err());
    }

    #[test]
    fn builder_rejects_zero_capacity() {
        assert!(Index::builder(64, Metric::L2).capacity(0).build().is_err());
    }

    #[test]
    fn builder_rejects_m_below_2() {
        assert!(Index::builder(64, Metric::L2).m(1).build().is_err());
    }

    #[test]
    fn set_ef_search() {
        let idx = Index::builder(64, Metric::L2).build().unwrap();
        idx.set_ef_search(100);
        assert_eq!(idx.get_ef_search(), 100);
    }

    #[test]
    fn add_single_vector() {
        let idx = Index::builder(3, Metric::L2).capacity(100).build().unwrap();
        idx.add(1, &[1.0, 2.0, 3.0]).unwrap();
        assert_eq!(idx.size(), 1);
        assert!(idx.contains(1));
        assert_eq!(idx.get_vector(1).unwrap(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn add_multiple_vectors() {
        let idx = Index::builder(3, Metric::L2).capacity(100).build().unwrap();
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
        let idx = Index::builder(3, Metric::L2).capacity(100).build().unwrap();
        idx.add(1, &[1.0, 2.0, 3.0]).unwrap();
        assert!(idx.add(1, &[4.0, 5.0, 6.0]).is_err());
    }

    #[test]
    fn add_rejects_wrong_dim() {
        let idx = Index::builder(3, Metric::L2).capacity(100).build().unwrap();
        assert!(idx.add(1, &[1.0, 2.0]).is_err());
    }

    #[test]
    fn add_rejects_when_full() {
        let idx = Index::builder(2, Metric::L2).capacity(2).build().unwrap();
        idx.add(0, &[0.0, 0.0]).unwrap();
        idx.add(1, &[1.0, 1.0]).unwrap();
        assert!(matches!(idx.add(2, &[2.0, 2.0]), Err(VaneError::IndexFull)));
    }

    #[test]
    fn search_finds_exact_match() {
        let idx = Index::builder(3, Metric::L2)
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
        let idx = Index::builder(2, Metric::L2)
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
        let idx = Index::builder(3, Metric::L2).capacity(100).build().unwrap();
        let results = idx.search(&[1.0, 2.0, 3.0], 5).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn search_wrong_dimension() {
        let idx = Index::builder(3, Metric::L2).capacity(100).build().unwrap();
        assert!(idx.search(&[1.0, 2.0], 5).is_err());
    }
}
