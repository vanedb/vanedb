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
use crate::flat::SearchResult;
use crate::validation::{compare_distances, validate_finite};
use storage::ChunkedVectors;

/// Upper bound on how many sibling-array slots a capacity hint pre-reserves.
/// The vectors themselves grow chunk by chunk; these arrays are small per
/// entry, but a hint of 10^9 should still not allocate gigabytes up front.
const RESERVE_CAP: usize = 1 << 20;

/// Largest number of vectors this engine will accept, from a builder hint or
/// from a file. Mirrors `MAX_VEC_SIZE` in vanedb-cpp `src/core/index.h`.
///
/// Storage grows on demand, so this is not an allocation bound. It rejects
/// values that can only be a mistake -- a capacity of `usize::MAX / 4096` is
/// a bug in the caller, not a plan.
pub(super) const MAX_ELEMENTS: usize = 100_000_000;

fn check_slot_growth(stored: usize, additional: usize) -> Result<()> {
    if stored
        .checked_add(additional)
        .is_none_or(|total| total > MAX_ELEMENTS)
    {
        return Err(VaneError::InvalidParameter(
            "stored slots exceed the 100 million persistence limit; compact deleted slots first",
        ));
    }
    Ok(())
}

mod graph_format;
mod persistence;
mod storage;

// Versioned thread-local visited tracker. `marks[i] == epoch` means visited;
// the epoch is bumped each `search_layer` call, so the per-search work stays
// O(visited) instead of O(N) (which a fresh-bitmap-per-call or HashSet
// becomes at scale). On the rare epoch wrap (every 65k searches with u16
// the buffer is reset once. Buffer is shared across ApproxIndex instances on
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
            // shared across every ApproxIndex on this thread and never shrunk, so
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
        compare_distances(self.0, other.0)
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
/// [`m`](ApproxIndexBuilder::m) and
/// [`ef_construction`](ApproxIndexBuilder::ef_construction).
///
/// Built through [`ApproxIndex::builder`]. Mutating methods take `&self`; the
/// index is internally synchronised.
pub struct ApproxIndex {
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
    pub(super) vectors: ChunkedVectors,
    pub(super) ext_ids: Vec<u64>,
    pub(super) id_map: HashMap<u64, usize>,
    pub(super) levels: Vec<i32>,
    pub(super) neighbors: Vec<Vec<Vec<usize>>>,
    pub(super) entry_point: Option<usize>,
    pub(super) max_level: i32,
    pub(super) count: usize,
    /// Tombstones, indexed by internal id. A deleted node keeps its links and
    /// keeps being traversed -- it may be the only path to a live
    /// neighbourhood -- but never appears in a result.
    pub(super) deleted: Vec<bool>,
    /// Live nodes, i.e. `count` minus the tombstones. Kept rather than
    /// recomputed so `len()` stays O(1).
    pub(super) live: usize,
    pub(super) rng: StdRng,
    pub(super) persisted_rng: Option<persistence::RngState>,
}

/// Configures an [`ApproxIndex`] before construction.
pub struct ApproxIndexBuilder {
    dim: usize,
    metric: Metric,
    capacity: usize,
    m: usize,
    ef_construction: usize,
    seed: u64,
}

impl std::fmt::Debug for ApproxIndex {
    /// Identity and size only; the graph sits behind a lock.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ApproxIndex")
            .field("dim", &self.dim)
            .field("metric", &self.metric)
            .field("size", &self.size())
            .field("m", &self.m)
            .field("ef_construction", &self.ef_construction)
            .finish()
    }
}

/// Per-query options for [`ApproxIndex::search_with`].
///
/// Build with [`SearchParams::new`] and the setters. The fields are private,
/// so an option added later is not a breaking change; `#[non_exhaustive]`
/// records that intent for readers.
///
/// Carries a lifetime it does not yet use. Rust has no default lifetime
/// parameters, so `SearchParams<'a>` cannot be introduced later without
/// breaking every mention of the type — which would permanently rule out the
/// first option callers are likely to want, a filter borrowing a bitmap or a
/// set rather than owning one behind an `Arc`.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct SearchParams<'a> {
    ef_search: Option<usize>,
    _borrowed: std::marker::PhantomData<&'a ()>,
}

impl SearchParams<'_> {
    /// Options that follow the index's own settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Beam width for this query only, overriding the index's `ef_search`.
    ///
    /// Larger widens the search: better recall, more work. Values below `k`
    /// are raised to `k`, since fewer candidates than results is meaningless.
    pub fn ef_search(mut self, ef: usize) -> Self {
        self.ef_search = Some(ef);
        self
    }
}

impl ApproxIndex {
    /// Starts configuring an index over vectors of `dim` components.
    pub fn builder(dim: usize, metric: Metric) -> ApproxIndexBuilder {
        ApproxIndexBuilder {
            dim,
            metric,
            capacity: 100_000,
            m: 16,
            ef_construction: 200,
            seed: 42,
        }
    }

    /// Number of vectors in the graph, excluding deleted ones.
    pub fn size(&self) -> usize {
        self.inner.read().live
    }

    /// Number of vectors in the graph, excluding deleted ones.
    ///
    /// `len` and `size` are the same call; `len` is the Rust spelling and
    /// `size` is what the C++ engine and the wasm bindings expose.
    pub fn len(&self) -> usize {
        self.inner.read().live
    }

    /// Whether the graph holds no vectors.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The capacity this index reserved for.
    ///
    /// A hint, not a limit: storage grows as vectors arrive, so adding beyond
    /// this succeeds. It exists so a known-size bulk load can avoid growth
    /// pauses. The persistence format limits the graph to 100 million stored
    /// slots, including tombstones; compact deleted slots to reclaim room.
    pub fn capacity(&self) -> usize {
        self.max_elements
    }

    /// Component count of every vector in this index.
    pub fn dimension(&self) -> usize {
        self.dim
    }

    /// Links a new node receives per layer. The layer-0 cap is `2 * m`.
    ///
    /// Reported for the same reason as [`metric`](Self::metric): a loaded
    /// index read this from the file, and a caller that did not build it has
    /// no other way to know what graph they are searching.
    pub fn m(&self) -> usize {
        self.m
    }

    /// Beam width used while building. Larger values explore more candidates
    /// and increase construction work; measure recall for your data.
    pub fn ef_construction(&self) -> usize {
        self.ef_construction
    }

    /// The seed the level distribution was drawn from.
    ///
    /// Preserved across save and load. This does not promise identical future
    /// graph topology across engines or dependency versions.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// The metric this index ranks by.
    pub fn metric(&self) -> Metric {
        self.metric
    }

    /// Whether a vector is stored under `id`.
    pub fn contains(&self, id: u64) -> bool {
        self.inner.read().id_map.contains_key(&id)
    }

    /// Inserts `vector` under `id`, replacing any existing entry.
    ///
    /// Both halves happen under one write lock, so a concurrent reader never
    /// observes the id missing — `remove` followed by `add` takes the lock
    /// twice and does.
    ///
    /// The old slot is tombstoned rather than overwritten: its graph links
    /// were built for the old vector's position and would be wrong for the
    /// new one. That leaves a slot behind, so a long upsert loop still needs
    /// [`compact`](Self::compact). At the 100 million stored-slot limit, this
    /// returns an error and leaves the existing entry unchanged.
    pub fn upsert(&self, id: u64, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dim {
            return Err(VaneError::DimensionMismatch {
                expected: self.dim,
                got: vector.len(),
            });
        }
        validate_finite(vector, "vector")?;

        let mut inner = self.inner.write();
        check_slot_growth(inner.count, 1)?;
        if let Some(iid) = inner.id_map.remove(&id) {
            inner.deleted[iid] = true;
            inner.live -= 1;
        }
        self.insert_into(&mut inner, id, vector);
        Ok(())
    }

    /// Number of tombstoned slots: removed vectors whose space has not been
    /// reclaimed.
    ///
    /// Each `remove` leaves one, and re-adding the same id allocates a fresh
    /// slot rather than reusing it — so a long-lived upsert loop grows the
    /// index even at constant [`len`](Self::len). Call [`compact`](Self::compact)
    /// when this gets large relative to `len`.
    pub fn tombstones(&self) -> usize {
        let inner = self.inner.read();
        inner.count - inner.live
    }

    /// Rebuilds the graph without the tombstoned slots, reclaiming their space.
    ///
    /// This is a full rebuild — cost is comparable to constructing the index
    /// from the live vectors — and it takes the write lock throughout, so
    /// concurrent searches block. Ids, vectors and the configured parameters
    /// are preserved; the graph itself is rebuilt, so results may shift
    /// exactly as much as any two independently built graphs differ.
    ///
    /// A tombstone-free index is rebuilt too rather than skipped, so the
    /// result does not depend on how the index got to its current contents.
    pub fn compact(&self) -> Result<()> {
        let mut inner = self.inner.write();
        if inner.count == 0 {
            return Ok(());
        }

        // Snapshot the live entries in slot order, so a compacted index
        // matches one built by inserting them in that order.
        let live: Vec<(u64, Vec<f32>)> = (0..inner.count)
            .filter(|&iid| !inner.deleted[iid])
            .map(|iid| (inner.ext_ids[iid], inner.vectors.get(iid).to_vec()))
            .collect();

        *inner = Inner {
            vectors: ChunkedVectors::with_capacity(self.dim, live.len()),
            ext_ids: Vec::with_capacity(live.len()),
            id_map: HashMap::with_capacity(live.len()),
            levels: Vec::with_capacity(live.len()),
            neighbors: Vec::with_capacity(live.len()),
            deleted: Vec::with_capacity(live.len()),
            live: 0,
            entry_point: None,
            max_level: -1,
            count: 0,
            // Replay from the original seed so a compacted index is the one
            // you would have built from these vectors in this order.
            rng: StdRng::seed_from_u64(self.seed),
            persisted_rng: None,
        };
        for (id, vector) in &live {
            self.insert_into(&mut inner, *id, vector);
        }
        Ok(())
    }

    /// Removes the vector stored under `id`.
    ///
    /// The node is tombstoned rather than unlinked: its edges may be the only
    /// route between live neighbourhoods, so it keeps being traversed and
    /// simply never appears in a result. The id becomes free for reuse.
    ///
    /// Space is not reclaimed. A graph that is mostly tombstones is slower to
    /// search than its live count suggests; rebuild it if that happens.
    pub fn remove(&self, id: u64) -> Result<()> {
        let mut inner = self.inner.write();
        let iid = inner.id_map.remove(&id).ok_or(VaneError::NotFound { id })?;
        debug_assert!(!inner.deleted[iid], "id_map held a tombstoned node");
        inner.deleted[iid] = true;
        inner.live -= 1;
        Ok(())
    }

    /// Returns a copy of the vector stored under `id`.
    pub fn get(&self, id: u64) -> Result<Vec<f32>> {
        self.get_vector(id)
    }

    /// The vector stored under `id`. Same as [`get`](Self::get), which is the
    /// spelling `FlatIndex` and `DiskIndex` use; both exist so a program is
    /// not tied to one index type (#85).
    pub fn get_vector(&self, id: u64) -> Result<Vec<f32>> {
        let inner = self.inner.read();
        let &iid = inner.id_map.get(&id).ok_or(VaneError::NotFound { id })?;
        Ok(inner.vectors.get(iid).to_vec())
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
    ///
    /// Returns an error at 100 million stored slots, including tombstones.
    /// Call [`compact`](Self::compact) to reclaim deleted slots.
    pub fn add(&self, id: u64, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dim {
            return Err(VaneError::DimensionMismatch {
                expected: self.dim,
                got: vector.len(),
            });
        }
        validate_finite(vector, "vector")?;

        let mut inner = self.inner.write();

        if inner.id_map.contains_key(&id) {
            return Err(VaneError::DuplicateId { id });
        }

        check_slot_growth(inner.count, 1)?;
        self.insert_into(&mut inner, id, vector);
        Ok(())
    }

    /// Insert many vectors under a single lock acquisition. `vectors` is the
    /// row-major concatenation of `ids.len()` vectors of `dimension()` floats.
    /// All-or-nothing: the batch shape, stored-slot limit and every id are
    /// validated before any insert, so an error leaves the index unchanged.
    /// Levels are drawn in batch order, producing the same graph as serial `add`.
    pub fn add_batch(&self, ids: &[u64], vectors: &[f32]) -> Result<()> {
        // Checked: `ids.len() * dim` wraps for absurd dimensions, and a
        // wrapped zero would match an empty slice, accepting a batch that
        // inserts nothing.
        let expected = ids
            .len()
            .checked_mul(self.dim)
            .ok_or(VaneError::InvalidParameter(
                "ids.len() * dim overflows usize",
            ))?;
        if vectors.len() != expected {
            return Err(VaneError::BatchLengthMismatch {
                ids: ids.len(),
                vectors: vectors.len(),
                dim: self.dim,
            });
        }
        validate_finite(vectors, "vector batch")?;

        let mut inner = self.inner.write();

        let mut seen = HashSet::with_capacity(ids.len());
        for &id in ids {
            if inner.id_map.contains_key(&id) || !seen.insert(id) {
                return Err(VaneError::DuplicateId { id });
            }
        }

        check_slot_growth(inner.count, ids.len())?;
        for (&id, chunk) in ids.iter().zip(vectors.chunks_exact(self.dim)) {
            self.insert_into(&mut inner, id, chunk);
        }
        Ok(())
    }

    /// Graph insertion body shared by `add` and `add_batch`. Caller must hold
    /// the write lock and have already validated dimension and id
    /// uniqueness and stored-slot limit — from here on insertion cannot fail.
    fn insert_into(&self, inner: &mut Inner, id: u64, vector: &[f32]) {
        let iid = inner.count;
        inner.count += 1;

        // Storage grows here rather than being pre-sized to capacity, so an
        // empty index costs nothing and the reservation hint is not a ceiling.
        inner.vectors.push(vector);
        inner.deleted.push(false);
        inner.live += 1;
        debug_assert_eq!(
            inner.vectors.len(),
            inner.count,
            "storage and count diverged"
        );
        inner.ext_ids.push(id);
        inner.id_map.insert(id, iid);

        // Generate random level
        let level = Self::get_level(&mut inner.rng, self.mult);
        inner.levels.push(level);

        // Allocate neighbor lists for each layer
        inner
            .neighbors
            .push((0..=level as usize).map(|_| Vec::new()).collect());

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
            let d = (self.dist_fn)(inner.vectors.get(cur_ep), vector);
            let mut cur_dist = d;

            let mut changed = true;
            while changed {
                changed = false;
                let neighbor_list = inner.neighbors[cur_ep]
                    .get(lev)
                    .cloned()
                    .unwrap_or_default();
                for &nb in &neighbor_list {
                    let nb_dist = (self.dist_fn)(inner.vectors.get(nb), vector);
                    if compare_distances(nb_dist, cur_dist).is_lt() {
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
                &inner.neighbors,
                vector,
                ep_for_layer,
                self.ef_construction,
                lev,
                inner.count,
                // Build links to whatever is nearest, tombstoned or not:
                // excluding them here could disconnect the graph.
                &[],
            );

            // New nodes get M links; m_for_layer (2M at level 0) is only the
            // overflow cap for existing nodes' reverse links. Mirrors
            // vanedb-cpp add() / hnswlib semantics.
            let m_for_layer = if lev == 0 { self.m_max0 } else { self.m_max };
            let neighbors_to_add =
                Self::select_neighbors(&inner.vectors, self.dist_fn, &results, self.m);

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
                        let nb_vec = inner.vectors.get(nb);
                        let mut candidates: Vec<(f32, usize)> = inner.neighbors[nb][lev]
                            .iter()
                            .map(|&n| {
                                let d = (self.dist_fn)(nb_vec, inner.vectors.get(n));
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
    /// [`set_ef_search`](Self::set_ef_search) to trade speed for recall, or
    /// pass it for one query with [`search_with`](Self::search_with).
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        self.search_with(query, k, &SearchParams::new())
    }

    /// [`search`](Self::search) with per-query options.
    ///
    /// Options given here apply to this call only and leave the index
    /// untouched, so callers with different needs can share one index without
    /// changing each other's results.
    pub fn search_with(
        &self,
        query: &[f32],
        k: usize,
        params: &SearchParams<'_>,
    ) -> Result<Vec<SearchResult>> {
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
        let mut d = (self.dist_fn)(query, inner.vectors.get(curr));

        // Greedy descent through upper layers
        for l in (1..=inner.max_level).rev() {
            let lu = l as usize;
            let mut changed = true;
            while changed {
                changed = false;
                if lu < inner.neighbors[curr].len() {
                    for &n in &inner.neighbors[curr][lu] {
                        let nd = (self.dist_fn)(query, inner.vectors.get(n));
                        if compare_distances(nd, d).is_lt() {
                            d = nd;
                            curr = n;
                            changed = true;
                        }
                    }
                }
            }
        }

        // Search at layer 0 with ef = max(ef_search, k)
        let ef = params
            .ef_search
            .unwrap_or_else(|| self.ef_search.load(Ordering::Relaxed))
            .max(k);
        let top = Self::search_layer(
            &inner.vectors,
            self.dist_fn,
            &inner.neighbors,
            query,
            curr,
            ef,
            0,
            inner.count,
            &inner.deleted,
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
        // Clamped at both ends: a negative level is later used as
        // `0..=level as usize`, which wraps to a ~2^64 range.
        level.clamp(0, MAX_LEVEL)
    }

    /// Beam search on a single graph layer.
    /// Returns results sorted by distance ascending.
    ///
    /// `total` is the number of live nodes (== `inner.count`) and bounds the
    /// thread-local visited bitmap. Caller must guarantee `entry < total`.
    #[allow(clippy::too_many_arguments)]
    fn search_layer(
        vectors: &ChunkedVectors,
        dist_fn: DistanceFn,
        neighbors: &[Vec<Vec<usize>>],
        query: &[f32],
        entry: usize,
        ef: usize,
        level: usize,
        total: usize,
        // Tombstones. Deleted nodes still expand the frontier -- their edges
        // may be the only route to a live neighbourhood -- but never occupy a
        // result slot. Empty means nothing is deleted, which is the build path.
        deleted: &[bool],
    ) -> Vec<(f32, usize)> {
        debug_assert!(entry < total, "search_layer: entry out of range");
        let live = |iid: usize| deleted.get(iid).is_none_or(|d| !d);

        VISITED.with_borrow_mut(|vb| {
            let epoch = vb.begin(total);

            let entry_dist = dist_fn(vectors.get(entry), query);

            // Min-heap of candidates (closest first)
            let mut candidates: BinaryHeap<Reverse<(FloatOrd, usize)>> = BinaryHeap::new();
            candidates.push(Reverse((FloatOrd(entry_dist), entry)));

            // Max-heap of results (farthest first, capped at ef)
            let mut results: BinaryHeap<(FloatOrd, usize)> = BinaryHeap::new();
            if live(entry) {
                results.push((FloatOrd(entry_dist), entry));
            }

            vb.marks[entry] = epoch;

            while let Some(Reverse((FloatOrd(c_dist), c_id))) = candidates.pop() {
                // Stop only once the result set is FULL and the closest
                // remaining candidate is farther than the farthest result.
                //
                // The `results.len() >= ef` conjunct is load-bearing when
                // tombstones are present: `results` holds live nodes only,
                // while `candidates` still holds deleted ones, so a popped
                // tombstone can be farther than the farthest live result while
                // the result set is nowhere near full. Breaking there abandons
                // exactly the traversal tombstones are kept for, and search
                // silently returns a fraction of `k`.
                //
                // On the build path `deleted` is empty, so every candidate is
                // also a result; an unfull `results` therefore holds every
                // visited node and `c_dist > f_dist` cannot hold. Construction
                // is unchanged.
                if results.len() >= ef {
                    if let Some(&(FloatOrd(f_dist), _)) = results.peek() {
                        if compare_distances(c_dist, f_dist).is_gt() {
                            break;
                        }
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

                    let nb_dist = dist_fn(vectors.get(nb), query);

                    let should_add = if results.len() < ef {
                        true
                    } else if let Some(&(FloatOrd(f_dist), _)) = results.peek() {
                        compare_distances(nb_dist, f_dist).is_lt()
                    } else {
                        true
                    };

                    if should_add {
                        candidates.push(Reverse((FloatOrd(nb_dist), nb)));
                        if live(nb) {
                            results.push((FloatOrd(nb_dist), nb));
                            if results.len() > ef {
                                results.pop();
                            }
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
        vectors: &ChunkedVectors,
        dist_fn: DistanceFn,
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
                let inter_dist = dist_fn(vectors.get(cid), vectors.get(sid));
                !compare_distances(inter_dist, dist).is_lt()
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

impl std::fmt::Debug for ApproxIndexBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ApproxIndexBuilder")
            .field("dim", &self.dim)
            .field("metric", &self.metric)
            .field("capacity", &self.capacity)
            .field("m", &self.m)
            .field("ef_construction", &self.ef_construction)
            .field("seed", &self.seed)
            .finish()
    }
}

/// Level-generation constant, derived entirely from `m`.
pub(super) fn derive_mult(m: usize) -> f64 {
    if m > 1 {
        1.0 / (m as f64).ln()
    } else {
        1.0
    }
}

impl ApproxIndexBuilder {
    /// Vectors to reserve room for. A hint, not a ceiling: chunks are
    /// allocated as vectors arrive and adding beyond this succeeds.
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
    ///
    /// # Errors
    ///
    /// Returns [`VaneError::ZeroDimension`] if `dim` is zero, or
    /// [`VaneError::InvalidParameter`] if a graph parameter is out of range or
    /// implies an allocation that would overflow.
    pub fn build(self) -> Result<ApproxIndex> {
        if self.dim == 0 {
            return Err(VaneError::ZeroDimension);
        }
        // A dimension whose byte size overflows can never hold a vector, and
        // would wrap to zero in the chunk sizing, dividing by zero.
        if self.dim.checked_mul(std::mem::size_of::<f32>()).is_none() {
            return Err(VaneError::InvalidParameter(
                "dim * size_of::<f32>() overflows usize",
            ));
        }
        if self.capacity == 0 {
            return Err(VaneError::InvalidParameter("capacity must be > 0"));
        }
        if self.m < 2 {
            return Err(VaneError::InvalidParameter("M must be >= 2"));
        }
        let ef_construction = self.ef_construction.max(self.m);
        let mult = derive_mult(self.m);
        // Derived sizes are checked before any allocation: unchecked `m * 2`
        // and `capacity * dim` panicked on overflow, which a fallible builder
        // must not do — and which aborts the host process when the C ABI calls
        // it (#43).
        let m_max0 = self
            .m
            .checked_mul(2)
            .ok_or(VaneError::InvalidParameter("M * 2 overflows usize"))?;
        // capacity is a reserve hint, not a ceiling: storage grows as vectors
        // arrive. Nothing is allocated until the first insert, so an unused
        // index costs nothing however large the hint (#90).
        self.capacity
            .checked_mul(self.dim)
            .ok_or(VaneError::InvalidParameter(
                "capacity * dim overflows usize",
            ))?;
        if self.capacity > MAX_ELEMENTS {
            return Err(VaneError::InvalidParameter(
                "capacity exceeds the maximum this engine accepts",
            ));
        }
        let vectors = ChunkedVectors::with_capacity(self.dim, self.capacity);

        Ok(ApproxIndex {
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
                ext_ids: Vec::with_capacity(self.capacity.min(RESERVE_CAP)),
                id_map: HashMap::new(),
                levels: Vec::with_capacity(self.capacity.min(RESERVE_CAP)),
                neighbors: Vec::with_capacity(self.capacity.min(RESERVE_CAP)),
                deleted: Vec::with_capacity(self.capacity.min(RESERVE_CAP)),
                live: 0,
                entry_point: None,
                max_level: -1,
                count: 0,
                rng: StdRng::seed_from_u64(self.seed),
                persisted_rng: None,
            }),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_defaults() {
        let idx = ApproxIndex::builder(128, Metric::Cosine).build().unwrap();
        assert_eq!(idx.dimension(), 128);
        assert_eq!(idx.capacity(), 100_000);
        assert!(idx.is_empty());
        assert_eq!(idx.size(), 0);
        assert_eq!(idx.get_ef_search(), 50);
    }

    #[test]
    fn builder_custom_params() {
        let idx = ApproxIndex::builder(64, Metric::L2)
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
        assert!(ApproxIndex::builder(0, Metric::L2).build().is_err());
    }

    #[test]
    fn builder_rejects_zero_capacity() {
        assert!(ApproxIndex::builder(64, Metric::L2)
            .capacity(0)
            .build()
            .is_err());
    }

    #[test]
    fn builder_rejects_m_below_2() {
        assert!(ApproxIndex::builder(64, Metric::L2).m(1).build().is_err());
    }

    #[test]
    fn set_ef_search() {
        let idx = ApproxIndex::builder(64, Metric::L2).build().unwrap();
        idx.set_ef_search(100);
        assert_eq!(idx.get_ef_search(), 100);
    }

    #[test]
    fn add_single_vector() {
        let idx = ApproxIndex::builder(3, Metric::L2)
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
        let idx = ApproxIndex::builder(3, Metric::L2)
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
        let idx = ApproxIndex::builder(3, Metric::L2)
            .capacity(100)
            .build()
            .unwrap();
        idx.add(1, &[1.0, 2.0, 3.0]).unwrap();
        assert!(idx.add(1, &[4.0, 5.0, 6.0]).is_err());
    }

    #[test]
    fn add_rejects_wrong_dim() {
        let idx = ApproxIndex::builder(3, Metric::L2)
            .capacity(100)
            .build()
            .unwrap();
        assert!(idx.add(1, &[1.0, 2.0]).is_err());
    }

    #[test]
    fn adding_past_the_capacity_hint_grows_instead_of_failing() {
        let idx = ApproxIndex::builder(2, Metric::L2)
            .capacity(2)
            .build()
            .unwrap();
        for i in 0..50u64 {
            idx.add(i, &[i as f32, i as f32])
                .expect("capacity is a hint, not a ceiling");
        }
        assert_eq!(idx.size(), 50);
        // The graph must still be usable well past the hint.
        let hits = idx.search(&[49.0, 49.0], 1).unwrap();
        assert_eq!(hits[0].id, 49);
    }

    // Exercise the real public mutation paths at the format boundary without
    // allocating 100 million nodes. Restore the simulated count before checking
    // the graph, so any partial mutation remains observable.
    fn rejects_growth_unchanged(count: usize, operation: impl FnOnce(&ApproxIndex) -> Result<()>) {
        let index = ApproxIndex::builder(1, Metric::L2)
            .capacity(1)
            .build()
            .unwrap();
        index.add(7, &[1.0]).unwrap();
        let snapshot = || {
            let inner = index.inner.read();
            let mut bytes = Vec::new();
            graph_format::write(&mut bytes, &index, &inner).unwrap();
            bytes
        };
        let before = snapshot();
        index.inner.write().count = count;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| operation(&index)));
        let after_count = index.inner.read().count;
        index.inner.write().count = 1;
        assert!(matches!(
            result.expect("growth must return an error, not panic"),
            Err(VaneError::InvalidParameter(_))
        ));
        assert_eq!(after_count, count);
        assert_eq!(snapshot(), before, "failed growth changed the graph");
        assert_eq!(index.get_vector(7).unwrap(), [1.0]);
    }

    #[test]
    fn stored_slot_limit_accepts_the_boundary_and_rejects_overflow() {
        assert!(check_slot_growth(MAX_ELEMENTS - 1, 1).is_ok());
        assert!(check_slot_growth(MAX_ELEMENTS, 0).is_ok());
        assert!(check_slot_growth(MAX_ELEMENTS, 1).is_err());
        assert!(check_slot_growth(usize::MAX, 1).is_err());
    }

    #[test]
    fn add_rejects_the_persistence_slot_limit_without_mutation() {
        rejects_growth_unchanged(MAX_ELEMENTS, |index| index.add(8, &[2.0]));
    }

    #[test]
    fn batch_rejects_the_persistence_slot_limit_atomically() {
        rejects_growth_unchanged(MAX_ELEMENTS - 1, |index| {
            index.add_batch(&[8, 9], &[2.0, 3.0])
        });
    }

    #[test]
    fn upsert_rejects_the_persistence_slot_limit_without_deleting_the_old_id() {
        rejects_growth_unchanged(MAX_ELEMENTS, |index| index.upsert(7, &[2.0]));
        rejects_growth_unchanged(MAX_ELEMENTS, |index| index.upsert(8, &[2.0]));
    }

    #[test]
    fn search_finds_exact_match() {
        let idx = ApproxIndex::builder(3, Metric::L2)
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
        let idx = ApproxIndex::builder(2, Metric::L2)
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
        let idx = ApproxIndex::builder(3, Metric::L2)
            .capacity(100)
            .build()
            .unwrap();
        let results = idx.search(&[1.0, 2.0, 3.0], 5).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn search_wrong_dimension() {
        let idx = ApproxIndex::builder(3, Metric::L2)
            .capacity(100)
            .build()
            .unwrap();
        assert!(idx.search(&[1.0, 2.0], 5).is_err());
    }
}
