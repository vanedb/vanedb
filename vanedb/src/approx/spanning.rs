//! Layer-0 spanning tree that keeps every stored node reachable (#299).
//!
//! Reverse-link pruning keeps only the nearest entries of an overflowing
//! list. Nothing in that rule knows whether a dropped entry was the last
//! edge leading to its target, so under DOT, and under cosine on tightly
//! clustered data, whole groups of live vectors ended up with no path from
//! the entry point: present by every accessor, returned by no beam search.
//!
//! The remedy is to protect one edge pair per node. Call `p` a *mutual
//! elder* of `x` when `p` was stored before `x` and each lists the other on
//! layer 0. The **parent** of `x` is its nearest mutual elder that has not
//! already taken [`child_limit`] children, ties broken by slot. Pruning
//! never evicts the edge between a node and its parent, in either
//! direction. Parents always have the smaller slot, so the protected edges
//! form a tree rooted at slot 0 that spans the layer both ways, and layer 0
//! is strongly connected whatever start the upper layers hand the beam.
//!
//! The child limit is what keeps this cheap. Under DOT a few high-norm
//! nodes are everyone's nearest neighbour; without a limit their lists
//! would fill with protected children and stop tracking the nodes that
//! matter for navigation. With it a node gives up at most `child_limit + 1`
//! of its `2m` slots, and a node nobody has room for is chained to the
//! node stored just before it, which always has a free slot.
//!
//! The tree is a function of the graph and the vectors, not extra state
//! with a history. Mutual elders of `x` are created only while `x` is being
//! inserted and can only be evicted afterwards; the parent never is; and
//! child counts replay identically in slot order. That is why nothing is
//! persisted and the VNDB graph format is unchanged: [`derive`] recomputes
//! exactly the tree the writer had, and an index resumed from a file keeps
//! building the graph it would have built without the round trip.
//!
//! "Exactly" assumes the same distance kernel. Distances that differ in the
//! last bit, as between NEON, AVX2 and scalar, or between the two engines,
//! can order a near-tie differently and so derive a different tree. That
//! tree is still a valid one and is protected consistently from then on;
//! what is lost is only the byte-for-byte equality with a never-saved index.
//!
//! The degree cap holds without further rules: an overflowing list is one
//! over its cap because a node was just pushed onto it, and that node has
//! no parent yet, so there is always an entry that may be dropped.
//!
//! Reachable is a statement about the graph, not about a bounded beam. A
//! node chained to its predecessor is on a path, but a narrow beam that
//! never comes near it will still not return it.
//!
//! A file written by an older release of either engine is loaded as it was
//! saved. Its nodes are protected where they have a mutual elder; nodes it
//! already stranded stay stranded until `compact()` rebuilds the graph.

use super::storage::ChunkedVectors;
use crate::distance::DistanceFn;
use crate::validation::compare_distances;

/// Marks a node without a parent: slot 0, or a node a loaded graph left
/// with no mutual elder.
pub(super) const NO_PARENT: u32 = u32::MAX;

/// Whether the layer-0 edge between `a` and `b` belongs to the tree.
#[inline]
pub(super) fn is_tree_edge(parent: &[u32], a: usize, b: usize) -> bool {
    parent[a] == b as u32 || parent[b] == a as u32
}

/// How many children one node may take: an eighth of `m`, at least two.
/// Together with its own parent that is the most slots a node gives up to
/// the tree.
///
/// The floor is two, not one. With a limit of one every node has at most
/// one child, the tree is a path, and the only node still free to take a
/// child is the one stored last: the "tree" becomes the insertion order,
/// every node spends two slots on its predecessor and successor, and
/// low-beam recall falls. 0.2.0-rc.3 shipped that for `m < 16`.
pub(super) fn child_limit(m: usize) -> u16 {
    (m / 8).clamp(2, u16::MAX as usize) as u16
}

/// Position of the entry an overflowing list should drop: the farthest one
/// that is not a tree edge. `sorted` is `node`'s list, nearest first.
///
/// The entry just pushed is never a tree edge, so the fallback to the
/// farthest entry outright is unreachable for a list built by insertion.
pub(super) fn eviction_index(parent: &[u32], node: usize, sorted: &[(f32, usize)]) -> usize {
    sorted
        .iter()
        .rposition(|&(_, other)| !is_tree_edge(parent, node, other))
        .unwrap_or(sorted.len() - 1)
}

/// The parent of `node`: its nearest mutual elder with fewer than `limit`
/// children, ties broken by slot, or [`NO_PARENT`] if there is none.
///
/// Insertion and [`derive`] both go through here, which is what makes the
/// tree the same before and after a save.
pub(super) fn eligible_elder(
    neighbors: &[Vec<Vec<usize>>],
    vectors: &ChunkedVectors,
    dist_fn: DistanceFn,
    children: &[u16],
    limit: u16,
    node: usize,
) -> u32 {
    nearest_elder(neighbors, vectors, dist_fn, node, |elder| {
        children[elder] < limit
    })
}

/// The nearest mutual elder of `node` that `admit` accepts, ties broken by
/// slot, or [`NO_PARENT`].
fn nearest_elder(
    neighbors: &[Vec<Vec<usize>>],
    vectors: &ChunkedVectors,
    dist_fn: DistanceFn,
    node: usize,
    admit: impl Fn(usize) -> bool,
) -> u32 {
    let layer0 = |n: usize| neighbors[n].first().map_or(&[][..], Vec::as_slice);
    let mut best: Option<(f32, usize)> = None;
    for &elder in layer0(node) {
        if elder >= node || !admit(elder) || !layer0(elder).contains(&node) {
            continue;
        }
        let d = dist_fn(vectors.get(elder), vectors.get(node));
        let nearer =
            best.is_none_or(|(bd, b)| compare_distances(d, bd).then_with(|| elder.cmp(&b)).is_lt());
        if nearer {
            best = Some((d, elder));
        }
    }
    best.map_or(NO_PARENT, |(_, elder)| elder as u32)
}

/// Recomputes the whole tree from a graph, without changing the graph:
/// each slot's parent, and each slot's child count. Slots are visited in
/// order because a parent's count at the time a child arrived decides
/// whether it was eligible.
///
/// A node whose mutual elders are all full still takes the nearest one.
/// That never happens when the graph is derived with the distance kernel
/// that built it, because the recorded parent is always eligible on replay.
/// It can when the kernels differ: NEON against AVX2 or scalar, or one
/// engine reading the other's file. A last-bit difference flips a near-tie,
/// the counts shift, and a node could otherwise come out with no parent and
/// no protection. Going over the limit there is the lesser cost.
pub(super) fn derive(
    neighbors: &[Vec<Vec<usize>>],
    vectors: &ChunkedVectors,
    dist_fn: DistanceFn,
    limit: u16,
) -> (Vec<u32>, Vec<u16>) {
    let mut parent = Vec::with_capacity(neighbors.len());
    let mut children = vec![0u16; neighbors.len()];
    for node in 0..neighbors.len() {
        let mut p = eligible_elder(neighbors, vectors, dist_fn, &children, limit, node);
        if p == NO_PARENT {
            p = nearest_elder(neighbors, vectors, dist_fn, node, |_| true);
        }
        if p != NO_PARENT {
            children[p as usize] = children[p as usize].saturating_add(1);
        }
        parent.push(p);
    }
    (parent, children)
}

/// Counts layer-0 nodes the entry cannot reach, and nodes that cannot reach
/// the entry. Both are zero exactly when the layer is strongly connected.
#[cfg(test)]
pub(super) fn disconnected(neighbors: &[Vec<Vec<usize>>], entry: usize) -> (usize, usize) {
    let n = neighbors.len();
    let mut reverse: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (node, layers) in neighbors.iter().enumerate() {
        for &other in &layers[0] {
            reverse[other].push(node);
        }
    }
    let unreached = |forward: bool| {
        let mut seen = vec![false; n];
        seen[entry] = true;
        let mut stack = vec![entry];
        let mut reached = 1;
        while let Some(node) = stack.pop() {
            let list = if forward {
                &neighbors[node][0]
            } else {
                &reverse[node]
            };
            for &other in list {
                if !seen[other] {
                    seen[other] = true;
                    reached += 1;
                    stack.push(other);
                }
            }
        }
        n - reached
    };
    (unreached(true), unreached(false))
}
