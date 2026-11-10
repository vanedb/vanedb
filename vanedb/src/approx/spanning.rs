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
//! layer 0. The **parent** of `x` is its nearest mutual elder, ties broken
//! by slot. Pruning never evicts the edge between a node and its parent, in
//! either direction. Parents always have the smaller slot, so the protected
//! edges form a tree rooted at slot 0 that spans the layer both ways, and
//! layer 0 is strongly connected whatever start the upper layers hand the
//! beam.
//!
//! The parent is a function of the graph and the vectors, not extra state
//! with a history. Mutual elders of `x` are created only while `x` is being
//! inserted and can only be evicted afterwards, and the nearest one never
//! is, so it stays the nearest. That is why the tree is not persisted and
//! the VNDB graph format is unchanged: [`derive`] recomputes exactly the
//! tree the writer had, and an index resumed from a file keeps building the
//! graph it would have built without the round trip.
//!
//! The degree cap holds without further rules: an overflowing list is one
//! over its cap because a node was just pushed onto it, and that node has
//! no parent yet, so there is always an entry that may be dropped. When the
//! farthest entry is not a tree edge, which is the ordinary case, the
//! outcome is exactly the old truncation.
//!
//! A file written by an older release of either engine is loaded as it was
//! saved. Nodes it already stranded have no mutual elder, stay outside
//! the tree, and stay stranded until `compact()` rebuilds the graph.

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

/// Number of tree edges in `node`'s layer-0 list.
pub(super) fn tree_degree(parent: &[u32], list: &[usize], node: usize) -> usize {
    list.iter()
        .filter(|&&other| is_tree_edge(parent, node, other))
        .count()
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

/// The parent of `node`: its nearest mutual elder, ties broken by slot, or
/// [`NO_PARENT`] if it has none.
///
/// Insertion and [`derive`] both go through here, which is what makes the
/// tree the same before and after a save.
pub(super) fn nearest_mutual_elder(
    neighbors: &[Vec<Vec<usize>>],
    vectors: &ChunkedVectors,
    dist_fn: DistanceFn,
    node: usize,
) -> u32 {
    let layer0 = |n: usize| neighbors[n].first().map_or(&[][..], Vec::as_slice);
    let mut best: Option<(f32, usize)> = None;
    for &elder in layer0(node) {
        if elder >= node || !layer0(elder).contains(&node) {
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

/// Recomputes the whole tree from a graph, without changing the graph.
pub(super) fn derive(
    neighbors: &[Vec<Vec<usize>>],
    vectors: &ChunkedVectors,
    dist_fn: DistanceFn,
) -> Vec<u32> {
    (0..neighbors.len())
        .map(|node| nearest_mutual_elder(neighbors, vectors, dist_fn, node))
        .collect()
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
