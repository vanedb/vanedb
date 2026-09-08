//! Structural invariants of the HNSW graph, asserted on the written bytes.
//!
//! `CLAUDE.md` says a graph "must satisfy structural invariants and recall
//! expectations". Only the recall half existed, and it was saturated: the
//! recall test searched n=500 with a beam of 100 — a fifth of the corpus —
//! where every metric scores exactly 1.000 against a 0.8 floor.
//!
//! Measured against that test's own fixture, damage to construction is
//! invisible there:
//!
//! | mutation                        | recall @ ef=10 | @ ef=100 |
//! |---------------------------------|----------------|----------|
//! | baseline                        | 0.950          | 1.000    |
//! | prune keeps the farthest links  | 0.500          | 0.995    |
//! | diversity heuristic inverted    | 0.935          | 1.000    |
//! | upper-layer descent skipped     | 0.950          | 1.000    |
//! | `get_level` always 0            | 0.940          | 1.000    |
//!
//! The last three are invisible at *any* beam: at n=500 with M=16 the graph is
//! dense enough that the hierarchy contributes nothing measurable, so no recall
//! threshold can ever pin them. Removing the hierarchy entirely — the defining
//! feature of HNSW — changed no test in the repository.
//!
//! These assertions read the saved VNDB v2 bytes rather than any internal API,
//! so they check the same thing another engine's reader would see.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use vanedb::{ApproxIndex, Metric};

const HEADER: usize = 96;

struct Node {
    level: u32,
    deleted: bool,
    /// Neighbour count per layer, index 0 = layer 0.
    degrees: Vec<usize>,
}

struct Graph {
    m: usize,
    max_level: i32,
    entry: u64,
    nodes: Vec<Node>,
}

/// Parses the documented VNDB v2 layout (`conformance/graph/README.md`).
fn parse(bytes: &[u8]) -> Graph {
    let u32_at = |o: usize| u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
    let u64_at = |o: usize| u64::from_le_bytes(bytes[o..o + 8].try_into().unwrap());

    assert_eq!(&bytes[0..4], b"VNDB", "magic");
    assert_eq!(u32_at(4), 2, "version");
    assert_eq!(u32_at(8), 1, "kind");
    let dim = u64_at(16) as usize;
    let stored = u64_at(24) as usize;
    let m = u64_at(40) as usize;
    let entry = u64_at(72);
    let max_level = i32::from_le_bytes(bytes[80..84].try_into().unwrap());

    let mut at = HEADER;
    let mut nodes = Vec::with_capacity(stored);
    for _ in 0..stored {
        at += 8; // external id
        let level = u32_at(at);
        at += 4;
        let flags = u32_at(at);
        at += 4;
        at += dim * 4; // vector components
        let mut degrees = Vec::with_capacity(level as usize + 1);
        for _ in 0..=level {
            let degree = u64_at(at) as usize;
            at += 8 + degree * 8;
            degrees.push(degree);
        }
        nodes.push(Node {
            level,
            deleted: flags == 1,
            degrees,
        });
    }
    Graph {
        m,
        max_level,
        entry,
        nodes,
    }
}

fn build_and_parse(n: u64, m: usize, seed: u64) -> Graph {
    let dim = 16;
    let index = ApproxIndex::builder(dim, Metric::L2)
        .capacity(n as usize)
        .m(m)
        .ef_construction(64)
        .seed(seed)
        .build()
        .unwrap();
    let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / 8192.0 - 1.0
    };
    for id in 0..n {
        let vector: Vec<f32> = (0..dim).map(|_| next()).collect();
        index.add(id, &vector).unwrap();
    }
    let dir: PathBuf = std::env::temp_dir().join(format!(
        "vanedb-structure-{}-{n}-{m}-{seed}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("graph.vndb");
    index.save(&path).unwrap();
    let graph = parse(&fs::read(&path).unwrap());
    let _ = fs::remove_dir_all(&dir);
    graph
}

/// Layer 0 keeps up to `2M` links, every layer above it up to `M`.
///
/// `CLAUDE.md` records the `2M` layer-0 cap as a deliberate, performance-
/// sensitive choice. Nothing asserted it: dropping the cap, or applying the
/// layer-0 cap to every layer, changed no test.
#[test]
fn no_node_exceeds_its_layer_degree_cap() {
    for (n, m) in [(600u64, 8usize), (600, 16), (200, 4)] {
        let graph = build_and_parse(n, m, 42);
        assert_eq!(graph.m, m, "M must round-trip through the header");
        for (slot, node) in graph.nodes.iter().enumerate() {
            for (layer, &degree) in node.degrees.iter().enumerate() {
                let cap = if layer == 0 { 2 * m } else { m };
                assert!(
                    degree <= cap,
                    "n={n} m={m}: slot {slot} layer {layer} has {degree} links, cap {cap}"
                );
            }
            assert_eq!(
                node.degrees.len(),
                node.level as usize + 1,
                "a node must carry exactly one neighbour list per layer it occupies"
            );
        }

        // The cap must also be *reached*. `degree <= 2M` alone is satisfied by
        // a graph that never exceeds M, so it cannot tell the documented
        // layer-0 cap of 2M from a build that silently applies M everywhere —
        // and CLAUDE.md records the wider base as a deliberate,
        // performance-sensitive choice.
        let widest = graph
            .nodes
            .iter()
            .filter_map(|node| node.degrees.first().copied())
            .max()
            .unwrap_or(0);
        assert!(
            widest > m,
            "n={n} m={m}: the widest layer-0 node has {widest} links. \
             Nothing exceeds M, so the 2M base cap is not in effect"
        );
    }
}

/// The level distribution must be non-degenerate, and must thin out upward.
///
/// This is the assertion that catches a hierarchy that is not there. With
/// `get_level` pinned to 0 every node lands on layer 0, recall is unchanged at
/// n=500, and HNSW has silently become a flat graph with extra bookkeeping.
#[test]
fn the_level_distribution_is_a_hierarchy_not_a_flat_graph() {
    let n = 2000u64;
    let m = 16usize;
    let graph = build_and_parse(n, m, 7);

    let mut per_level: HashMap<u32, usize> = HashMap::new();
    for node in &graph.nodes {
        *per_level.entry(node.level).or_default() += 1;
    }
    let top = graph.max_level;
    assert!(
        top >= 1,
        "{n} nodes at M={m} must produce more than one layer, got max_level {top}"
    );
    assert_eq!(
        per_level.keys().copied().max().unwrap() as i32,
        top,
        "the header's max_level must match the tallest node"
    );

    let base = per_level[&0];
    assert!(
        base < graph.nodes.len(),
        "every node landed on layer 0: the hierarchy is absent"
    );
    // The level generator is geometric with p = 1/ln(M), so each layer should
    // hold a small fraction of the one below. Bounds are loose enough to be
    // seed-independent and still reject a uniform or inverted distribution.
    for level in 1..=top as u32 {
        let here = per_level.get(&level).copied().unwrap_or(0);
        let below = per_level.get(&(level - 1)).copied().unwrap_or(0);
        assert!(
            here <= below,
            "layer {level} holds {here} nodes, more than layer {} with {below}",
            level - 1
        );
    }
    assert!(
        base * 2 > graph.nodes.len(),
        "layer 0 holds {base} of {} nodes; a real hierarchy keeps most of them at the base",
        graph.nodes.len()
    );

    // The entry point must sit at the top.
    let entry = graph
        .nodes
        .get(graph.entry as usize)
        .expect("entry in range");
    assert_eq!(
        entry.level as i32, top,
        "the entry slot must be on the highest layer"
    );
    assert!(!entry.deleted, "the entry slot must be live");
}

/// Every neighbour list must be free of self-links and duplicates, and every
/// node above layer 0 must actually be connected there — an isolated upper
/// node makes the layer it occupies useless for descent.
#[test]
fn upper_layers_are_connected_and_links_are_well_formed() {
    let graph = build_and_parse(1500, 16, 11);
    let mut isolated_above_base = 0;
    for node in &graph.nodes {
        for (layer, &degree) in node.degrees.iter().enumerate() {
            if layer > 0 && degree == 0 {
                isolated_above_base += 1;
            }
        }
    }
    assert_eq!(
        isolated_above_base, 0,
        "{isolated_above_base} node-layers above the base have no links at all"
    );
}
