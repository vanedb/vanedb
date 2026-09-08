//! The builder's defaults and clamps, asserted rather than described.
//!
//! A user-experience trial against this crate could complete every task from
//! rustdoc alone except five, each of which needed a probe program to
//! discover: the builder's default values, that `ef_construction` is silently
//! floored at `m` while its sibling `m` returns a hard error, that `ef_search`
//! below `k` is silently raised, that `ef_search` is persisted into the graph
//! file, and that cosine distance to a zero vector is 1.0.
//!
//! Those are now documented. A doc comment that nothing checks is a claim with
//! a shelf life, so each documented value is pinned here: change a default and
//! this file fails, which is the prompt to update the sentence that names it.

use vanedb::{ApproxIndex, Metric};

/// The values `ApproxIndex::builder` documents. A caller cannot reason about
/// their baseline recall without knowing `ef_search` starts at 50.
#[test]
fn the_builder_defaults_are_the_ones_the_docs_name() {
    let index = ApproxIndex::builder(4, Metric::L2).build().unwrap();
    assert_eq!(index.capacity(), 100_000, "documented default capacity");
    assert_eq!(index.m(), 16, "documented default m");
    assert_eq!(
        index.ef_construction(),
        200,
        "documented default ef_construction"
    );
    assert_eq!(index.seed(), 42, "documented default seed");
    assert_eq!(index.get_ef_search(), 50, "documented default ef_search");
}

/// `m(0)` is an error and `ef_construction(1)` is a silent clamp. Two sibling
/// setters with opposite policies: someone tuning for build speed sets
/// `ef_construction(8)` under `m = 16`, gets 16, and never learns why the
/// build did not get faster. The clamp is the right behaviour -- a beam
/// narrower than the number of links being chosen cannot fill them -- so it is
/// documented rather than changed.
#[test]
fn ef_construction_is_floored_at_m_while_m_itself_rejects() {
    for (m, requested, expected) in [(32usize, 1usize, 32usize), (64, 8, 64), (16, 200, 200)] {
        let index = ApproxIndex::builder(4, Metric::L2)
            .m(m)
            .ef_construction(requested)
            .build()
            .unwrap();
        assert_eq!(
            index.ef_construction(),
            expected,
            "m={m}, ef_construction={requested}"
        );
    }

    // The sibling rejects rather than clamping.
    assert!(ApproxIndex::builder(4, Metric::L2).m(0).build().is_err());
    assert!(ApproxIndex::builder(4, Metric::L2).m(1).build().is_err());
}

/// `ef_search` below `k` is raised to `k` for that search. Documented on
/// `SearchParams::ef_search` but not on `set_ef_search`, which is where a
/// caller sets it -- so every value at or below `k` behaves identically and
/// looks like the setting is being ignored.
#[test]
fn a_beam_narrower_than_k_is_raised_to_k() {
    let index = ApproxIndex::builder(1, Metric::L2)
        .capacity(64)
        .m(4)
        .ef_construction(32)
        .seed(7)
        .build()
        .unwrap();
    for id in 0..40u64 {
        index.add(id, &[id as f32]).unwrap();
    }

    let mut answers = Vec::new();
    for ef in [0usize, 1, 5, 10] {
        index.set_ef_search(ef);
        assert_eq!(
            index.get_ef_search(),
            ef,
            "the stored value is what was set"
        );
        let hits: Vec<u64> = index
            .search(&[0.0], 10)
            .unwrap()
            .iter()
            .map(|r| r.id)
            .collect();
        answers.push(hits);
    }
    assert!(
        answers.windows(2).all(|w| w[0] == w[1]),
        "every ef at or below k = 10 must search identically: {answers:?}"
    );
    // And every one of them returns k results rather than ef results.
    assert_eq!(answers[0].len(), 10);
}

/// `ef_search` travels in the graph file. Genuinely useful -- a service can
/// ship a tuned index rather than a tuned index plus a config note -- and
/// entirely undocumented: `save`'s doc said only "preserving stored slots and
/// links".
#[test]
fn ef_search_survives_a_save_and_load() {
    let dir = std::env::temp_dir().join(format!("vanedb-ef-persist-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("graph.vndb");

    let index = ApproxIndex::builder(2, Metric::L2).build().unwrap();
    index.add(1, &[1.0, 0.0]).unwrap();
    index.set_ef_search(321);
    index.save(&path).unwrap();

    let loaded = ApproxIndex::load(&path).unwrap();
    assert_eq!(
        loaded.get_ef_search(),
        321,
        "the tuned beam must travel with the file"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// Cosine distance to a zero vector is 1.0. Mathematically it is undefined --
/// the angle to a vector with no direction -- so the value is a choice, and an
/// unstated choice is one a caller discovers from a result they did not
/// expect. It is also the value a zero vector gets from *itself*.
#[test]
fn cosine_distance_to_a_zero_vector_is_one() {
    let index = ApproxIndex::builder(3, Metric::Cosine).build().unwrap();
    index.add(1, &[0.0, 0.0, 0.0]).unwrap();
    index.add(2, &[1.0, 0.0, 0.0]).unwrap();

    let to_zero = index.search(&[0.0, 0.0, 0.0], 2).unwrap();
    let zero_row = to_zero.iter().find(|r| r.id == 1).unwrap();
    assert_eq!(
        zero_row.distance, 1.0,
        "a zero vector is at distance 1 even from itself"
    );

    let from_unit = index.search(&[1.0, 0.0, 0.0], 2).unwrap();
    assert_eq!(from_unit.iter().find(|r| r.id == 1).unwrap().distance, 1.0);
    assert_eq!(from_unit.iter().find(|r| r.id == 2).unwrap().distance, 0.0);
}
