//! The VNDB v2 graph contract, anchored to golden fixtures.
//!
//! `vanedb/tests/fixtures/vndb_graph/*.vndb` are encoded from the field table
//! in `conformance/graph/README.md` by `conformance/graph/generate.py`, which
//! packs the bytes directly and never calls either engine. That independence
//! is the whole point: a codec validated only by its own round-trip agrees
//! with itself no matter what it does, which is the loop these fixtures exist
//! to break.
//!
//! Each fixture runs in both directions, matching `vndb_format.rs` for the
//! disk format: the engine must read it, and re-writing what it read must
//! reproduce the bytes.

use std::path::PathBuf;

use vanedb::{ApproxIndex, Metric};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/vndb_graph")
        .join(name)
}

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("vanedb_graph_fixture_{tag}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Every fixture the spec generator produces.
const FIXTURES: [&str; 13] = [
    "l2_rng1.vndb",
    "l2_rng2.vndb",
    "l2_rng3.vndb",
    "cosine_rng1.vndb",
    "cosine_rng2.vndb",
    "cosine_rng3.vndb",
    "dot_rng1.vndb",
    "dot_rng2.vndb",
    "dot_rng3.vndb",
    "empty.vndb",
    "all_deleted.vndb",
    "deleted_entry.vndb",
    "deleted_id_reuse.vndb",
];

#[test]
fn every_fixture_loads() {
    for name in FIXTURES {
        let path = fixture(name);
        let index = ApproxIndex::load(&path).unwrap_or_else(|e| panic!("{name} must load: {e}"));
        // The table fixes dim = 2 for every case.
        assert_eq!(index.dimension(), 2, "{name}");
        // Metric is encoded at offset 12 and must survive the round trip.
        // Metrics per conformance/graph/generate.py, not guessed from the
        // filename: deleted_id_reuse is cosine, the other special cases L2.
        let expected = match name {
            n if n.starts_with("cosine") => Metric::Cosine,
            n if n.starts_with("dot") => Metric::Dot,
            "deleted_id_reuse.vndb" => Metric::Cosine,
            _ => Metric::L2,
        };
        assert_eq!(index.metric(), expected, "{name}");
    }
}

#[test]
fn a_loaded_fixture_is_searchable() {
    // empty and all_deleted have no live vectors by construction.
    for name in FIXTURES {
        if name == "empty.vndb" || name == "all_deleted.vndb" {
            continue;
        }
        let index = ApproxIndex::load(fixture(name)).unwrap();
        assert!(!index.is_empty(), "{name} should have live slots");
        let hits = index.search(&[1.0, 0.0], 2).unwrap();
        assert!(!hits.is_empty(), "{name} returned nothing");
    }
}

#[test]
fn rewriting_what_was_read_reproduces_the_fixture_byte_for_byte() {
    // The other direction. Reading correctly proves the decoder; only writing
    // the same bytes back proves the encoder agrees with the same table,
    // rather than with the decoder's private interpretation of it.
    for name in FIXTURES {
        let path = fixture(name);
        let original = std::fs::read(&path).unwrap();
        let index = ApproxIndex::load(&path).unwrap();

        let out = scratch(name.trim_end_matches(".vndb")).join(name);
        index.save(&out).unwrap();
        let written = std::fs::read(&out).unwrap();

        assert_eq!(
            written.len(),
            original.len(),
            "{name}: wrote {} bytes, fixture is {}",
            written.len(),
            original.len()
        );
        if written != original {
            let at = written
                .iter()
                .zip(&original)
                .position(|(a, b)| a != b)
                .unwrap();
            panic!("{name}: bytes diverge at offset {at}");
        }
        let _ = std::fs::remove_dir_all(out.parent().unwrap());
    }
}

#[test]
fn an_empty_and_an_all_deleted_graph_are_distinguishable() {
    // Both have no live vectors; they must not be the same file, or the
    // format cannot express "had entries, all removed".
    let empty = std::fs::read(fixture("empty.vndb")).unwrap();
    let all_deleted = std::fs::read(fixture("all_deleted.vndb")).unwrap();
    assert_ne!(empty, all_deleted);

    assert_eq!(ApproxIndex::load(fixture("empty.vndb")).unwrap().len(), 0);
    let dead = ApproxIndex::load(fixture("all_deleted.vndb")).unwrap();
    assert_eq!(dead.len(), 0);
    assert!(
        dead.tombstones() > 0,
        "all_deleted must retain its tombstones"
    );
}
