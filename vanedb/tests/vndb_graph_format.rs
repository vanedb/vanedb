//! The VNDB v2 graph contract, anchored to golden fixtures.
//!
//! `vanedb/tests/fixtures/vndb_graph/*.vndb` are encoded from the field table
//! in `conformance/graph/README.md` by `conformance/graph/generate.py`, which
//! packs the bytes directly and never calls either engine. That independence
//! is the whole point: a codec validated only by its own round-trip agrees
//! with itself no matter what it does, which is the loop these fixtures exist
//! to break.
//!
//! Each fixture runs in both directions: the engine must read it, and
//! re-writing what it read must reproduce the bytes.
//!
//! Note what the second direction does *not* prove. It is a round trip
//! through the reader, so a transposition applied to both writer and reader
//! reproduces the bytes exactly. That is why the contents are asserted against
//! the field table here, and the graph geometry — entry slot, levels,
//! neighbour lists — in `approx::graph_format::spec_geometry`, which can see
//! fields the public API cannot.

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

/// Every header value in `l2_rng1.vndb`, read from
/// `conformance/graph/README.md`'s field table rather than from the engine.
///
/// This is the assertion the round-trip below cannot make. `load` then `save`
/// compares the reader against the writer, so transposing two header fields in
/// both directions reproduces the bytes exactly and passes — the very
/// misreading an independently generated fixture exists to catch. Only
/// comparing to the table catches it.
#[test]
fn the_header_matches_the_field_table() {
    let index = ApproxIndex::load(fixture("l2_rng1.vndb")).unwrap();
    assert_eq!(index.dimension(), 2, "dim, offset 16");
    assert_eq!(index.len(), 3, "count, offset 24");
    assert_eq!(index.capacity(), 4, "capacity hint, offset 32");
    assert_eq!(index.m(), 5, "M, offset 40");
    assert_eq!(index.ef_construction(), 16, "ef_construction, offset 48");
    assert_eq!(index.get_ef_search(), 32, "ef_search, offset 56");
    assert_eq!(index.seed(), 42, "seed, offset 64");
    assert_eq!(index.metric(), Metric::L2, "metric, offset 12");
}

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
fn the_fixture_contents_match_the_field_table() {
    // The header test pins the parameters; this pins the data. Reversing the
    // vector component order in writer and reader together round-trips byte
    // for byte and returns the wrong neighbour — the round-trip test cannot
    // see it, and neither can anything that only checks dim and metric.
    let index = ApproxIndex::load(fixture("l2_rng1.vndb")).unwrap();
    for (id, vector) in [
        (101u64, [1.0f32, 0.0]),
        (202, [0.0, 1.0]),
        (u64::MAX, [0.8, 0.2]),
    ] {
        assert!(index.contains(id), "id {id} from the table is missing");
        assert_eq!(
            index.get(id).unwrap(),
            vector.to_vec(),
            "vector for id {id}"
        );
    }

    // And the query the geometry decides: (1,0) is id 101 exactly.
    let hits = index.search(&[1.0, 0.0], 1).unwrap();
    assert_eq!(
        hits[0].id, 101,
        "nearest to (1,0) is the vector stored as (1,0)"
    );
    assert!(
        hits[0].distance.abs() < 1e-6,
        "exact match should be distance 0"
    );
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

// --- Negative cases -------------------------------------------------------
//
// Reading a valid fixture proves the accept path. It says nothing about the
// reject paths, and a loader that accepts everything reads every fixture
// perfectly. Each case below patches one field of a real fixture, so the file
// differs from a valid one in exactly one way.

fn corrupt(name: &str, patch: impl FnOnce(&mut Vec<u8>)) -> vanedb::VaneError {
    let mut bytes = std::fs::read(fixture(name)).unwrap();
    patch(&mut bytes);
    // A directory per call: these run in parallel, and a shared scratch path
    // had them deleting each other's files, which reads as a loader bug.
    static SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = scratch(&format!("negative_{n}"));
    let path = dir.join("patched.vndb");
    std::fs::write(&path, &bytes).unwrap();
    let err = ApproxIndex::load(&path).expect_err("a patched fixture must be rejected");
    let _ = std::fs::remove_dir_all(&dir);
    err
}

#[test]
fn a_wrong_magic_is_rejected() {
    let err = corrupt("l2_rng1.vndb", |b| b[0..4].copy_from_slice(b"BDNV"));
    assert!(format!("{err}").contains("magic"), "got: {err}");
}

#[test]
fn an_unsupported_version_is_rejected() {
    // Offset 4, per the field table. A future version must not be read as v2.
    let err = corrupt("l2_rng1.vndb", |b| {
        b[4..8].copy_from_slice(&7u32.to_le_bytes())
    });
    assert!(format!("{err}").contains("version"), "got: {err}");
}

#[test]
fn an_unsupported_kind_is_rejected() {
    // Offset 8. Kind 1 is the HNSW graph; nothing else is defined.
    let err = corrupt("l2_rng1.vndb", |b| {
        b[8..12].copy_from_slice(&9u32.to_le_bytes())
    });
    assert!(format!("{err}").contains("kind"), "got: {err}");
}

#[test]
fn an_invalid_metric_is_rejected() {
    // Offset 12: 0, 1 and 2 are defined.
    let err = corrupt("l2_rng1.vndb", |b| {
        b[12..16].copy_from_slice(&99u32.to_le_bytes())
    });
    assert!(format!("{err}").contains("metric"), "got: {err}");
}

#[test]
fn trailing_bytes_are_rejected() {
    // A file that decodes correctly and then continues is not this format.
    // Without this, a reader silently ignores appended content.
    let err = corrupt("l2_rng1.vndb", |b| b.extend_from_slice(&[0u8; 8]));
    assert!(format!("{err}").contains("trailing"), "got: {err}");
}

#[test]
fn a_truncated_file_is_rejected_at_every_length() {
    // Every prefix of a valid file must fail rather than read past the end.
    let bytes = std::fs::read(fixture("l2_rng1.vndb")).unwrap();
    let dir = scratch("truncated");
    for cut in 0..bytes.len() {
        let path = dir.join("cut.vndb");
        std::fs::write(&path, &bytes[..cut]).unwrap();
        assert!(
            ApproxIndex::load(&path).is_err(),
            "a {cut}-byte prefix of a {}-byte file must not load",
            bytes.len()
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn an_absurd_dimension_is_rejected() {
    // Offset 16 is dim. A few hundred bytes must not describe terabytes.
    // Named for what it asserts: that the file is refused. It does not
    // observe whether an allocation was attempted first.
    let err = corrupt("l2_rng1.vndb", |b| {
        b[16..24].copy_from_slice(&(1u64 << 62).to_le_bytes())
    });
    assert!(format!("{err}").contains("dimension"), "got: {err}");
}

#[test]
fn a_count_larger_than_the_file_is_rejected() {
    // Offset 24 is the stored slot count, cross-checked against the bytes
    // actually present. A count of u64::MAX would trip the MAX_ELEMENTS cap
    // first and never reach that check, so this uses a value that is small
    // enough to be plausible and still larger than the file can hold.
    let err = corrupt("l2_rng1.vndb", |b| {
        b[24..32].copy_from_slice(&1000u64.to_le_bytes())
    });
    // Positively, so a reworded or wrong message fails rather than passing
    // for lack of two words.
    let text = format!("{err}");
    assert!(
        text.contains("dimensions or parameters"),
        "should fail the file-size cross-check, not the element cap: {text}"
    );
}

#[test]
fn a_count_beyond_the_element_cap_is_rejected() {
    let err = corrupt("l2_rng1.vndb", |b| {
        b[24..32].copy_from_slice(&u64::MAX.to_le_bytes())
    });
    assert!(
        format!("{err}").contains("dimensions or parameters"),
        "got: {err}"
    );
}

#[test]
fn two_live_slots_with_the_same_id_are_rejected() {
    // `deleted_id_reuse.vndb` stores id 101 twice: slot 0 live, slot 1
    // tombstoned. That is legal — an id may be reused after removal. Clearing
    // slot 1's deleted flag makes both live, which is not, because lookups
    // would resolve one id to two slots.
    //
    // Slot layout per the field table: id u64, level u32, deleted u32, then
    // the vector, then the neighbour lists. Slot 1 begins at 160, so its
    // deleted flag is at 160 + 12.
    const SLOT1_DELETED: usize = 160 + 12;
    let bytes = std::fs::read(fixture("deleted_id_reuse.vndb")).unwrap();
    assert_eq!(
        u32::from_le_bytes(bytes[SLOT1_DELETED..SLOT1_DELETED + 4].try_into().unwrap()),
        1,
        "slot 1 is not tombstoned at the expected offset; layout changed"
    );

    let err = corrupt("deleted_id_reuse.vndb", |b| {
        b[SLOT1_DELETED..SLOT1_DELETED + 4].copy_from_slice(&0u32.to_le_bytes())
    });
    assert!(
        format!("{err}").contains("duplicate live"),
        "two live slots sharing an id must be rejected, got: {err}"
    );
}
