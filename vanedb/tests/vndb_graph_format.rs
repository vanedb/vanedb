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
    let dir =
        std::env::temp_dir().join(format!("vanedb_graph_fixture_{}_{tag}", std::process::id()));
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
    assert!(!format!("{err}").is_empty());
}

#[test]
fn an_invalid_metric_is_rejected() {
    // Offset 12: 0, 1 and 2 are defined.
    let err = corrupt("l2_rng1.vndb", |b| {
        b[12..16].copy_from_slice(&99u32.to_le_bytes())
    });
    assert!(!format!("{err}").is_empty());
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
fn an_absurd_dimension_is_rejected_before_allocating() {
    // Offset 16 is dim. A few hundred bytes must not request terabytes.
    let err = corrupt("l2_rng1.vndb", |b| {
        b[16..24].copy_from_slice(&(1u64 << 62).to_le_bytes())
    });
    assert!(!format!("{err}").is_empty());
}

#[test]
fn a_count_larger_than_the_file_is_rejected() {
    // Offset 24 is the stored slot count. It must be cross-checked against
    // the bytes actually present, not trusted.
    let err = corrupt("l2_rng1.vndb", |b| {
        b[24..32].copy_from_slice(&u64::MAX.to_le_bytes())
    });
    assert!(!format!("{err}").is_empty());
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
        format!("{err}").contains("duplicate"),
        "two live slots sharing an id must be rejected, got: {err}"
    );
}
