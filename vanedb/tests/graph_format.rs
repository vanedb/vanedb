//! Independent VNDB graph fixtures anchor the portable graph contract.

use std::fs;
use std::path::PathBuf;
use vanedb::{ApproxIndex, VaneError};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/vndb_graph")
        .join(name)
}

fn temporary(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("vanedb-graph-{}-{name}.vndb", std::process::id()))
}

#[test]
fn graph_fixtures_roundtrip_byte_for_byte_and_support_mutation() {
    for name in [
        "l2_rng1",
        "cosine_rng1",
        "dot_rng1",
        "l2_rng2",
        "cosine_rng2",
        "dot_rng2",
        "l2_rng3",
        "cosine_rng3",
        "dot_rng3",
        "deleted_id_reuse",
        "empty",
        "deleted_entry",
        "all_deleted",
    ] {
        let original = fixture(&format!("{name}.vndb"));
        let index = ApproxIndex::load(&original).unwrap();
        let saved = temporary(name);
        index.save(&saved).unwrap();
        assert_eq!(
            fs::read(&original).unwrap(),
            fs::read(&saved).unwrap(),
            "{name}"
        );
        assert_eq!(index.dimension(), 2);
        assert_eq!(index.capacity(), 4);
        if name == "empty" || name == "all_deleted" {
            assert!(index.is_empty());
            assert!(index.search(&[1.0, 0.0], 3).unwrap().is_empty());
        } else {
            let hits = index.search(&[1.0, 0.0], 3).unwrap();
            assert_eq!(hits.len(), if name.starts_with("deleted_") { 2 } else { 3 });
            assert_eq!(
                hits[0].id,
                if name == "deleted_entry" {
                    u64::MAX
                } else {
                    101
                }
            );
            assert_eq!(
                hits[1].id,
                if name == "deleted_entry" {
                    202
                } else {
                    u64::MAX
                }
            );
            if name != "deleted_entry" {
                assert_eq!(
                    hits[0].distance,
                    if name.starts_with("dot") { -1.0 } else { 0.0 }
                );
            }
            assert_eq!(index.get_vector(u64::MAX).unwrap(), [0.8, 0.2]);
        }
        index.add(303, &[0.25, 0.75]).unwrap();
        index.save(&saved).unwrap();
        let reloaded = ApproxIndex::load(&saved).unwrap();
        fs::remove_file(&saved).unwrap();
        assert_eq!(reloaded.get_vector(303).unwrap(), [0.25, 0.75]);
        assert_eq!(reloaded.len(), index.len());
    }
}

#[test]
fn graph_reader_rejects_every_truncation_and_trailing_bytes() {
    let mut bytes = fs::read(fixture("l2_rng1.vndb")).unwrap();
    let path = temporary("truncated");
    for end in 0..bytes.len() {
        fs::write(&path, &bytes[..end]).unwrap();
        assert!(
            matches!(ApproxIndex::load(&path), Err(VaneError::Corrupt { .. })),
            "accepted prefix {end}"
        );
    }
    bytes.push(0);
    fs::write(&path, &bytes).unwrap();
    assert!(matches!(
        ApproxIndex::load(&path),
        Err(VaneError::Corrupt { .. })
    ));
    fs::remove_file(path).unwrap();
}

#[test]
fn graph_reader_rejects_invalid_parameters_identity_and_edges() {
    // Offsets are independently defined by conformance/graph/README.md.
    let base = fs::read(fixture("l2_rng1.vndb")).unwrap();
    let path = temporary("corrupted");
    let changes: Vec<(usize, Vec<u8>)> = vec![
        (4, 3u32.to_le_bytes().to_vec()),
        (8, 2u32.to_le_bytes().to_vec()),
        (12, 99u32.to_le_bytes().to_vec()),
        (16, 0u64.to_le_bytes().to_vec()),
        (24, u64::MAX.to_le_bytes().to_vec()),
        (32, 2u64.to_le_bytes().to_vec()),
        (40, u64::MAX.to_le_bytes().to_vec()),
        (40, 1u64.to_le_bytes().to_vec()),
        (48, 0u64.to_le_bytes().to_vec()),
        (72, 3u64.to_le_bytes().to_vec()),
        (72, 1u64.to_le_bytes().to_vec()),
        (72, u64::MAX.to_le_bytes().to_vec()),
        (80, (-1i32).to_le_bytes().to_vec()),
        (84, 0u32.to_le_bytes().to_vec()),
        (88, u64::MAX.to_le_bytes().to_vec()),
        (104, 33u32.to_le_bytes().to_vec()),
        (108, 2u32.to_le_bytes().to_vec()),
        (112, f32::NAN.to_le_bytes().to_vec()),
        (120, 5u64.to_le_bytes().to_vec()),
        (128, 0u64.to_le_bytes().to_vec()),
        (136, 1u64.to_le_bytes().to_vec()),
        (136, 3u64.to_le_bytes().to_vec()),
        (152, 1u64.to_le_bytes().to_vec()),
        (160, 101u64.to_le_bytes().to_vec()),
    ];
    for (offset, value) in changes {
        let mut bytes = base.clone();
        bytes[offset..offset + value.len()].copy_from_slice(&value);
        fs::write(&path, bytes).unwrap();
        assert!(
            matches!(ApproxIndex::load(&path), Err(VaneError::Corrupt { .. })),
            "accepted bad field at {offset}"
        );
    }
    fs::remove_file(path).unwrap();
}

#[test]
fn foreign_continuation_survives_removal_and_switches_after_insertion() {
    let index = ApproxIndex::load(fixture("l2_rng2.vndb")).unwrap();
    let original = fs::read(fixture("l2_rng2.vndb")).unwrap();
    let path = temporary("continuation");
    index.remove(101).unwrap();
    index.save(&path).unwrap();
    let removed = fs::read(&path).unwrap();
    assert_eq!(&removed[84..96], &original[84..96]);
    assert_eq!(&removed[272..], &original[272..]);
    index.add(303, &[0.25, 0.75]).unwrap();
    index.save(&path).unwrap();
    let added = fs::read(&path).unwrap();
    assert_eq!(&added[84..88], &1u32.to_le_bytes());
    assert_eq!(&added[88..96], &0u64.to_le_bytes());
    assert_eq!(ApproxIndex::load(&path).unwrap().len(), 3);
    fs::remove_file(path).unwrap();
}

#[test]
fn graph_reader_validates_continuation_words_and_framing() {
    let base = fs::read(fixture("l2_rng1.vndb")).unwrap();
    let state = fs::read(fixture("l2_rng2.vndb")).unwrap();
    let words = std::str::from_utf8(&state[272..]).unwrap();
    let path = temporary("rng-corrupt");
    for (kind, text) in [
        (1, "0".to_owned()),
        (2, String::new()),
        (2, words.replacen("42", "-42", 1)),
        (2, words.replacen("42", "+42", 1)),
        (2, words.replacen("42", "4294967296", 1)),
        (2, format!("{} 625", words.rsplit_once(' ').unwrap().0)),
        (3, words.to_owned()),
        (2, format!("{words}\u{a0}")),
    ] {
        let mut bytes = base.clone();
        bytes[84..88].copy_from_slice(&(kind as u32).to_le_bytes());
        bytes[88..96].copy_from_slice(&(text.len() as u64).to_le_bytes());
        bytes.extend_from_slice(text.as_bytes());
        fs::write(&path, bytes).unwrap();
        assert!(matches!(
            ApproxIndex::load(&path),
            Err(VaneError::Corrupt { .. })
        ));
    }
    fs::remove_file(path).unwrap();
}
