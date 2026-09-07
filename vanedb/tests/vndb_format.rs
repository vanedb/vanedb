//! The VNDB v1 on-disk contract, anchored to golden fixtures.
//!
//! `tests/fixtures/conformance/vndb/*.vndb` are written from the specification by
//! `generate.py`, not by either engine. That is the point: the Rust and C++
//! engines are otherwise only ever compared to *each other*, so a layout
//! change applied to both would pass every test in the repo. These fixtures
//! are the independent anchor.
//!
//! Each test runs in both directions — the engine must read the fixture, and
//! writing the same logical content must reproduce the fixture's bytes.

#![cfg(feature = "disk")]

use std::path::PathBuf;

use vanedb::{DiskIndex, DiskIndexBuilder, Metric};

const DIM: usize = 4;
const IDS: [u64; 6] = [10, 20, 30, 40, 50, 60];

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/conformance/vndb")
        .join(name)
}

/// Mirrors `generate.py`: row i, component d is (i * DIM + d) / 8.
fn rows() -> Vec<Vec<f32>> {
    (0..IDS.len())
        .map(|i| (0..DIM).map(|d| (i * DIM + d) as f32 / 8.0).collect())
        .collect()
}

fn cases() -> [(&'static str, Metric); 3] {
    [
        ("v1_l2.vndb", Metric::L2),
        ("v1_cosine.vndb", Metric::Cosine),
        ("v1_dot.vndb", Metric::Dot),
    ]
}

#[test]
fn every_metric_fixture_loads_with_its_contents_intact() {
    for (name, metric) in cases() {
        // SAFETY: these checked-in fixtures are not modified by the test.
        let index = unsafe { DiskIndex::open(fixture(name)) }
            .unwrap_or_else(|e| panic!("{name} failed to open: {e}"));

        assert_eq!(index.dimension(), DIM, "{name}");
        assert_eq!(index.size(), IDS.len(), "{name}");
        assert_eq!(index.metric(), metric, "{name}: metric decoded wrongly");

        for (i, id) in IDS.iter().enumerate() {
            assert!(index.contains(*id), "{name}: missing id {id}");
            assert_eq!(
                index.get(*id).unwrap(),
                rows()[i],
                "{name}: vector for id {id} decoded wrongly"
            );
        }
    }
}

/// The writer is half the contract. Without this, the reader could be fixed to
/// match a drifting writer and the fixture would still pass.
#[test]
fn writing_the_same_content_reproduces_the_fixture_byte_for_byte() {
    let dir = std::env::temp_dir().join(format!("vndb-golden-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    for (name, metric) in cases() {
        let mut builder = DiskIndexBuilder::new(DIM, metric).unwrap();
        for (i, id) in IDS.iter().enumerate() {
            builder.add(*id, &rows()[i]).unwrap();
        }
        let written = dir.join(name);
        builder.save(&written).unwrap();

        let ours = std::fs::read(&written).unwrap();
        let golden = std::fs::read(fixture(name)).unwrap();
        assert_eq!(
            ours.len(),
            golden.len(),
            "{name}: wrote {} bytes, fixture is {}",
            ours.len(),
            golden.len()
        );
        assert!(
            ours == golden,
            "{name}: bytes differ from the fixture at offset {:?}",
            ours.iter().zip(&golden).position(|(a, b)| a != b)
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// The header is fixed-width little-endian by specification, so assert the
/// literal bytes rather than round-tripping through the reader.
#[test]
fn the_header_matches_the_documented_layout() {
    for (name, metric) in cases() {
        let b = std::fs::read(fixture(name)).unwrap();
        assert_eq!(&b[0..4], b"VNDB", "{name}: magic");
        assert_eq!(
            u32::from_le_bytes(b[4..8].try_into().unwrap()),
            1,
            "{name}: version"
        );
        assert_eq!(u64::from_le_bytes(b[8..16].try_into().unwrap()), DIM as u64);
        assert_eq!(
            u64::from_le_bytes(b[16..24].try_into().unwrap()),
            IDS.len() as u64
        );
        let expected_metric = match metric {
            Metric::L2 => 0,
            Metric::Cosine => 1,
            Metric::Dot => 2,
            _ => unreachable!("Metric is non_exhaustive; extend this table"),
        };
        assert_eq!(
            u32::from_le_bytes(b[24..28].try_into().unwrap()),
            expected_metric,
            "{name}: metric field"
        );
        assert_eq!(
            u32::from_le_bytes(b[28..32].try_into().unwrap()),
            0,
            "{name}: reserved"
        );
        assert_eq!(
            b.len(),
            32 + IDS.len() * 8 + IDS.len() * DIM * 4,
            "{name}: size"
        );
    }
}
