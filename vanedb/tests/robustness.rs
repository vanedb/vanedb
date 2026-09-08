//! No panic, abort, or silent wrong answer from an untrusted file or a
//! caller-supplied parameter — the C ABI turns any of them into a dead host
//! application.

use std::fs;
use vanedb::{ApproxIndex, FlatIndex, Metric, VaneError};

fn scratch(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("vanedb-robust-{tag}-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn an_index_grown_past_its_capacity_hint_round_trips() {
    // capacity() is a hint that may be grown past, so save() records what it
    // wrote: bounding the file by the hint makes it unreadable.
    let path = scratch("grown").join(format!("grown-{}.vane", std::process::id()));
    let idx = ApproxIndex::builder(4, Metric::L2)
        .capacity(2)
        .build()
        .unwrap();
    for i in 0..50u64 {
        idx.add(i, &[i as f32, 0.0, 0.0, 0.0]).unwrap();
    }
    idx.save(&path).unwrap();

    let loaded = ApproxIndex::load(&path).expect("an index that saved must load");
    assert_eq!(loaded.len(), 50);
    assert_eq!(loaded.get_vector(7).unwrap(), vec![7.0, 0.0, 0.0, 0.0]);
    let _ = fs::remove_file(&path);
}

#[test]
fn a_caller_supplied_k_cannot_abort_the_process() {
    // Reserving a caller-supplied k aborts with "capacity overflow".
    let store = FlatIndex::new(2, Metric::L2).unwrap();
    store.add(1, &[0.0, 0.0]).unwrap();
    assert_eq!(store.search(&[0.0, 0.0], usize::MAX).unwrap().len(), 1);
    assert_eq!(store.search(&[0.0, 0.0], 2_000_000_000).unwrap().len(), 1);
}

#[test]
fn an_absurd_dimension_is_an_error_not_a_panic() {
    // dim * size_of::<f32>() would wrap to zero and divide by zero. load()
    // needs the same bound, since dim arrives from a file.
    match ApproxIndex::builder(1usize << 62, Metric::L2)
        .capacity(1)
        .build()
    {
        Ok(_) => panic!("an index of 2^62 dimensions should not build"),
        Err(VaneError::InvalidParameter(_)) => {}
        Err(other) => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn a_batch_whose_length_overflows_is_rejected() {
    // A wrapped ids.len() * dim would match the empty slice and insert
    // nothing.
    let dim = usize::MAX / std::mem::size_of::<f32>();
    let store = FlatIndex::new(dim, Metric::L2).unwrap();
    let result = store.add_batch(&[1, 2, 3, 4, 5], &[]);
    assert!(
        matches!(result, Err(VaneError::InvalidParameter(_))),
        "expected an overflow error, got {result:?}"
    );
    assert_eq!(store.len(), 0);
}

#[test]
fn a_batch_length_mismatch_names_ids_and_floats_not_dimensions() {
    let store = FlatIndex::new(4, Metric::L2).unwrap();
    let err = store.add_batch(&[1, 2], &[1.0, 2.0, 3.0, 4.0]).unwrap_err();
    assert!(matches!(
        err,
        VaneError::BatchLengthMismatch {
            ids: 2,
            vectors: 4,
            dim: 4
        }
    ));
    // Named in ids and floats: "expected 8" alone would read as a dimension.
    assert_eq!(
        err.to_string(),
        "batch length mismatch: 2 ids need 8 floats at dimension 4, got 4"
    );
}

/// `mult` is derived from `m`, so load recomputes it. A negative value would
/// make `get_level` return a negative level, and `0..=level as usize` wraps
/// that into a ~2^64 range.
#[test]
fn a_crafted_negative_mult_cannot_abort_the_process() {
    let dir = scratch("mult");
    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/legacy_graph/v2_l2.hnsw");
    let mut bytes = fs::read(fixture).unwrap();
    // Layout is fixint little-endian; mult sits at offset 68 (verified by
    // locating the derived 1/ln(m) value in a freshly written file).
    const MULT_OFFSET: usize = 68;
    let found = f64::from_le_bytes(bytes[MULT_OFFSET..MULT_OFFSET + 8].try_into().unwrap());
    assert!(
        (found - 1.0 / 2f64.ln()).abs() < 1e-12,
        "mult is not at offset {MULT_OFFSET}; layout changed, update this test (found {found})"
    );
    bytes[MULT_OFFSET..MULT_OFFSET + 8].copy_from_slice(&(-1000.0f64).to_le_bytes());

    let hostile = dir.join(format!("negative-mult-{}.vane", std::process::id()));
    fs::write(&hostile, &bytes).unwrap();

    let loaded = ApproxIndex::load(&hostile).expect("mult is recomputed, so the file still loads");
    // Inserting is where a bad level would be used.
    for i in 10..40u64 {
        loaded.add(i, &[i as f32, 0.0]).unwrap();
    }
    assert_eq!(loaded.len(), 33);
    let _ = fs::remove_dir_all(&dir);
}
