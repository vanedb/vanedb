//! No panic, abort, or silent wrong answer from an untrusted file or a
//! caller-supplied parameter.
//!
//! Every case here aborted the process, panicked, or silently did nothing
//! before the fix. The C ABI turns any of those into a dead host application,
//! so they are correctness bugs rather than hygiene.

use std::fs;
use vanedb::{ApproxIndex, FlatIndex, Metric, VaneError};

fn scratch(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("vanedb-robust-{tag}-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn an_index_grown_past_its_capacity_hint_round_trips() {
    // capacity() is a hint and growing past it is supported, but save() wrote
    // the hint while load() rejected count > max_elements — so the file could
    // be written and never read back. Silent data loss.
    let path = scratch("grown").join("grown.vane");
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
    // BinaryHeap::with_capacity(k) on a caller-supplied k aborted with
    // "capacity overflow" long before any allocation was actually needed.
    let store = FlatIndex::new(2, Metric::L2).unwrap();
    store.add(1, &[0.0, 0.0]).unwrap();
    assert_eq!(store.search(&[0.0, 0.0], usize::MAX).unwrap().len(), 1);
    assert_eq!(store.search(&[0.0, 0.0], 2_000_000_000).unwrap().len(), 1);
}

#[test]
fn an_absurd_dimension_is_an_error_not_a_panic() {
    // dim * size_of::<f32>() wrapped to zero and divided by zero. dim also
    // arrives from a file, so this was reachable on the load path.
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
    // ids.len() * dim wrapped to zero, which matched the empty slice, so
    // add_batch returned Ok and inserted nothing.
    let store = FlatIndex::new(1usize << 63, Metric::L2).unwrap();
    let result = store.add_batch(&[1, 2], &[]);
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
    // Previously "dimension mismatch: expected 8, got 4" on a 4-dim index.
    assert_eq!(
        err.to_string(),
        "batch length mismatch: 2 ids need 8 floats at dimension 4, got 4"
    );
}

/// `mult` is fully derived from `m`, but it was deserialized and trusted. A
/// negative value made `get_level` return a negative level, which
/// `0..=level as usize` wrapped into a ~2^64 range, aborting on the next add.
#[test]
fn a_crafted_negative_mult_cannot_abort_the_process() {
    let dir = scratch("mult");
    let seed_path = dir.join("seed.vane");
    let idx = ApproxIndex::builder(4, Metric::L2)
        .capacity(8)
        .m(16)
        .seed(7)
        .build()
        .unwrap();
    idx.add(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
    idx.save(&seed_path).unwrap();

    let mut bytes = fs::read(&seed_path).unwrap();
    // Layout is fixint little-endian; mult sits at offset 68 (verified by
    // locating the derived 1/ln(m) value in a freshly written file).
    const MULT_OFFSET: usize = 68;
    let found = f64::from_le_bytes(bytes[MULT_OFFSET..MULT_OFFSET + 8].try_into().unwrap());
    assert!(
        (found - 1.0 / 16f64.ln()).abs() < 1e-12,
        "mult is not at offset {MULT_OFFSET}; layout changed, update this test (found {found})"
    );
    bytes[MULT_OFFSET..MULT_OFFSET + 8].copy_from_slice(&(-1000.0f64).to_le_bytes());

    let hostile = dir.join("negative-mult.vane");
    fs::write(&hostile, &bytes).unwrap();

    let loaded = ApproxIndex::load(&hostile).expect("mult is recomputed, so the file still loads");
    // The abort happened here, on the first insert after loading.
    for i in 10..40u64 {
        loaded.add(i, &[i as f32, 0.0, 0.0, 0.0]).unwrap();
    }
    assert_eq!(loaded.len(), 31);
    let _ = fs::remove_dir_all(&dir);
}
