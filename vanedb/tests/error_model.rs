//! The error model's public contract.
//!
//! The motivating use case is "load the index, build it if it isn't there".
//! That needs a missing file, a corrupt file and a failing disk to be three
//! distinguishable things, decided by variant rather than by message text.

use std::error::Error as _;
use std::io;

use vanedb::{ApproxIndex, FlatIndex, Metric, VaneError};

/// The repo's convention for scratch paths: process-unique, no dev-dependency.
fn scratch(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("vanedb-errors-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn missing_file_is_its_own_variant_and_keeps_the_io_kind() {
    let err = ApproxIndex::load("/nonexistent/directory/index.vane").unwrap_err();
    assert!(matches!(err, VaneError::FileNotFound { .. }), "got {err:?}");

    let source = err.source().expect("io-backed errors expose their source");
    let io_err = source
        .downcast_ref::<io::Error>()
        .expect("the source is the original io::Error");
    assert_eq!(io_err.kind(), io::ErrorKind::NotFound);
}

#[test]
fn a_garbage_file_is_corrupt_not_io() {
    let path = scratch("garbage").join("index.vane");
    std::fs::write(&path, b"this is not a vanedb file").unwrap();

    let err = ApproxIndex::load(&path).unwrap_err();
    assert!(matches!(err, VaneError::Corrupt { .. }), "got {err:?}");
    assert!(
        err.source().is_none(),
        "a corrupt file is a decoding failure, not an io failure"
    );
    let _ = std::fs::remove_file(&path);
}

#[test]
fn load_or_build_is_expressible_without_matching_on_strings() {
    let path = scratch("load-or-build").join("index.vane");
    let _ = std::fs::remove_file(&path);

    // The whole point of the split: this match compiles, and the missing-file
    // arm is reached without inspecting any message text.
    let index = match ApproxIndex::load(&path) {
        Ok(index) => index,
        Err(VaneError::FileNotFound { .. }) => ApproxIndex::builder(4, Metric::L2).build().unwrap(),
        Err(other) => panic!("expected a missing file, got {other:?}"),
    };

    index.add(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
    assert_eq!(index.len(), 1);
}

#[test]
fn a_zero_dimension_is_named_for_what_it_means() {
    // Named for the cause: a dimension of zero at construction.
    assert!(matches!(
        FlatIndex::new(0, Metric::L2),
        Err(VaneError::ZeroDimension)
    ));
}

#[test]
fn an_actually_empty_vector_still_reports_a_dimension_mismatch() {
    // An empty vector is a length mismatch, not a zero dimension.
    let index = FlatIndex::new(4, Metric::L2).unwrap();
    assert!(matches!(
        index.add(1, &[]),
        Err(VaneError::DimensionMismatch {
            expected: 4,
            got: 0
        })
    ));
}

#[test]
fn io_errors_convert_and_classify_themselves() {
    let missing: VaneError = io::Error::from(io::ErrorKind::NotFound).into();
    assert!(
        matches!(missing, VaneError::FileNotFound { .. }),
        "{missing:?}"
    );

    let denied: VaneError = io::Error::from(io::ErrorKind::PermissionDenied).into();
    assert!(matches!(denied, VaneError::Io { .. }), "{denied:?}");
}

#[test]
fn logical_errors_have_no_source() {
    for err in [
        VaneError::InvalidK,
        VaneError::ZeroDimension,
        VaneError::NotFound { id: 7 },
        VaneError::DuplicateId { id: 7 },
        VaneError::DimensionMismatch {
            expected: 4,
            got: 2,
        },
    ] {
        assert!(err.source().is_none(), "{err:?} should have no source");
    }
}

#[test]
fn a_failing_disk_is_distinguishable_from_a_missing_one() {
    // Saving into a path whose parent is a file, not a directory, is a real
    // io failure that is not NotFound.
    let dir = scratch("not-a-dir");
    let blocker = dir.join("blocker");
    std::fs::write(&blocker, b"x").unwrap();

    let index = ApproxIndex::builder(4, Metric::L2).build().unwrap();
    index.add(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
    let err = index.save(blocker.join("index.vane")).unwrap_err();

    assert!(
        !matches!(err, VaneError::Corrupt { .. }),
        "a write failure is not corruption: {err:?}"
    );
    assert!(
        err.source().is_some(),
        "io failures keep their source: {err:?}"
    );
    let _ = std::fs::remove_file(&blocker);
}
