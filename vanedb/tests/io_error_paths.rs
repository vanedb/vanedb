//! The failure side of every persistence entry point.
//!
//! Coverage on the shipping engine showed the save and load paths well
//! exercised on success and barely at all on failure: most of what was
//! unreached in `disk.rs` and `approx/persistence.rs` was the `map_err`
//! closure on an I/O call. Those closures are not decoration — they choose the
//! `VaneError` variant a caller branches on, and `Corrupt` ("retrying will not
//! help") versus `Io` ("retrying may help") is a decision the caller cannot
//! make from a string.
//!
//! Only failures reachable without root or a special filesystem are here. A
//! full disk mid-write, or an fsync that fails after a successful write, needs
//! fault injection and is out of scope.

use std::fs;
use std::path::PathBuf;

use vanedb::{ApproxIndex, DiskIndexBuilder, Metric, VaneError};

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("vanedb-io-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

/// A destination whose parent directory does not exist fails at `create`,
/// before a byte is written. The temporary file is named beside the
/// destination, so this is the first call that touches the filesystem.
#[test]
fn saving_under_a_missing_directory_reports_io_not_corruption() {
    let dir = scratch("missing-parent");
    let nowhere = dir.join("does/not/exist/index.vndb");

    let index = ApproxIndex::builder(2, Metric::L2).build().unwrap();
    index.add(1, &[1.0, 0.0]).unwrap();
    let err = index.save(&nowhere).unwrap_err();
    assert!(
        matches!(err, VaneError::Io { .. } | VaneError::FileNotFound { .. }),
        "graph save under a missing parent must be an io error, got {err:?}"
    );
    assert!(err.to_string().contains("create"), "{err}");

    let mut builder = DiskIndexBuilder::new(2, Metric::L2).unwrap();
    builder.add(1, &[1.0, 0.0]).unwrap();
    let err = builder.save(&nowhere).unwrap_err();
    assert!(
        matches!(err, VaneError::Io { .. } | VaneError::FileNotFound { .. }),
        "disk save under a missing parent must be an io error, got {err:?}"
    );

    let _ = fs::remove_dir_all(&dir);
}

/// A path that is not there at all is `FileNotFound`, which exists as its own
/// variant so "load it, or build it if absent" can branch on the variant
/// rather than parse a message.
#[test]
fn loading_something_absent_is_file_not_found() {
    let dir = scratch("absent");
    let absent = dir.join("nothing-here.vndb");

    let err = ApproxIndex::load(&absent).unwrap_err();
    assert!(
        matches!(err, VaneError::FileNotFound { .. }),
        "absent graph must be FileNotFound, got {err:?}"
    );

    #[cfg(feature = "disk")]
    {
        // Safe: the path does not exist, so nothing is mapped.
        let err = unsafe { vanedb::DiskIndex::open(&absent) }.unwrap_err();
        assert!(
            matches!(err, VaneError::FileNotFound { .. }),
            "absent disk store must be FileNotFound, got {err:?}"
        );
    }

    let _ = fs::remove_dir_all(&dir);
}

/// A directory where a file belongs is an I/O failure, not a corrupt file.
/// The distinction matters: `Corrupt` tells a caller the bytes are wrong and
/// retrying is pointless, which would be the wrong advice here.
#[test]
fn loading_a_directory_is_an_io_error_not_corruption() {
    let dir = scratch("is-a-directory");

    let err = ApproxIndex::load(&dir).unwrap_err();
    assert!(
        !matches!(err, VaneError::Corrupt { .. }),
        "a directory is not a corrupt file: {err:?}"
    );

    #[cfg(feature = "disk")]
    {
        // Safe: a directory cannot be mapped, so open fails before mmap.
        let err = unsafe { vanedb::DiskIndex::open(&dir) }.unwrap_err();
        assert!(
            !matches!(err, VaneError::Corrupt { .. }),
            "a directory is not a corrupt store: {err:?}"
        );
    }

    let _ = fs::remove_dir_all(&dir);
}

/// A file too short to hold a header is corrupt, and must say so rather than
/// panicking on the slice. Both formats declare a fixed-width header, so every
/// length below it is rejected before any field is read.
#[test]
fn a_file_shorter_than_its_header_is_corrupt_at_every_length() {
    let dir = scratch("short");
    for len in 0..40usize {
        let path = dir.join(format!("short-{len}.vndb"));
        fs::write(&path, vec![0u8; len]).unwrap();

        let err = ApproxIndex::load(&path).unwrap_err();
        assert!(
            matches!(err, VaneError::Corrupt { .. }),
            "a {len}-byte graph must be Corrupt, got {err:?}"
        );

        #[cfg(feature = "disk")]
        {
            // Safe: this test owns the file and never truncates it.
            let err = unsafe { vanedb::DiskIndex::open(&path) }.unwrap_err();
            assert!(
                matches!(err, VaneError::Corrupt { .. }),
                "a {len}-byte store must be Corrupt, got {err:?}"
            );
        }
    }
    let _ = fs::remove_dir_all(&dir);
}

/// A successful save leaves no temporary file behind, at either format. The
/// temp name is a hidden sibling of the destination, so a leak would sit in
/// the user's data directory.
#[test]
fn a_successful_save_leaves_no_temporary_behind() {
    let dir = scratch("no-temp-leak");

    let index = ApproxIndex::builder(2, Metric::L2).build().unwrap();
    index.add(1, &[1.0, 0.0]).unwrap();
    index.save(dir.join("graph.vndb")).unwrap();

    let mut builder = DiskIndexBuilder::new(2, Metric::L2).unwrap();
    builder.add(1, &[1.0, 0.0]).unwrap();
    builder.save(dir.join("store.vndb")).unwrap();

    let leftovers: Vec<String> = fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.contains("tmp"))
        .collect();
    assert!(
        leftovers.is_empty(),
        "temporary files left behind: {leftovers:?}"
    );

    let _ = fs::remove_dir_all(&dir);
}
