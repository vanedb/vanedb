//! Exercise the public surface that other tests reach only incidentally.
//!
//! Coverage on the shipping engine was never measured — `.github/codecov.yml`
//! pointed exclusively at `cpp/`. Measuring it turned up whole groups of
//! public items no test touched: the duplicate `size`/`len` accessors that
//! exist for cross-engine parity, several `VaneError` constructors, and most
//! of the `Display` implementation. None of these are hard to reach; nothing
//! had reason to.
//!
//! A `Display` arm nobody renders is a real risk rather than a coverage
//! statistic: the string is what a Python or C caller sees, and a `{}`
//! placeholder naming a field that was renamed still compiles.

use vanedb::{ApproxIndex, DiskIndexBuilder, FlatIndex, Metric, VaneError};

/// `get` and `get_vector` are the same read under two names, for the same
/// reason `size`/`len` are (#85). The pair existed on `ApproxIndex` in Rust
/// and on every type in the Python and C bindings, but not on Rust's
/// `FlatIndex` or `DiskIndex` — so the one migration the pair exists to make
/// painless, `ApproxIndex` to an exact index, was the one that stopped
/// compiling.
#[test]
fn both_spellings_of_the_read_agree_on_every_type() {
    let flat = FlatIndex::new(2, Metric::L2).unwrap();
    flat.add(1, &[1.0, 0.0]).unwrap();
    assert_eq!(flat.get(1).unwrap(), flat.get_vector(1).unwrap());
    assert!(matches!(
        flat.get_vector(2),
        Err(VaneError::NotFound { id: 2 })
    ));

    let approx = ApproxIndex::builder(2, Metric::L2).build().unwrap();
    approx.add(1, &[1.0, 0.0]).unwrap();
    assert_eq!(approx.get(1).unwrap(), approx.get_vector(1).unwrap());

    #[cfg(feature = "disk")]
    {
        let path =
            std::env::temp_dir().join(format!("vanedb-read-spellings-{}.vndb", std::process::id()));
        let mut builder = DiskIndexBuilder::new(2, Metric::L2).unwrap();
        builder.add(1, &[1.0, 0.0]).unwrap();
        builder.save(&path).unwrap();
        // SAFETY: this test does not modify the file while it is mapped.
        let disk = unsafe { vanedb::DiskIndex::open(&path) }.unwrap();
        assert_eq!(disk.get(1).unwrap(), disk.get_vector(1).unwrap());
        assert!(matches!(
            disk.get_vector(2),
            Err(VaneError::NotFound { id: 2 })
        ));
        drop(disk);
        std::fs::remove_file(&path).ok();
    }
}

/// `size` and `len` are the same count under two names so a program is not
/// tied to one engine (#85). Both spellings must agree on every type, and on
/// every type `is_empty` must agree with them.
#[test]
fn both_spellings_of_the_count_agree_on_every_type() {
    let flat = FlatIndex::new(2, Metric::L2).unwrap();
    assert_eq!(flat.size(), 0);
    assert_eq!(flat.len(), 0);
    assert!(flat.is_empty());
    flat.add(1, &[1.0, 0.0]).unwrap();
    assert_eq!(flat.size(), flat.len());
    assert_eq!(flat.size(), 1);
    assert!(!flat.is_empty());

    let approx = ApproxIndex::builder(2, Metric::L2).build().unwrap();
    assert_eq!(approx.size(), 0);
    assert_eq!(approx.len(), 0);
    assert!(approx.is_empty());
    approx.add(1, &[1.0, 0.0]).unwrap();
    assert_eq!(approx.size(), approx.len());
    assert!(!approx.is_empty());

    let mut builder = DiskIndexBuilder::new(2, Metric::L2).unwrap();
    assert_eq!(builder.size(), 0);
    assert_eq!(builder.len(), 0);
    assert!(builder.is_empty());
    assert_eq!(builder.dimension(), 2);
    builder.add(1, &[1.0, 0.0]).unwrap();
    assert_eq!(builder.size(), builder.len());
    assert_eq!(builder.size(), 1);
    assert!(!builder.is_empty());
}

/// Every `Display` arm, rendered. These strings cross the FFI boundary as the
/// only failure detail a Python or C caller receives, so an arm that formats a
/// stale field name is a user-visible defect that still compiles.
#[test]
fn every_error_variant_renders_a_useful_message() {
    let cases: Vec<(VaneError, &[&str])> = vec![
        (
            VaneError::DimensionMismatch {
                expected: 4,
                got: 3,
            },
            &["dimension mismatch", "4", "3"],
        ),
        (
            VaneError::BatchLengthMismatch {
                ids: 3,
                vectors: 5,
                dim: 4,
            },
            &["batch length mismatch", "3", "12", "5"],
        ),
        (VaneError::ZeroDimension, &["dimension must be > 0"]),
        (VaneError::NotFound { id: 7 }, &["not found", "7"]),
        (VaneError::DuplicateId { id: 9 }, &["duplicate id", "9"]),
        (VaneError::InvalidK, &["k must be > 0"]),
        (
            VaneError::NonFiniteValue { input: "query" },
            &["query", "finite"],
        ),
        (
            VaneError::InvalidParameter("m must be >= 2"),
            &["invalid parameter", "m must be >= 2"],
        ),
        (
            VaneError::corrupt("header too short"),
            &["corrupt file", "header too short"],
        ),
        (
            VaneError::backend("Metal device unavailable"),
            &["backend unavailable", "Metal device unavailable"],
        ),
    ];
    for (error, fragments) in cases {
        let rendered = error.to_string();
        for fragment in fragments {
            assert!(
                rendered.contains(fragment),
                "{error:?} rendered as {rendered:?}, missing {fragment:?}"
            );
        }
        // (A length floor used to sit here. The shortest fragment asserted
        // above is "finite" at six characters, so every passing `contains`
        // already implies it — it could not fail.)
    }
}

/// The io paths carry their context forward, and `From<io::Error>` is the
/// blanket conversion `?` uses. `BatchLengthMismatch` reports the number of
/// floats the ids *needed*, computed with a saturating multiply so an absurd
/// id count cannot panic while formatting an error.
#[test]
fn io_errors_keep_their_context_and_batch_arithmetic_saturates() {
    let converted: VaneError = std::io::Error::from(std::io::ErrorKind::PermissionDenied).into();
    assert!(matches!(converted, VaneError::Io { .. }));
    assert!(converted.to_string().contains("io"));

    let missing = VaneError::from_io("open", std::io::Error::from(std::io::ErrorKind::NotFound));
    assert!(matches!(missing, VaneError::FileNotFound { .. }));
    assert!(missing.to_string().contains("open"));

    let absurd = VaneError::BatchLengthMismatch {
        ids: usize::MAX,
        vectors: 0,
        dim: 64,
    };
    let rendered = absurd.to_string();
    assert!(rendered.contains(&usize::MAX.to_string()));
}

/// `ApproxIndex` reports the construction parameters it was built with, so a
/// caller who loaded a file they did not write can check the query convention
/// matches. Nothing else reads these back.
#[test]
fn construction_parameters_are_readable_after_building() {
    let index = ApproxIndex::builder(3, Metric::Cosine)
        .capacity(64)
        .m(5)
        .ef_construction(32)
        .seed(1234)
        .build()
        .unwrap();
    assert_eq!(index.dimension(), 3);
    assert_eq!(index.metric(), Metric::Cosine);
    assert_eq!(index.capacity(), 64);
    assert_eq!(index.m(), 5);
    assert_eq!(index.ef_construction(), 32);
    assert_eq!(index.seed(), 1234);

    index.set_ef_search(99);
    assert_eq!(index.get_ef_search(), 99);
}
