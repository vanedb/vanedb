//! `.unwrap_err()` and `#[derive(Debug)]` on a user struct both need `Debug`
//! on the public types, and neither compiled before these impls existed.

use vanedb::{ApproxIndex, DiskIndexBuilder, FlatIndex, Metric};

#[test]
fn unwrap_err_compiles_against_the_public_types() {
    // The idiom the whole suite had to avoid: it requires Debug on the Ok type.
    let err = ApproxIndex::builder(0, Metric::L2).build().unwrap_err();
    assert!(matches!(err, vanedb::VaneError::EmptyVector));
    let err = FlatIndex::new(0, Metric::L2).unwrap_err();
    assert!(matches!(err, vanedb::VaneError::EmptyVector));
}

#[test]
fn a_user_struct_holding_one_can_derive_debug() {
    #[derive(Debug)]
    #[allow(dead_code)]
    struct App {
        store: FlatIndex,
        index: ApproxIndex,
    }
    let app = App {
        store: FlatIndex::new(2, Metric::L2).unwrap(),
        index: ApproxIndex::builder(2, Metric::Cosine)
            .capacity(4)
            .build()
            .unwrap(),
    };
    let shown = format!("{app:?}");
    assert!(shown.contains("FlatIndex"), "{shown}");
    assert!(shown.contains("ApproxIndex"), "{shown}");
}

#[test]
fn debug_shows_identity_not_contents() {
    let store = FlatIndex::new(3, Metric::Dot).unwrap();
    store.add(1, &[1.0, 2.0, 3.0]).unwrap();
    let shown = format!("{store:?}");
    assert!(shown.contains("dim: 3"), "{shown}");
    assert!(shown.contains("len: 1"), "{shown}");
    // The vectors themselves must not be dumped.
    assert!(
        !shown.contains("1.0"),
        "contents leaked into Debug: {shown}"
    );

    let builder = DiskIndexBuilder::new(4, Metric::L2).unwrap();
    assert!(format!("{builder:?}").contains("DiskIndexBuilder"));
}
