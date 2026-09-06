//! Deleting from an approximate index.
//!
//! A tombstoned node must keep participating in graph traversal — its links
//! are what hold the neighbourhood together — while never appearing in a
//! result. Removing it from the graph instead would disconnect whatever it
//! was the bridge to.

use vanedb::{ApproxIndex, Metric, VaneError};

fn index(n: u64) -> ApproxIndex {
    let idx = ApproxIndex::builder(2, Metric::L2)
        .capacity(n as usize)
        .seed(7)
        .build()
        .unwrap();
    for i in 0..n {
        idx.add(i, &[i as f32, 0.0]).unwrap();
    }
    idx
}

#[test]
fn a_removed_vector_stops_being_found() {
    let idx = index(50);
    assert!(idx.contains(7));
    idx.remove(7).unwrap();
    assert!(!idx.contains(7));
    assert_eq!(idx.len(), 49);

    // Querying its exact position must not return it.
    let hits = idx.search(&[7.0, 0.0], 5).unwrap();
    assert!(
        hits.iter().all(|r| r.id != 7),
        "removed id came back: {hits:?}"
    );
    // The neighbours around it are still reachable.
    assert!(hits.iter().any(|r| r.id == 6 || r.id == 8), "{hits:?}");
}

#[test]
fn removing_the_same_id_twice_is_an_error() {
    let idx = index(10);
    idx.remove(3).unwrap();
    assert!(matches!(idx.remove(3), Err(VaneError::NotFound { id: 3 })));
}

#[test]
fn removing_an_unknown_id_is_an_error() {
    let idx = index(10);
    assert!(matches!(
        idx.remove(999),
        Err(VaneError::NotFound { id: 999 })
    ));
}

#[test]
fn the_id_can_be_reused_after_removal() {
    let idx = index(10);
    idx.remove(4).unwrap();
    idx.add(4, &[100.0, 0.0]).unwrap();
    assert!(idx.contains(4));
    assert_eq!(idx.len(), 10);
    assert_eq!(idx.get_vector(4).unwrap(), vec![100.0, 0.0]);
    // The new position is what is found, not the old one.
    let hits = idx.search(&[100.0, 0.0], 1).unwrap();
    assert_eq!(hits[0].id, 4);
}

#[test]
fn deleting_many_leaves_the_rest_searchable() {
    let idx = index(200);
    for i in (0..200).step_by(2) {
        idx.remove(i).unwrap();
    }
    assert_eq!(idx.len(), 100);
    // Every surviving odd id must still be its own nearest neighbour.
    let mut found = 0;
    for i in (1..200).step_by(2) {
        let hits = idx.search(&[i as f32, 0.0], 1).unwrap();
        assert!(
            hits.iter().all(|r| r.id % 2 == 1),
            "even id returned: {hits:?}"
        );
        if hits[0].id == i {
            found += 1;
        }
    }
    assert!(found >= 95, "recall collapsed after deletion: {found}/100");
}

#[test]
fn an_emptied_index_searches_without_panicking() {
    let idx = index(20);
    for i in 0..20 {
        idx.remove(i).unwrap();
    }
    assert_eq!(idx.len(), 0);
    assert!(idx.is_empty());
    assert!(idx.search(&[1.0, 0.0], 5).unwrap().is_empty());
}

#[test]
fn deletions_survive_save_and_load() {
    let dir = std::env::temp_dir().join(format!("vanedb-del-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("idx.vndb");

    let idx = index(40);
    idx.remove(5).unwrap();
    idx.remove(9).unwrap();
    idx.save(&path).unwrap();

    let loaded = ApproxIndex::load(&path).unwrap();
    assert_eq!(loaded.len(), 38);
    assert!(!loaded.contains(5));
    assert!(!loaded.contains(9));
    assert!(loaded.contains(6));
    let hits = loaded.search(&[5.0, 0.0], 5).unwrap();
    assert!(hits.iter().all(|r| r.id != 5 && r.id != 9), "{hits:?}");

    let _ = std::fs::remove_dir_all(&dir);
}

/// The entry point is where every search starts. Deleting it must not strand
/// the graph — the classic tombstone failure mode.
#[test]
fn removing_the_entry_point_keeps_the_graph_searchable() {
    let idx = index(300);
    // The first inserted vector is the entry point until a higher level is
    // drawn; remove a spread of early ids to be sure it goes.
    for id in 0..8 {
        idx.remove(id).unwrap();
    }
    let mut found = 0;
    for i in 8..300u64 {
        let hits = idx.search(&[i as f32, 0.0], 1).unwrap();
        assert!(
            hits.iter().all(|r| r.id >= 8),
            "deleted entry-point region returned"
        );
        if hits[0].id == i {
            found += 1;
        }
    }
    assert!(
        found >= 277,
        "recall collapsed after deleting the entry point: {found}/292"
    );
}

/// Deleting most of the graph should degrade gracefully, not silently return
/// nothing for live vectors that are still there.
#[test]
fn a_mostly_tombstoned_index_still_finds_its_survivors() {
    let idx = index(500);
    for id in 0..500 {
        if id % 25 != 0 {
            idx.remove(id).unwrap();
        }
    }
    assert_eq!(idx.len(), 20);
    let mut found = 0;
    for id in (0..500).step_by(25) {
        let hits = idx.search(&[id as f32, 0.0], 1).unwrap();
        assert!(!hits.is_empty(), "no result at all for live id {id}");
        if hits[0].id == id {
            found += 1;
        }
    }
    assert!(
        found >= 18,
        "survivors unreachable at 96% tombstones: {found}/20"
    );
}

/// `remove` then `add` takes the write lock twice, so a concurrent reader can
/// observe the id missing in between. `upsert` does both under one lock.
///
/// What this test actually proves: it fails against a `remove`-then-`add`
/// implementation with a `yield_now` between the two, and passes against
/// `upsert`. Against an un-widened `remove`-then-`add` it also passes — the
/// natural window is microseconds and a sampling reader misses it. So this
/// catches the class of bug but is not a guarantee; the guarantee comes from
/// `upsert` taking the lock exactly once, which is visible in the code.
#[test]
fn upsert_is_atomic_for_concurrent_readers() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    let idx = Arc::new(index(200));
    let stop = Arc::new(AtomicBool::new(false));
    let missing = Arc::new(AtomicBool::new(false));

    let reader = {
        let (idx, stop, missing) = (idx.clone(), stop.clone(), missing.clone());
        std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                if !idx.contains(42) {
                    missing.store(true, Ordering::Relaxed);
                }
            }
        })
    };

    for i in 0..2000 {
        idx.upsert(42, &[i as f32, 0.0]).unwrap();
    }
    stop.store(true, Ordering::Relaxed);
    reader.join().unwrap();

    assert!(
        !missing.load(Ordering::Relaxed),
        "a concurrent reader saw id 42 disappear during upsert"
    );
    assert_eq!(idx.len(), 200);
    assert_eq!(idx.search(&[1999.0, 0.0], 1).unwrap()[0].id, 42);
}

#[test]
fn upsert_inserts_when_the_id_is_new() {
    let idx = index(10);
    idx.upsert(999, &[999.0, 0.0]).unwrap();
    assert_eq!(idx.len(), 11);
    assert_eq!(idx.search(&[999.0, 0.0], 1).unwrap()[0].id, 999);
}
