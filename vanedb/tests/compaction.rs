//! Reclaiming tombstoned slots.
//!
//! `remove` tombstones rather than unlinking, so slots accumulate. A
//! remove-then-add cycle — the natural upsert — allocates a fresh slot every
//! time, so without compaction a long-lived index grows without bound even
//! at constant length.

use vanedb::{ApproxIndex, Metric};

fn index(n: u64) -> ApproxIndex {
    let idx = ApproxIndex::builder(4, Metric::L2)
        .capacity(n as usize)
        .seed(7)
        .build()
        .unwrap();
    for i in 0..n {
        idx.add(i, &[i as f32, 1.0, 2.0, 3.0]).unwrap();
    }
    idx
}

#[test]
fn tombstones_are_visible_so_a_caller_can_decide_to_compact() {
    let idx = index(20);
    assert_eq!(idx.tombstones(), 0);
    idx.remove(3).unwrap();
    idx.remove(4).unwrap();
    assert_eq!(idx.tombstones(), 2);
    assert_eq!(idx.len(), 18);
}

#[test]
fn compaction_reclaims_every_tombstoned_slot() {
    let idx = index(100);
    for i in (0..100).step_by(2) {
        idx.remove(i).unwrap();
    }
    assert_eq!(idx.tombstones(), 50);
    assert_eq!(idx.len(), 50);

    idx.compact().unwrap();

    assert_eq!(idx.tombstones(), 0, "compaction must reclaim the slots");
    assert_eq!(idx.len(), 50, "compaction must not lose live vectors");
}

#[test]
fn compaction_preserves_every_live_vector_and_its_id() {
    let idx = index(60);
    for i in (0..60).step_by(3) {
        idx.remove(i).unwrap();
    }
    let survivors: Vec<u64> = (0..60).filter(|i| i % 3 != 0).collect();

    idx.compact().unwrap();

    for &id in &survivors {
        assert!(idx.contains(id), "{id} lost in compaction");
        assert_eq!(
            idx.get_vector(id).unwrap(),
            vec![id as f32, 1.0, 2.0, 3.0],
            "{id} has the wrong vector after compaction"
        );
    }
    for id in (0..60).step_by(3) {
        assert!(!idx.contains(id), "{id} came back from the dead");
    }
}

#[test]
fn compaction_keeps_the_index_searchable() {
    let idx = index(200);
    for i in (0..200).step_by(2) {
        idx.remove(i).unwrap();
    }
    idx.compact().unwrap();

    let mut found = 0;
    for i in (1..200).step_by(2) {
        let hits = idx.search(&[i as f32, 1.0, 2.0, 3.0], 1).unwrap();
        assert!(hits.iter().all(|r| r.id % 2 == 1), "deleted id returned");
        if hits[0].id == i {
            found += 1;
        }
    }
    assert!(
        found >= 95,
        "recall collapsed after compaction: {found}/100"
    );
}

#[test]
fn an_upsert_loop_stays_bounded_after_compaction() {
    let idx = index(1);
    for i in 0..500 {
        idx.remove(0).unwrap();
        idx.add(0, &[i as f32, 1.0, 2.0, 3.0]).unwrap();
    }
    assert_eq!(idx.len(), 1);
    assert_eq!(idx.tombstones(), 500, "each cycle leaves a dead slot");

    idx.compact().unwrap();
    assert_eq!(idx.tombstones(), 0);
    assert_eq!(idx.len(), 1);
    assert_eq!(idx.search(&[499.0, 1.0, 2.0, 3.0], 1).unwrap()[0].id, 0);
}

#[test]
fn compacting_a_clean_index_changes_nothing_observable() {
    let idx = index(40);
    let before = idx.search(&[20.0, 1.0, 2.0, 3.0], 10).unwrap();
    idx.compact().unwrap();
    let after = idx.search(&[20.0, 1.0, 2.0, 3.0], 10).unwrap();
    assert_eq!(idx.len(), 40);
    assert_eq!(
        before, after,
        "compacting a tombstone-free index changed results"
    );
}

#[test]
fn compaction_survives_an_index_that_is_entirely_tombstones() {
    let idx = index(10);
    for i in 0..10 {
        idx.remove(i).unwrap();
    }
    idx.compact().unwrap();
    assert_eq!(idx.len(), 0);
    assert_eq!(idx.tombstones(), 0);
    assert!(idx.search(&[1.0, 1.0, 2.0, 3.0], 5).unwrap().is_empty());
    // And it must still accept new vectors afterwards.
    idx.add(99, &[9.0, 1.0, 2.0, 3.0]).unwrap();
    assert_eq!(idx.search(&[9.0, 1.0, 2.0, 3.0], 1).unwrap()[0].id, 99);
}
