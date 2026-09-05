//! Regression coverage for #38: two saves whose destinations share a stem must
//! not fight over a single temporary file.
#![cfg(feature = "disk")]

use std::sync::{Arc, Barrier};
use std::thread;

use vanedb::{DiskStore, DiskStoreBuilder, Index, Metric};

const DIM: usize = 4;
const N: u64 = 64;
const ROUNDS: usize = 25;

fn vector(i: u64) -> Vec<f32> {
    vec![i as f32, 1.0, 2.0, 3.0]
}

/// `index.bin` and `index.idx` share the stem `index`, so deriving the
/// temporary path with `with_extension("tmp")` made both writers target
/// `index.tmp`.
#[test]
fn concurrent_saves_sharing_a_stem_both_succeed() {
    let dir = std::env::temp_dir().join(format!("vanedb-issue-38-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let store_path = dir.join("index.bin");
    let index_path = dir.join("index.idx");

    for round in 0..ROUNDS {
        let mut builder = DiskStoreBuilder::new(DIM, Metric::L2).unwrap();
        let index = Index::builder(DIM, Metric::L2)
            .capacity(N as usize)
            .build()
            .unwrap();
        for i in 0..N {
            builder.add(i, &vector(i)).unwrap();
            index.add(i, &vector(i)).unwrap();
        }

        let barrier = Arc::new(Barrier::new(2));
        let (store_barrier, index_barrier) = (Arc::clone(&barrier), Arc::clone(&barrier));
        let (store_dest, index_dest) = (store_path.clone(), index_path.clone());

        let store_writer = thread::spawn(move || {
            store_barrier.wait();
            builder.save(&store_dest)
        });
        let index_writer = thread::spawn(move || {
            index_barrier.wait();
            index.save(&index_dest)
        });

        if let Err(e) = store_writer.join().unwrap() {
            panic!("round {round}: store save failed: {e}");
        }
        if let Err(e) = index_writer.join().unwrap() {
            panic!("round {round}: index save failed: {e}");
        }

        let store = DiskStore::open(&store_path)
            .unwrap_or_else(|e| panic!("round {round}: store reload failed: {e}"));
        assert_eq!(store.size(), N as usize, "round {round}");
        assert_eq!(
            store.search(&vector(7), 1).unwrap()[0].id,
            7,
            "round {round}"
        );

        let reloaded = Index::load(&index_path)
            .unwrap_or_else(|e| panic!("round {round}: index reload failed: {e}"));
        assert_eq!(
            reloaded.search(&vector(7), 1).unwrap()[0].id,
            7,
            "round {round}"
        );
    }

    let _ = std::fs::remove_dir_all(&dir);
}

/// A failed save must not leave its temporary file behind for the next writer
/// to trip over — unique temp names make orphans accumulate otherwise.
#[test]
fn failed_save_leaves_no_temporary_file() {
    let dir = std::env::temp_dir().join(format!("vanedb-issue-38-cleanup-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let mut builder = DiskStoreBuilder::new(DIM, Metric::L2).unwrap();
    builder.add(1, &vector(1)).unwrap();

    // A directory as the destination makes the final rename fail.
    let dest = dir.join("occupied.bin");
    std::fs::create_dir(&dest).unwrap();
    assert!(builder.save(&dest).is_err());

    let leftovers: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|name| name.ends_with(".tmp"))
        .collect();
    assert!(
        leftovers.is_empty(),
        "temporary files left behind: {leftovers:?}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}
