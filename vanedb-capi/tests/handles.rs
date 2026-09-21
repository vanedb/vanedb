//! Handles are ids, and an id that is not live must fail every entry point
//! with `VANEDB_RS_INVALID_HANDLE` rather than crash (RFC 0002 stage 1).
//!
//! The motivating defect: a `ctypes` caller without `restype` receives the
//! returned handle truncated to a C `int`. With pointer handles the next call
//! dereferenced garbage and the process died inside libffi. With ids the
//! truncation clears the generation half, which is never 0, so the lookup
//! fails and the call reports a code.
//!
//! The set of entry points under test is read from the header, not typed
//! here: a function added to the ABI without a probe below fails
//! `every_handle_taking_entry_point_has_a_probe`, so the coverage cannot
//! silently shrink as the surface grows.

mod common;

use std::collections::BTreeSet;
use std::ffi::c_void;
use std::ptr::{null, null_mut};

use vanedb_capi::*;

const NULL: u64 = VANEDB_RS_NULL_HANDLE;

/// Calls one entry point with `handle` and otherwise valid arguments, and
/// reports whether the return value was the failure value for its type.
/// Frees have no return value, so they report through the error code alone.
struct Probe {
    name: &'static str,
    kind: &'static str,
    rejected: fn(u64) -> bool,
}

static V2: [f32; 2] = [1.0, 0.0];
static IDS: [u64; 1] = [1];

unsafe extern "C-unwind" fn accept_all(_: u64, _: *mut c_void) -> bool {
    true
}

/// Clears the thread's error state through a call that cannot fail, so a
/// free probe's verdict is its own and not the previous probe's.
fn settle() {
    let n = unsafe { vanedb_rs_l2_sq(V2.as_ptr(), V2.as_ptr(), 2) };
    assert_eq!(n, 0.0);
    assert_eq!(vanedb_rs_last_error(), VANEDB_RS_OK);
}

fn probes() -> Vec<Probe> {
    macro_rules! probe {
        ($kind:literal, $name:ident, |$h:ident| $body:expr) => {
            Probe {
                name: stringify!($name),
                kind: $kind,
                // Safe entry points (no raw pointer) make the block redundant.
                #[allow(unused_unsafe)]
                rejected: |$h: u64| -> bool { unsafe { $body } },
            }
        };
    }
    vec![
        // -- store ----------------------------------------------------------
        probe!("store", vanedb_rs_store_add, |h| vanedb_rs_store_add(
            h,
            1,
            V2.as_ptr()
        ) == 1),
        probe!(
            "store",
            vanedb_rs_store_add_batch,
            |h| vanedb_rs_store_add_batch(h, IDS.as_ptr(), V2.as_ptr(), 1) == 1
        ),
        probe!("store", vanedb_rs_store_search, |h| {
            let (mut ids, mut ds) = ([0u64; 1], [0f32; 1]);
            vanedb_rs_store_search(h, V2.as_ptr(), 1, ids.as_mut_ptr(), ds.as_mut_ptr()) == 0
        }),
        probe!("store", vanedb_rs_store_search_filtered, |h| {
            let (mut ids, mut ds) = ([0u64; 1], [0f32; 1]);
            vanedb_rs_store_search_filtered(
                h,
                V2.as_ptr(),
                1,
                Some(accept_all),
                null_mut(),
                null(),
                0,
                null(),
                0,
                ids.as_mut_ptr(),
                ds.as_mut_ptr(),
            ) == 0
        }),
        probe!("store", vanedb_rs_store_free, |h| {
            settle();
            vanedb_rs_store_free(h);
            vanedb_rs_last_error() == VANEDB_RS_INVALID_HANDLE
        }),
        probe!("store", vanedb_rs_store_len, |h| vanedb_rs_store_len(h)
            == 0),
        probe!(
            "store",
            vanedb_rs_store_dimension,
            |h| vanedb_rs_store_dimension(h) == 0
        ),
        probe!("store", vanedb_rs_store_metric, |h| vanedb_rs_store_metric(
            h
        ) == 0),
        probe!("store", vanedb_rs_store_contains, |h| {
            !vanedb_rs_store_contains(h, 1)
        }),
        probe!("store", vanedb_rs_store_get, |h| {
            let mut out = [0f32; 2];
            vanedb_rs_store_get(h, 1, out.as_mut_ptr()) == 1
        }),
        probe!("store", vanedb_rs_store_get_vector, |h| {
            let mut out = [0f32; 2];
            vanedb_rs_store_get_vector(h, 1, out.as_mut_ptr()) == 1
        }),
        probe!("store", vanedb_rs_store_remove, |h| vanedb_rs_store_remove(
            h, 1
        ) == 1),
        // -- index ----------------------------------------------------------
        probe!("index", vanedb_rs_index_add, |h| vanedb_rs_index_add(
            h,
            1,
            V2.as_ptr()
        ) == 1),
        probe!(
            "index",
            vanedb_rs_index_add_batch,
            |h| vanedb_rs_index_add_batch(h, IDS.as_ptr(), V2.as_ptr(), 1) == 1
        ),
        probe!("index", vanedb_rs_index_search, |h| {
            let (mut ids, mut ds) = ([0u64; 1], [0f32; 1]);
            vanedb_rs_index_search(h, V2.as_ptr(), 1, 0, ids.as_mut_ptr(), ds.as_mut_ptr()) == 0
        }),
        probe!("index", vanedb_rs_index_search_filtered, |h| {
            let (mut ids, mut ds) = ([0u64; 1], [0f32; 1]);
            vanedb_rs_index_search_filtered(
                h,
                V2.as_ptr(),
                1,
                0,
                Some(accept_all),
                null_mut(),
                null(),
                0,
                null(),
                0,
                ids.as_mut_ptr(),
                ds.as_mut_ptr(),
            ) == 0
        }),
        probe!("index", vanedb_rs_index_save, |h| {
            // A path that must never be written: the handle is rejected first.
            let path = std::ffi::CString::new(
                std::env::temp_dir()
                    .join(format!("vanedb-invalid-handle-{}.vndb", std::process::id()))
                    .to_str()
                    .unwrap(),
            )
            .unwrap();
            let rejected = vanedb_rs_index_save(h, path.as_ptr()) == 1;
            assert!(!std::path::Path::new(path.to_str().unwrap()).exists());
            rejected
        }),
        probe!("index", vanedb_rs_index_save_to_buffer, |h| {
            let mut written = 7usize;
            let rejected = vanedb_rs_index_save_to_buffer(h, null_mut(), 0, &mut written) == 1;
            assert_eq!(written, 7, "a rejected call must not touch outputs");
            rejected
        }),
        probe!("index", vanedb_rs_index_free, |h| {
            settle();
            vanedb_rs_index_free(h);
            vanedb_rs_last_error() == VANEDB_RS_INVALID_HANDLE
        }),
        probe!("index", vanedb_rs_index_metric, |h| vanedb_rs_index_metric(
            h
        ) == 0),
        probe!("index", vanedb_rs_index_len, |h| vanedb_rs_index_len(h)
            == 0),
        probe!(
            "index",
            vanedb_rs_index_dimension,
            |h| vanedb_rs_index_dimension(h) == 0
        ),
        probe!("index", vanedb_rs_index_contains, |h| {
            !vanedb_rs_index_contains(h, 1)
        }),
        probe!("index", vanedb_rs_index_get_vector, |h| {
            let mut out = [0f32; 2];
            vanedb_rs_index_get_vector(h, 1, out.as_mut_ptr()) == 1
        }),
        probe!("index", vanedb_rs_index_get, |h| {
            let mut out = [0f32; 2];
            vanedb_rs_index_get(h, 1, out.as_mut_ptr()) == 1
        }),
        probe!("index", vanedb_rs_index_upsert, |h| vanedb_rs_index_upsert(
            h,
            1,
            V2.as_ptr()
        ) == 1),
        probe!("index", vanedb_rs_index_remove, |h| vanedb_rs_index_remove(
            h, 1
        ) == 1),
        probe!(
            "index",
            vanedb_rs_index_tombstones,
            |h| vanedb_rs_index_tombstones(h) == 0
        ),
        probe!(
            "index",
            vanedb_rs_index_compact,
            |h| vanedb_rs_index_compact(h) == 1
        ),
        probe!("index", vanedb_rs_index_m, |h| vanedb_rs_index_m(h) == 0),
        probe!(
            "index",
            vanedb_rs_index_ef_construction,
            |h| vanedb_rs_index_ef_construction(h) == 0
        ),
        probe!("index", vanedb_rs_index_seed, |h| vanedb_rs_index_seed(h)
            == 0),
        probe!(
            "index",
            vanedb_rs_index_capacity,
            |h| vanedb_rs_index_capacity(h) == 0
        ),
        probe!(
            "index",
            vanedb_rs_index_set_ef_search,
            |h| vanedb_rs_index_set_ef_search(h, 9) == 1
        ),
        probe!(
            "index",
            vanedb_rs_index_ef_search,
            |h| vanedb_rs_index_ef_search(h) == 0
        ),
        // -- disk -----------------------------------------------------------
        probe!("disk", vanedb_rs_disk_search, |h| {
            let (mut ids, mut ds) = ([0u64; 1], [0f32; 1]);
            vanedb_rs_disk_search(h, V2.as_ptr(), 1, ids.as_mut_ptr(), ds.as_mut_ptr()) == 0
        }),
        probe!("disk", vanedb_rs_disk_search_filtered, |h| {
            let (mut ids, mut ds) = ([0u64; 1], [0f32; 1]);
            vanedb_rs_disk_search_filtered(
                h,
                V2.as_ptr(),
                1,
                Some(accept_all),
                null_mut(),
                null(),
                0,
                null(),
                0,
                ids.as_mut_ptr(),
                ds.as_mut_ptr(),
            ) == 0
        }),
        probe!("disk", vanedb_rs_disk_free, |h| {
            settle();
            vanedb_rs_disk_free(h);
            vanedb_rs_last_error() == VANEDB_RS_INVALID_HANDLE
        }),
        probe!("disk", vanedb_rs_disk_metric, |h| vanedb_rs_disk_metric(h)
            == 0),
        probe!("disk", vanedb_rs_disk_len, |h| vanedb_rs_disk_len(h) == 0),
        probe!(
            "disk",
            vanedb_rs_disk_dimension,
            |h| vanedb_rs_disk_dimension(h) == 0
        ),
        probe!("disk", vanedb_rs_disk_contains, |h| {
            !vanedb_rs_disk_contains(h, 1)
        }),
        probe!("disk", vanedb_rs_disk_get, |h| {
            let mut out = [0f32; 2];
            vanedb_rs_disk_get(h, 1, out.as_mut_ptr()) == 1
        }),
        probe!("disk", vanedb_rs_disk_get_vector, |h| {
            let mut out = [0f32; 2];
            vanedb_rs_disk_get_vector(h, 1, out.as_mut_ptr()) == 1
        }),
    ]
}

/// One live handle of each kind, so every probe has a real id to truncate
/// and a wrong-kind id to be handed.
struct Live {
    store: u64,
    index: u64,
    disk: u64,
    path: std::path::PathBuf,
}

impl Live {
    fn new(tag: &str) -> Self {
        let path =
            std::env::temp_dir().join(format!("vanedb-handles-{tag}-{}.disk", std::process::id()));
        let c_path = std::ffi::CString::new(path.to_str().unwrap()).unwrap();
        unsafe {
            assert_eq!(
                vanedb_rs_disk_build(c_path.as_ptr(), 2, 0, IDS.as_ptr(), V2.as_ptr(), 1),
                0
            );
        }
        let store = vanedb_rs_store_new(2, 0);
        let index = vanedb_rs_index_new(2, 0, 8, 4, 16, 1);
        let disk = unsafe { vanedb_rs_disk_open(c_path.as_ptr()) };
        assert!(store != NULL && index != NULL && disk != NULL);
        Self {
            store,
            index,
            disk,
            path,
        }
    }

    fn of(&self, kind: &str) -> u64 {
        match kind {
            "store" => self.store,
            "index" => self.index,
            "disk" => self.disk,
            _ => unreachable!(),
        }
    }
}

impl Drop for Live {
    fn drop(&mut self) {
        vanedb_rs_store_free(self.store);
        vanedb_rs_index_free(self.index);
        vanedb_rs_disk_free(self.disk);
        let _ = std::fs::remove_file(&self.path);
    }
}

/// A freed handle of each kind: the stale ids.
fn freed(tag: &str) -> Live {
    let live = Live::new(tag);
    let stale = Live {
        store: live.store,
        index: live.index,
        disk: live.disk,
        path: live.path.clone(),
    };
    drop(live);
    // `stale`'s drop will free again, which is the double free the test
    // wants to see rejected; it must not crash.
    stale
}

/// A fixed sequence of random ids, so a failure reproduces.
fn random_ids() -> Vec<u64> {
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut ids = vec![
        1,
        2,
        0xFFFF_FFFF,
        1 << 32,
        u64::MAX,
        u64::MAX - 1,
        0x8000_0000_0000_0000,
    ];
    for _ in 0..64 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ids.push(state);
        // Small slot numbers with a wrong generation are the likeliest
        // collision shape; cover them too.
        ids.push((state << 32) | (state >> 60));
    }
    ids.retain(|&id| id != NULL);
    ids
}

fn assert_rejected(probe: &Probe, id: u64, why: &str) {
    assert!(
        (probe.rejected)(id),
        "{} must return its failure value for {why} id {id:#x}",
        probe.name
    );
    assert_eq!(
        vanedb_rs_last_error(),
        VANEDB_RS_INVALID_HANDLE,
        "{} must report INVALID_HANDLE for {why} id {id:#x}",
        probe.name
    );
}

#[test]
fn every_handle_taking_entry_point_has_a_probe() {
    let in_header: BTreeSet<String> = common::declarations()
        .iter()
        .filter(|d| common::handle_kind(d).is_some())
        .map(|d| d.name.clone())
        .collect();
    let probed: BTreeSet<String> = probes().iter().map(|p| p.name.to_string()).collect();
    assert_eq!(
        in_header, probed,
        "every function that takes a handle needs a probe here, and every probe a function"
    );
    for declaration in common::declarations() {
        if let Some(kind) = common::handle_kind(&declaration) {
            let probe = probes()
                .into_iter()
                .find(|p| p.name == declaration.name)
                .unwrap();
            assert_eq!(probe.kind, kind, "{} takes a {kind} handle", probe.name);
        }
    }
}

#[test]
fn truncated_ids_are_rejected_by_every_entry_point() {
    let live = Live::new("truncated");
    for probe in probes() {
        let id = live.of(probe.kind);
        // The C acceptance test found the first handle of a process truncating
        // to exactly 0, the null handle, which is a different code. No live
        // id may have a zero low half.
        assert_ne!(
            id & 0xFFFF_FFFF,
            0,
            "{}: low half must be nonzero",
            probe.name
        );
        // What a C `int` return leaves: the low 32 bits, zero-extended...
        assert_rejected(&probe, id & 0xFFFF_FFFF, "zero-extended truncated");
        // ...or sign-extended when bit 31 happens to be set.
        assert_rejected(
            &probe,
            id | 0xFFFF_FFFF_0000_0000,
            "sign-extended truncated",
        );
        // The high half alone, as if the low half were lost instead.
        assert_rejected(&probe, id & 0xFFFF_FFFF_0000_0000, "high-half-only");
    }
}

#[test]
fn freed_ids_are_rejected_by_every_entry_point_including_free() {
    let stale = freed("freed");
    for probe in probes() {
        assert_rejected(&probe, stale.of(probe.kind), "freed");
    }
    // And a second free, explicitly: the double free is INVALID_HANDLE.
    settle();
    vanedb_rs_store_free(stale.store);
    assert_eq!(vanedb_rs_last_error(), VANEDB_RS_INVALID_HANDLE);
}

#[test]
fn random_ids_are_rejected_by_every_entry_point() {
    let live = Live::new("random");
    let taken = [live.store, live.index, live.disk];
    for id in random_ids() {
        if taken.contains(&id) {
            continue;
        }
        for probe in probes() {
            assert_rejected(&probe, id, "random");
        }
    }
}

#[test]
fn a_handle_of_another_kind_is_rejected_not_reinterpreted() {
    let live = Live::new("kind");
    for probe in probes() {
        for other in ["store", "index", "disk"] {
            if other != probe.kind {
                assert_rejected(&probe, live.of(other), &format!("{other}-kind"));
            }
        }
    }
    // Every handle survived the wrong-kind frees above.
    assert_eq!(vanedb_rs_store_len(live.store), 0);
    assert_eq!(vanedb_rs_last_error(), VANEDB_RS_OK);
    assert_eq!(vanedb_rs_index_len(live.index), 0);
    assert_eq!(vanedb_rs_disk_len(live.disk), 1);
}

/// A live slot under the wrong generation is the shape a stale id takes
/// after its slot is reused, and the one shape the random ids above never
/// reach (their low halves name no slot). Flip a live handle's generation
/// half by one either way: every entry point must reject it and the live
/// handle must be untouched.
#[test]
fn a_live_slot_under_the_wrong_generation_is_rejected() {
    let live = Live::new("generation");
    for probe in probes() {
        let id = live.of(probe.kind);
        let generation = id >> 32;
        let low = id & 0xFFFF_FFFF;
        for wrong in [generation + 1, generation.wrapping_sub(1)] {
            assert_rejected(&probe, (wrong << 32) | low, "wrong-generation");
        }
    }
    assert_eq!(vanedb_rs_store_len(live.store), 0);
    assert_eq!(vanedb_rs_last_error(), VANEDB_RS_OK);
    assert_eq!(vanedb_rs_index_len(live.index), 0);
    assert_eq!(vanedb_rs_disk_len(live.disk), 1);
}

/// 0 keeps its old meaning: the ABI's null, `NULL_ARGUMENT` on use and a
/// no-op on free, so a C caller's `if (!h) ...; free(h)` idiom still holds.
#[test]
fn the_null_handle_is_a_null_argument_and_freeing_it_is_a_no_op() {
    for probe in probes() {
        if probe.name.ends_with("_free") {
            continue;
        }
        assert!(
            (probe.rejected)(NULL),
            "{} must reject the null handle",
            probe.name
        );
        assert_eq!(
            vanedb_rs_last_error(),
            VANEDB_RS_NULL_ARGUMENT,
            "{}: the null handle is NULL_ARGUMENT, not INVALID_HANDLE",
            probe.name
        );
    }
    // A free of the null handle preserves whatever error came before it.
    assert_eq!(vanedb_rs_store_new(0, 0), NULL);
    assert_eq!(vanedb_rs_last_error(), VANEDB_RS_ZERO_DIMENSION);
    vanedb_rs_store_free(NULL);
    vanedb_rs_index_free(NULL);
    vanedb_rs_disk_free(NULL);
    assert_eq!(vanedb_rs_last_error(), VANEDB_RS_ZERO_DIMENSION);
}

/// A freed slot is reused, but under a new generation: the old id stays dead.
#[test]
fn a_freed_id_is_never_reissued() {
    let mut seen = BTreeSet::new();
    for _ in 0..2_000 {
        let h = vanedb_rs_store_new(1, 0);
        assert_ne!(h, NULL);
        assert!(seen.insert(h), "id {h:#x} was issued twice");
        vanedb_rs_store_free(h);
        assert_eq!(vanedb_rs_store_len(h), 0);
        assert_eq!(vanedb_rs_last_error(), VANEDB_RS_INVALID_HANDLE);
    }
}

/// A live handle is unaffected by every bad id thrown at its entry points.
#[test]
fn live_handles_keep_working_around_rejected_ids() {
    let live = Live::new("live");
    unsafe {
        assert_eq!(vanedb_rs_store_add(live.store, 5, V2.as_ptr()), 0);
        assert_eq!(vanedb_rs_index_add(live.index, 5, V2.as_ptr()), 0);
    }
    for id in random_ids().into_iter().take(16) {
        assert_eq!(vanedb_rs_store_len(id), 0);
        assert_eq!(vanedb_rs_last_error(), VANEDB_RS_INVALID_HANDLE);
    }
    assert_eq!(vanedb_rs_store_len(live.store), 1);
    assert!(vanedb_rs_index_contains(live.index, 5));
    assert!(vanedb_rs_disk_contains(live.disk, 1));
}

/// Freeing a handle while other threads are mid-search on it must not crash:
/// each call holds its own reference, and every later call is rejected.
#[test]
fn freeing_under_concurrent_use_is_rejected_afterwards_not_a_crash() {
    let h = vanedb_rs_index_new(2, 0, 64, 4, 16, 7);
    assert_ne!(h, NULL);
    for id in 0..64u64 {
        let v = [id as f32, 1.0];
        assert_eq!(unsafe { vanedb_rs_index_add(h, id, v.as_ptr()) }, 0);
    }
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let workers: Vec<_> = (0..4)
        .map(|_| {
            let stop = stop.clone();
            std::thread::spawn(move || {
                let mut ok = 0usize;
                let mut rejected = 0usize;
                while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                    let (mut ids, mut ds) = ([0u64; 4], [0f32; 4]);
                    let n = unsafe {
                        vanedb_rs_index_search(
                            h,
                            V2.as_ptr(),
                            4,
                            0,
                            ids.as_mut_ptr(),
                            ds.as_mut_ptr(),
                        )
                    };
                    match vanedb_rs_last_error() {
                        VANEDB_RS_OK => {
                            assert_eq!(n, 4);
                            ok += 1;
                        }
                        VANEDB_RS_INVALID_HANDLE => {
                            assert_eq!(n, 0);
                            rejected += 1;
                        }
                        other => panic!("unexpected code {other}"),
                    }
                }
                (ok, rejected)
            })
        })
        .collect();
    std::thread::sleep(std::time::Duration::from_millis(20));
    vanedb_rs_index_free(h);
    std::thread::sleep(std::time::Duration::from_millis(20));
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    let mut total_rejected = 0;
    for worker in workers {
        let (_, rejected) = worker.join().expect("a worker must not crash");
        total_rejected += rejected;
    }
    assert!(
        total_rejected > 0,
        "searches after the free must be rejected"
    );
    assert_eq!(vanedb_rs_index_len(h), 0);
    assert_eq!(vanedb_rs_last_error(), VANEDB_RS_INVALID_HANDLE);
}
