// Behavior tests for the vanedb_rs_* C ABI. Functions are unsafe (raw pointers).

/// A unique path under the system temp directory.
///
/// These files were bare relative names, so they landed in whatever working
/// directory the test binary inherited and collided between concurrently
/// running binaries. The pid keeps two runs apart; the name keeps two tests
/// in one run apart.
fn scratch_path(name: &str) -> String {
    std::env::temp_dir()
        .join(format!("vanedb-{name}-{}.bin", std::process::id()))
        .to_string_lossy()
        .into_owned()
}

#[test]
fn search_beam_width_is_per_call() {
    unsafe {
        let h = std::ptr::NonNull::new(vanedb_capi::vanedb_rs_index_new(1, 0, 10, 2, 10, 42))
            .expect("index construction failed");
        // The constructor transfers a Box allocation. Reclaim ownership so
        // assertions use safe references and a panic still frees the index.
        let mut index = Box::from_raw(h.as_ptr());
        index.set_ef_search(73);
        let vector = [1.0];
        assert_eq!(
            vanedb_capi::vanedb_rs_index_add(&mut *index, 1, vector.as_ptr()),
            0
        );
        let mut ids = [0];
        let mut distances = [0.0];
        for ef in [1, 100] {
            assert_eq!(
                vanedb_capi::vanedb_rs_index_search(
                    &mut *index,
                    vector.as_ptr(),
                    1,
                    ef,
                    ids.as_mut_ptr(),
                    distances.as_mut_ptr(),
                ),
                1
            );
            assert_eq!(ids, [1]);
            assert_eq!(
                index.get_ef_search(),
                73,
                "a query must not change another query's beam width"
            );
        }
    }
}

#[test]
fn null_path_guards() {
    unsafe {
        assert_eq!(
            vanedb_capi::vanedb_rs_index_save(std::ptr::null_mut(), std::ptr::null()),
            1
        );
        assert!(vanedb_capi::vanedb_rs_index_load(std::ptr::null()).is_null());
        assert_eq!(
            vanedb_capi::vanedb_rs_disk_build(
                std::ptr::null(),
                2,
                0,
                std::ptr::null(),
                std::ptr::null(),
                0
            ),
            1
        );
        assert!(vanedb_capi::vanedb_rs_disk_open(std::ptr::null()).is_null());
    }
}

#[test]
fn hnsw() {
    let v0 = [0.0f32, 0.0];
    let v1 = [1.0f32, 1.0];
    let q = [0.1f32, 0.1];
    let path = std::ffi::CString::new(scratch_path("rs_capi_hnsw")).unwrap();
    unsafe {
        let h = vanedb_capi::vanedb_rs_index_new(2, 0, 100, 16, 200, 42);
        assert!(!h.is_null());
        assert_eq!(vanedb_capi::vanedb_rs_index_add(h, 10, v0.as_ptr()), 0);
        assert_eq!(vanedb_capi::vanedb_rs_index_add(h, 20, v1.as_ptr()), 0);
        let mut ids = [0u64; 2];
        let mut ds = [0.0f32; 2];
        let n = vanedb_capi::vanedb_rs_index_search(
            h,
            q.as_ptr(),
            2,
            50,
            ids.as_mut_ptr(),
            ds.as_mut_ptr(),
        );
        assert_eq!(n, 2);
        assert_eq!(ids[0], 10);
        assert_eq!(vanedb_capi::vanedb_rs_index_save(h, path.as_ptr()), 0);
        vanedb_capi::vanedb_rs_index_free(h);

        let h2 = vanedb_capi::vanedb_rs_index_load(path.as_ptr());
        assert!(!h2.is_null());
        let mut ids2 = [0u64; 1];
        let mut ds2 = [0.0f32; 1];
        let n2 = vanedb_capi::vanedb_rs_index_search(
            h2,
            q.as_ptr(),
            1,
            50,
            ids2.as_mut_ptr(),
            ds2.as_mut_ptr(),
        );
        assert_eq!(n2, 1);
        assert_eq!(ids2[0], 10);
        vanedb_capi::vanedb_rs_index_free(h2);
        // negative paths
        assert!(vanedb_capi::vanedb_rs_index_new(0, 0, 100, 16, 200, 42).is_null());
        assert_eq!(
            vanedb_capi::vanedb_rs_index_add(std::ptr::null_mut(), 1, v0.as_ptr()),
            1
        );
        assert_eq!(
            vanedb_capi::vanedb_rs_index_search(
                std::ptr::null_mut(),
                q.as_ptr(),
                1,
                50,
                ids2.as_mut_ptr(),
                ds2.as_mut_ptr()
            ),
            0
        );
        assert_eq!(
            vanedb_capi::vanedb_rs_index_save(std::ptr::null_mut(), path.as_ptr()),
            1
        );
    }
    let _ = std::fs::remove_file(scratch_path("rs_capi_hnsw"));
}

#[test]
fn mmap() {
    let ids_in = [10u64, 20];
    let vecs = [0.0f32, 0.0, 1.0, 1.0]; // row-major: id10=(0,0), id20=(1,1)
    let q = [0.1f32, 0.1];
    let path = std::ffi::CString::new(scratch_path("rs_capi_mmap")).unwrap();
    unsafe {
        assert_eq!(
            vanedb_capi::vanedb_rs_disk_build(
                path.as_ptr(),
                2,
                0,
                ids_in.as_ptr(),
                vecs.as_ptr(),
                2
            ),
            0
        );
        let m = vanedb_capi::vanedb_rs_disk_open(path.as_ptr());
        assert!(!m.is_null());
        let mut ids = [0u64; 2];
        let mut ds = [0.0f32; 2];
        let n =
            vanedb_capi::vanedb_rs_disk_search(m, q.as_ptr(), 2, ids.as_mut_ptr(), ds.as_mut_ptr());
        assert_eq!(n, 2);
        assert_eq!(ids[0], 10);
        vanedb_capi::vanedb_rs_disk_free(m);
        // negative path
        assert_eq!(
            vanedb_capi::vanedb_rs_disk_search(
                std::ptr::null_mut(),
                q.as_ptr(),
                2,
                ids.as_mut_ptr(),
                ds.as_mut_ptr()
            ),
            0
        );
    }
    let _ = std::fs::remove_file(scratch_path("rs_capi_mmap"));
}

/// n == 0 with null ids/vecs must build a valid empty store, matching the
/// null-safe-when-empty contract of the add_batch entry points.
#[test]
fn mmap_build_empty_with_null_pointers() {
    let path = std::ffi::CString::new(scratch_path("rs_capi_mmap_empty")).unwrap();
    unsafe {
        assert_eq!(
            vanedb_capi::vanedb_rs_disk_build(
                path.as_ptr(),
                2,
                0,
                std::ptr::null(),
                std::ptr::null(),
                0
            ),
            0
        );
        let m = vanedb_capi::vanedb_rs_disk_open(path.as_ptr());
        assert!(!m.is_null());
        let q = [0.0f32, 0.0];
        let mut ids = [0u64; 2];
        let mut ds = [0.0f32; 2];
        let n =
            vanedb_capi::vanedb_rs_disk_search(m, q.as_ptr(), 2, ids.as_mut_ptr(), ds.as_mut_ptr());
        assert_eq!(n, 0);
        vanedb_capi::vanedb_rs_disk_free(m);
    }
    let _ = std::fs::remove_file(scratch_path("rs_capi_mmap_empty"));
}

#[test]
fn distance() {
    let a = [1.0f32, 2.0, 3.0, 4.0];
    let b = [1.0f32, 2.0, 3.0, 5.0];
    unsafe {
        let l2 = vanedb_capi::vanedb_rs_l2_sq(a.as_ptr(), b.as_ptr(), 4);
        assert!((l2 - 1.0).abs() < 0.01); // (4-5)^2
        let dot = vanedb_capi::vanedb_rs_dot_product(a.as_ptr(), b.as_ptr(), 4);
        assert!((dot - 34.0).abs() < 0.01); // raw inner product, matches vanedb_cpp_dot_product
        let cos = vanedb_capi::vanedb_rs_cosine_distance(a.as_ptr(), a.as_ptr(), 4);
        assert!(cos.abs() < 0.01); // identical => ~0
    }
}

#[test]
fn store() {
    let v0 = [0.0f32, 0.0];
    let v1 = [1.0f32, 1.0];
    let q = [0.1f32, 0.1];
    unsafe {
        let s = vanedb_capi::vanedb_rs_store_new(2, 0); // L2
        assert!(!s.is_null());
        assert_eq!(vanedb_capi::vanedb_rs_store_add(s, 10, v0.as_ptr()), 0);
        assert_eq!(vanedb_capi::vanedb_rs_store_add(s, 20, v1.as_ptr()), 0);
        let mut ids = [0u64; 2];
        let mut ds = [0.0f32; 2];
        let n = vanedb_capi::vanedb_rs_store_search(
            s,
            q.as_ptr(),
            2,
            ids.as_mut_ptr(),
            ds.as_mut_ptr(),
        );
        assert_eq!(n, 2);
        assert_eq!(ids[0], 10); // (0,0) nearest to (0.1,0.1)
        assert!(ds[0] <= ds[1]);
        vanedb_capi::vanedb_rs_store_free(s);
        // negative paths (parity with C++ null guards)
        assert!(vanedb_capi::vanedb_rs_store_new(0, 0).is_null()); // dim=0 => Err => null
        assert_eq!(
            vanedb_capi::vanedb_rs_store_add(std::ptr::null_mut(), 1, v0.as_ptr()),
            1
        );
        assert_eq!(
            vanedb_capi::vanedb_rs_store_search(
                std::ptr::null_mut(),
                q.as_ptr(),
                2,
                ids.as_mut_ptr(),
                ds.as_mut_ptr()
            ),
            0
        );
    }
}

#[test]
fn non_finite_vectors_and_queries_are_rejected() {
    unsafe {
        let mut out_id = 0u64;
        let mut out_distance = 0.0f32;
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let invalid = [value, 0.0];
            let finite = [0.0, 0.0];

            let store = vanedb_capi::vanedb_rs_store_new(2, 0);
            assert_eq!(
                vanedb_capi::vanedb_rs_store_add(store, 1, invalid.as_ptr()),
                1
            );
            assert_eq!(
                vanedb_capi::vanedb_rs_store_add(store, 2, finite.as_ptr()),
                0
            );
            assert_eq!(
                vanedb_capi::vanedb_rs_store_search(
                    store,
                    invalid.as_ptr(),
                    1,
                    &mut out_id,
                    &mut out_distance,
                ),
                0
            );
            vanedb_capi::vanedb_rs_store_free(store);

            let index = vanedb_capi::vanedb_rs_index_new(2, 0, 4, 2, 10, 42);
            assert_eq!(
                vanedb_capi::vanedb_rs_index_add(index, 1, invalid.as_ptr()),
                1
            );
            assert_eq!(
                vanedb_capi::vanedb_rs_index_add(index, 2, finite.as_ptr()),
                0
            );
            assert_eq!(
                vanedb_capi::vanedb_rs_index_search(
                    index,
                    invalid.as_ptr(),
                    1,
                    10,
                    &mut out_id,
                    &mut out_distance,
                ),
                0
            );
            vanedb_capi::vanedb_rs_index_free(index);
        }
    }
}

#[test]
fn store_add_batch() {
    let ids = [1u64, 2, 3];
    let flat = [0.0f32, 0.0, 1.0, 1.0, 5.0, 5.0];
    unsafe {
        let s = vanedb_capi::vanedb_rs_store_new(2, 0);
        assert_eq!(
            vanedb_capi::vanedb_rs_store_add_batch(s, ids.as_ptr(), flat.as_ptr(), 3),
            0
        );
        let q = [0.9f32, 0.9];
        let mut out_ids = [0u64; 1];
        let mut out_ds = [0.0f32; 1];
        let n = vanedb_capi::vanedb_rs_store_search(
            s,
            q.as_ptr(),
            1,
            out_ids.as_mut_ptr(),
            out_ds.as_mut_ptr(),
        );
        assert_eq!(n, 1);
        assert_eq!(out_ids[0], 2);

        // duplicate id -> error, all-or-nothing (store unchanged: id 4 absent)
        let dup_ids = [4u64, 1];
        assert_eq!(
            vanedb_capi::vanedb_rs_store_add_batch(s, dup_ids.as_ptr(), flat.as_ptr(), 2),
            1
        );

        // empty batch is a no-op success, even with null pointers
        assert_eq!(
            vanedb_capi::vanedb_rs_store_add_batch(s, std::ptr::null(), std::ptr::null(), 0),
            0
        );

        // null handle guard
        assert_eq!(
            vanedb_capi::vanedb_rs_store_add_batch(
                std::ptr::null_mut(),
                ids.as_ptr(),
                flat.as_ptr(),
                3
            ),
            1
        );
        vanedb_capi::vanedb_rs_store_free(s);
    }
}

#[test]
fn hnsw_add_batch() {
    let ids = [10u64, 20];
    let flat = [0.0f32, 0.0, 1.0, 1.0];
    unsafe {
        let h = vanedb_capi::vanedb_rs_index_new(2, 0, 100, 16, 200, 42);
        assert_eq!(
            vanedb_capi::vanedb_rs_index_add_batch(h, ids.as_ptr(), flat.as_ptr(), 2),
            0
        );
        let q = [0.1f32, 0.1];
        let mut out_ids = [0u64; 2];
        let mut out_ds = [0.0f32; 2];
        let n = vanedb_capi::vanedb_rs_index_search(
            h,
            q.as_ptr(),
            2,
            50,
            out_ids.as_mut_ptr(),
            out_ds.as_mut_ptr(),
        );
        assert_eq!(n, 2);
        assert_eq!(out_ids[0], 10);

        // duplicate -> error
        assert_eq!(
            vanedb_capi::vanedb_rs_index_add_batch(h, ids.as_ptr(), flat.as_ptr(), 2),
            1
        );
        // null handle guard
        assert_eq!(
            vanedb_capi::vanedb_rs_index_add_batch(
                std::ptr::null_mut(),
                ids.as_ptr(),
                flat.as_ptr(),
                2
            ),
            1
        );
        vanedb_capi::vanedb_rs_index_free(h);
    }
}

#[test]
fn an_unknown_metric_is_rejected_rather_than_treated_as_l2() {
    // Mapping an unrecognised value to L2 would defeat `Metric`'s
    // #[non_exhaustive] across the boundary: a caller built against a newer
    // header would get silently wrong distances instead of a refusal.
    unsafe {
        for known in [0u32, 1, 2] {
            let h = vanedb_capi::vanedb_rs_store_new(4, known);
            assert!(!h.is_null(), "metric {known} must be accepted");
            vanedb_capi::vanedb_rs_store_free(h);
        }
        for unknown in [3u32, 99, u32::MAX] {
            assert!(
                vanedb_capi::vanedb_rs_store_new(4, unknown).is_null(),
                "metric {unknown} must be rejected"
            );
            assert!(
                vanedb_capi::vanedb_rs_index_new(4, unknown, 16, 4, 40, 7).is_null(),
                "metric {unknown} must be rejected"
            );
            let path = std::env::temp_dir().join(format!(
                "vanedb-invalid-metric-{}-{unknown}.vndb",
                std::process::id()
            ));
            let c_path = std::ffi::CString::new(path.to_str().unwrap()).unwrap();
            assert_eq!(
                vanedb_capi::vanedb_rs_disk_build(
                    c_path.as_ptr(),
                    4,
                    unknown,
                    std::ptr::null(),
                    std::ptr::null(),
                    0,
                ),
                1
            );
            assert!(!path.exists(), "invalid input must not write a file");
        }
    }
}

#[test]
fn every_metric_round_trips_and_is_reportable() {
    // Dot had no coverage through this ABI at all, and no handle could report
    // the metric it was built with — so a caller opening a file someone else
    // wrote had no way to confirm their query convention matched.
    unsafe {
        for metric in [0u32, 1, 2] {
            let s = vanedb_capi::vanedb_rs_store_new(2, metric);
            assert!(!s.is_null());
            assert_eq!(vanedb_capi::vanedb_rs_store_metric(s), metric);
            vanedb_capi::vanedb_rs_store_free(s);

            let h = vanedb_capi::vanedb_rs_index_new(2, metric, 16, 4, 40, 7);
            assert!(!h.is_null());
            assert_eq!(vanedb_capi::vanedb_rs_index_metric(h), metric);
            vanedb_capi::vanedb_rs_index_free(h);
        }
        // Null handles report 0, which is L2's value: documented, and the
        // reason a caller must check the handle first.
        assert_eq!(vanedb_capi::vanedb_rs_store_metric(std::ptr::null()), 0);
    }
}

#[test]
fn dot_ranks_by_largest_inner_product_through_the_abi() {
    unsafe {
        let s = vanedb_capi::vanedb_rs_store_new(2, 2); // dot
        assert!(!s.is_null());
        for (id, v) in [(1u64, [1.0f32, 0.0]), (2, [4.0, 0.0]), (3, [0.0, 1.0])] {
            assert_eq!(vanedb_capi::vanedb_rs_store_add(s, id, v.as_ptr()), 0);
        }
        let q = [1.0f32, 0.0];
        let mut ids = [0u64; 3];
        let mut ds = [0f32; 3];
        let n = vanedb_capi::vanedb_rs_store_search(
            s,
            q.as_ptr(),
            3,
            ids.as_mut_ptr(),
            ds.as_mut_ptr(),
        );
        assert_eq!(n, 3);
        assert_eq!(ids[0], 2, "dot must rank the largest inner product first");
        assert_eq!(ids[2], 3);
        vanedb_capi::vanedb_rs_store_free(s);
    }
}

/// Every entry point must reject a null handle rather than dereference it.
///
/// Coverage on the C ABI was 88% of lines, and almost all of the remainder was
/// these rejection branches — the ones that stand between a C caller's mistake
/// and undefined behaviour in Rust. `vanedb_capi::vanedb_rs_store_add(s, id, NULL)` going
/// straight into `from_raw_parts` was a real defect once (`54156f0`); it is
/// exactly this shape.
///
/// A null handle returns the failure code for its return type: 1 for status
/// codes, 0 for counts, and for the metric accessors 0, which is also L2's
/// value — so the header tells callers to check the handle first.
#[test]
fn every_entry_point_rejects_a_null_handle() {
    use std::ptr;
    let v = [1.0f32, 0.0];
    let mut ids = [0u64; 4];
    let mut ds = [0f32; 4];

    unsafe {
        // Status-code returning calls: 1 on rejection.
        assert_eq!(
            vanedb_capi::vanedb_rs_store_add(ptr::null_mut(), 1, v.as_ptr()),
            1
        );
        assert_eq!(
            vanedb_capi::vanedb_rs_store_add_batch(ptr::null_mut(), ids.as_ptr(), v.as_ptr(), 1),
            1
        );
        assert_eq!(vanedb_capi::vanedb_rs_store_remove(ptr::null_mut(), 1), 1);
        assert_eq!(
            vanedb_capi::vanedb_rs_store_get(ptr::null_mut(), 1, ds.as_mut_ptr()),
            1
        );
        assert_eq!(
            vanedb_capi::vanedb_rs_index_add(ptr::null_mut(), 1, v.as_ptr()),
            1
        );
        assert_eq!(
            vanedb_capi::vanedb_rs_index_add_batch(ptr::null_mut(), ids.as_ptr(), v.as_ptr(), 1),
            1
        );
        assert_eq!(
            vanedb_capi::vanedb_rs_index_upsert(ptr::null_mut(), 1, v.as_ptr()),
            1
        );
        assert_eq!(vanedb_capi::vanedb_rs_index_remove(ptr::null_mut(), 1), 1);
        assert_eq!(
            vanedb_capi::vanedb_rs_index_get_vector(ptr::null_mut(), 1, ds.as_mut_ptr()),
            1
        );
        assert_eq!(vanedb_capi::vanedb_rs_index_compact(ptr::null_mut()), 1);
        assert_eq!(
            vanedb_capi::vanedb_rs_disk_get(ptr::null_mut(), 1, ds.as_mut_ptr()),
            1
        );

        // Count-returning calls: 0 on rejection.
        assert_eq!(vanedb_capi::vanedb_rs_store_len(ptr::null()), 0);
        assert_eq!(vanedb_capi::vanedb_rs_store_dimension(ptr::null()), 0);
        assert_eq!(vanedb_capi::vanedb_rs_index_len(ptr::null()), 0);
        assert_eq!(vanedb_capi::vanedb_rs_index_dimension(ptr::null()), 0);
        assert_eq!(vanedb_capi::vanedb_rs_index_tombstones(ptr::null()), 0);
        assert_eq!(vanedb_capi::vanedb_rs_disk_len(ptr::null()), 0);
        assert_eq!(vanedb_capi::vanedb_rs_disk_dimension(ptr::null()), 0);

        // Membership: false on rejection.
        assert!(!vanedb_capi::vanedb_rs_store_contains(ptr::null(), 1));
        assert!(!vanedb_capi::vanedb_rs_index_contains(ptr::null(), 1));
        assert!(!vanedb_capi::vanedb_rs_disk_contains(ptr::null(), 1));

        // Search: 0 results on rejection.
        assert_eq!(
            vanedb_capi::vanedb_rs_store_search(
                ptr::null_mut(),
                v.as_ptr(),
                1,
                ids.as_mut_ptr(),
                ds.as_mut_ptr()
            ),
            0
        );
        assert_eq!(
            vanedb_capi::vanedb_rs_index_search(
                ptr::null_mut(),
                v.as_ptr(),
                1,
                1,
                ids.as_mut_ptr(),
                ds.as_mut_ptr()
            ),
            0
        );
        assert_eq!(
            vanedb_capi::vanedb_rs_disk_search(
                ptr::null_mut(),
                v.as_ptr(),
                1,
                ids.as_mut_ptr(),
                ds.as_mut_ptr()
            ),
            0
        );

        // Freeing a null handle is a no-op, not a crash — C callers free in
        // cleanup paths that may not have allocated.
        vanedb_capi::vanedb_rs_store_free(ptr::null_mut());
        vanedb_capi::vanedb_rs_index_free(ptr::null_mut());
        vanedb_capi::vanedb_rs_disk_free(ptr::null_mut());
    }
}

/// A null data pointer with a non-zero count must be rejected before it
/// reaches `from_raw_parts`. An empty batch is legal and must stay legal, so
/// the guard is `n != 0 && ptr.is_null()`, not `ptr.is_null()`.
#[test]
fn null_data_with_a_nonzero_count_is_rejected_but_empty_batches_are_not() {
    use std::ptr;
    unsafe {
        let store = vanedb_capi::vanedb_rs_store_new(2, 0);
        assert!(!store.is_null());
        let index = vanedb_capi::vanedb_rs_index_new(2, 0, 16, 4, 16, 42);
        assert!(!index.is_null());

        assert_eq!(vanedb_capi::vanedb_rs_store_add(store, 1, ptr::null()), 1);
        assert_eq!(vanedb_capi::vanedb_rs_index_add(index, 1, ptr::null()), 1);
        assert_eq!(
            vanedb_capi::vanedb_rs_index_upsert(index, 1, ptr::null()),
            1
        );
        assert_eq!(
            vanedb_capi::vanedb_rs_store_add_batch(store, ptr::null(), ptr::null(), 1),
            1
        );
        assert_eq!(
            vanedb_capi::vanedb_rs_index_add_batch(index, ptr::null(), ptr::null(), 1),
            1
        );

        // n == 0 never dereferences, so null is fine and the call succeeds.
        assert_eq!(
            vanedb_capi::vanedb_rs_store_add_batch(store, ptr::null(), ptr::null(), 0),
            0
        );
        assert_eq!(
            vanedb_capi::vanedb_rs_index_add_batch(index, ptr::null(), ptr::null(), 0),
            0
        );
        assert_eq!(vanedb_capi::vanedb_rs_store_len(store), 0);
        assert_eq!(vanedb_capi::vanedb_rs_index_len(index), 0);

        // A count whose product with the dimension overflows must be refused
        // before any slice is formed.
        let ids = [0u64; 1];
        let vecs = [0f32; 2];
        assert_eq!(
            vanedb_capi::vanedb_rs_store_add_batch(store, ids.as_ptr(), vecs.as_ptr(), usize::MAX),
            1
        );
        assert_eq!(
            vanedb_capi::vanedb_rs_index_add_batch(index, ids.as_ptr(), vecs.as_ptr(), usize::MAX),
            1
        );

        vanedb_capi::vanedb_rs_store_free(store);
        vanedb_capi::vanedb_rs_index_free(index);
    }
}

// ---------------------------------------------------------------------------
// ef_search = 0, the error channel, and the accessors a loaded handle needs.

/// `ef_search` is a required argument with no documented default, so `0` is the
/// idiom a C caller reaches for to mean "use whatever the index is set to".
/// It used to flow into `SearchParams::ef_search(0)` and get clamped to `k` —
/// for k=10 a beam 5x narrower than the index's own default, returning
/// plausible results at silently degraded recall. The parallel C++ ABI treats
/// the same call as an error and returns no results. Both ABIs are called
/// through one uniform FFI (`bench/src/ffi.rs`), so they must agree.
#[test]
fn zero_ef_search_means_the_indexs_own_setting() {
    // The first version of this used 40 one-dimensional points, where a beam
    // of 40 and a beam clamped to k = 10 return the same ids, so it passed
    // with the change fully reverted. A sparse graph over 5000 random vectors
    // separates them: recall differs by a wide margin.
    const DIM: usize = 32;
    const N: u64 = 5000;
    let mut state = 0x243f_6a88_85a3_08d3u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / 8192.0 - 1.0
    };
    let rows: Vec<(u64, Vec<f32>)> = (0..N)
        .map(|id| (id, (0..DIM).map(|_| next()).collect()))
        .collect();

    unsafe {
        let handle = vanedb_capi::vanedb_rs_index_new(DIM, 0, N as usize, 8, 32, 7);
        assert!(!handle.is_null());
        let mut index = Box::from_raw(handle);
        for (id, vector) in &rows {
            assert_eq!(
                vanedb_capi::vanedb_rs_index_add(&mut *index, *id, vector.as_ptr()),
                0
            );
        }
        assert_eq!(
            vanedb_capi::vanedb_rs_index_set_ef_search(&*index, 400),
            0,
            "a C caller must be able to set what 0 resolves to"
        );

        // A fresh point, not one of the stored vectors: querying a stored
        // vector finds itself trivially and its neighbourhood with it, which
        // gives recall 1.0 at any beam and separates nothing.
        let query: Vec<f32> = (0..DIM).map(|_| next()).collect();
        let query = &query;
        // Ground truth from a brute force written here, not from the engine.
        let mut exact: Vec<(f32, u64)> = rows
            .iter()
            .map(|(id, v)| {
                let d: f32 = v.iter().zip(query).map(|(a, b)| (a - b) * (a - b)).sum();
                (d, *id)
            })
            .collect();
        exact.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)));
        let truth: std::collections::HashSet<u64> =
            exact.iter().take(10).map(|(_, id)| *id).collect();

        let mut recall_at = |ef: usize| {
            let mut ids = [0u64; 10];
            let mut distances = [0.0f32; 10];
            let n = vanedb_capi::vanedb_rs_index_search(
                &mut *index,
                query.as_ptr(),
                10,
                ef,
                ids.as_mut_ptr(),
                distances.as_mut_ptr(),
            );
            assert_eq!(n, 10, "ef = {ef} must still fill k");
            ids.iter().filter(|id| truth.contains(id)).count() as f32 / 10.0
        };

        let with_zero = recall_at(0); // resolves to the stored 400
        let with_ten = recall_at(10); // the beam a clamped-to-k call would get
        assert!(
            with_zero > with_ten + 0.15,
            "0 must search at the stored ef_search, not be clamped to k: \
             ef=0 gave {with_zero}, ef=10 gave {with_ten}"
        );
        assert_eq!(
            vanedb_capi::vanedb_rs_index_ef_search(&*index),
            400,
            "resolving 0 must not mutate the handle"
        );
    }
}

/// Every status function returned a bare `1` for a duplicate id, a dimension
/// mismatch, a corrupt file and an I/O failure alike, and searches returned `0`
/// results for both "empty" and "your query had a NaN in it". A caller could
/// not implement the branching the core error type is designed for — retry on
/// Io, abort on Corrupt, skip on DuplicateId.
#[test]
fn failures_are_reported_through_the_error_channel() {
    unsafe {
        let handle = vanedb_capi::vanedb_rs_store_new(3, 0);
        assert!(!handle.is_null());
        let mut store = Box::from_raw(handle);

        let vector = [1.0f32, 2.0, 3.0];
        assert_eq!(
            vanedb_capi::vanedb_rs_store_add(&mut *store, 1, vector.as_ptr()),
            0
        );
        assert_eq!(
            vanedb_capi::vanedb_rs_last_error(),
            vanedb_capi::VANEDB_RS_OK,
            "success must clear the previous error"
        );

        assert_eq!(
            vanedb_capi::vanedb_rs_store_add(&mut *store, 1, vector.as_ptr()),
            1
        );
        assert_eq!(
            vanedb_capi::vanedb_rs_last_error(),
            vanedb_capi::VANEDB_RS_DUPLICATE_ID
        );

        let nonfinite = [f32::NAN, 0.0, 0.0];
        let mut ids = [0u64; 4];
        let mut distances = [0.0f32; 4];
        let n = vanedb_capi::vanedb_rs_store_search(
            &mut *store,
            nonfinite.as_ptr(),
            4,
            ids.as_mut_ptr(),
            distances.as_mut_ptr(),
        );
        assert_eq!(n, 0);
        assert_eq!(
            vanedb_capi::vanedb_rs_last_error(),
            vanedb_capi::VANEDB_RS_NON_FINITE_VALUE,
            "a failed search must be distinguishable from an empty one"
        );

        // The same zero return, with nothing wrong: an empty store.
        let empty = vanedb_capi::vanedb_rs_store_new(3, 0);
        let mut empty = Box::from_raw(empty);
        let n = vanedb_capi::vanedb_rs_store_search(
            &mut *empty,
            vector.as_ptr(),
            4,
            ids.as_mut_ptr(),
            distances.as_mut_ptr(),
        );
        assert_eq!(n, 0);
        assert_eq!(
            vanedb_capi::vanedb_rs_last_error(),
            vanedb_capi::VANEDB_RS_OK,
            "an empty store is not a failure"
        );

        // A null handle is the ABI's own misuse code, not a core error.
        assert_eq!(
            vanedb_capi::vanedb_rs_store_add(std::ptr::null_mut(), 2, vector.as_ptr()),
            1
        );
        assert_eq!(
            vanedb_capi::vanedb_rs_last_error(),
            vanedb_capi::VANEDB_RS_NULL_ARGUMENT
        );
    }
}

/// A constructor returning null carries no code of its own, so the enum is the
/// only way to tell "bad metric" from "allocation refused".
#[test]
fn a_null_returning_constructor_records_why() {
    unsafe {
        assert!(vanedb_capi::vanedb_rs_store_new(3, 99).is_null());
        assert_eq!(
            vanedb_capi::vanedb_rs_last_error(),
            vanedb_capi::VANEDB_RS_INVALID_PARAMETER
        );

        let missing = std::ffi::CString::new("/nonexistent/vanedb/nope.vndb").unwrap();
        assert!(vanedb_capi::vanedb_rs_index_load(missing.as_ptr()).is_null());
        assert_eq!(
            vanedb_capi::vanedb_rs_last_error(),
            vanedb_capi::VANEDB_RS_FILE_NOT_FOUND,
            "absent and corrupt must be separable: one is worth retrying"
        );
    }
}

/// The enum cannot carry `Corrupt`'s detail string, which is the part that says
/// *what* was wrong with the file.
#[test]
fn the_error_message_carries_the_detail_the_code_cannot() {
    unsafe {
        let dir = std::env::temp_dir().join(format!("vanedb-capi-msg-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("garbage.vndb");
        std::fs::write(&path, b"not a vndb file at all").unwrap();
        let c_path = std::ffi::CString::new(path.to_str().unwrap()).unwrap();

        assert!(vanedb_capi::vanedb_rs_index_load(c_path.as_ptr()).is_null());
        assert_eq!(
            vanedb_capi::vanedb_rs_last_error(),
            vanedb_capi::VANEDB_RS_CORRUPT
        );
        let message = std::ffi::CStr::from_ptr(vanedb_capi::vanedb_rs_last_error_message())
            .to_string_lossy()
            .into_owned();
        assert!(
            message.contains("magic"),
            "message must name the failure, got {message:?}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}

/// A caller who loaded a file they did not write has no other way to learn the
/// graph geometry they are searching. The core exposes these for exactly that
/// reason; the C ABI is the surface where a header is vendored and a missing
/// accessor is hardest to add later.
#[test]
fn a_loaded_handle_reports_the_geometry_it_was_built_with() {
    unsafe {
        let handle = vanedb_capi::vanedb_rs_index_new(4, 1, 512, 6, 48, 1234);
        assert!(!handle.is_null());
        let mut index = Box::from_raw(handle);
        assert_eq!(vanedb_capi::vanedb_rs_index_m(&*index), 6);
        // The accessors exist for a handle the caller did NOT build, so the
        // test has to load one. It previously only ever built.
        {
            let dir = std::env::temp_dir().join(format!("vanedb-capi-geom-{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            let path = dir.join("graph.vndb");
            let c_path = std::ffi::CString::new(path.to_str().unwrap()).unwrap();
            let v = [1.0f32, 0.0, 0.0, 0.0];
            assert_eq!(
                vanedb_capi::vanedb_rs_index_add(&mut *index, 1, v.as_ptr()),
                0
            );
            assert_eq!(vanedb_capi::vanedb_rs_index_set_ef_search(&*index, 77), 0);
            assert_eq!(
                vanedb_capi::vanedb_rs_index_save(&mut *index, c_path.as_ptr()),
                0
            );
            let loaded = vanedb_capi::vanedb_rs_index_load(c_path.as_ptr());
            assert!(!loaded.is_null());
            let loaded = Box::from_raw(loaded);
            assert_eq!(vanedb_capi::vanedb_rs_index_m(&*loaded), 6);
            assert_eq!(vanedb_capi::vanedb_rs_index_ef_construction(&*loaded), 48);
            assert_eq!(vanedb_capi::vanedb_rs_index_seed(&*loaded), 1234);
            assert_eq!(
                vanedb_capi::vanedb_rs_index_ef_search(&*loaded),
                77,
                "the stored beam travels with the file"
            );
            let _ = std::fs::remove_dir_all(&dir);
        }
        assert_eq!(vanedb_capi::vanedb_rs_index_ef_construction(&*index), 48);
        assert_eq!(vanedb_capi::vanedb_rs_index_seed(&*index), 1234);
        assert_eq!(vanedb_capi::vanedb_rs_index_capacity(&*index), 512);

        // Null is the documented no-handle case for every accessor.
        assert_eq!(vanedb_capi::vanedb_rs_index_m(std::ptr::null()), 0);
        assert_eq!(
            vanedb_capi::vanedb_rs_index_ef_construction(std::ptr::null()),
            0
        );
        assert_eq!(vanedb_capi::vanedb_rs_index_seed(std::ptr::null()), 0);
        assert_eq!(vanedb_capi::vanedb_rs_index_capacity(std::ptr::null()), 0);
        assert_eq!(vanedb_capi::vanedb_rs_index_ef_search(std::ptr::null()), 0);
    }
}

/// `get` and `get_vector` are the same operation under two names so a program
/// is not tied to one index type (#85). The core carries both; this ABI offered
/// `_store_get` and `_index_get_vector` and nothing else, so swapping index
/// type meant renaming call sites.
#[test]
fn both_spellings_of_the_read_exist_on_every_handle() {
    unsafe {
        let handle = vanedb_capi::vanedb_rs_index_new(2, 0, 16, 4, 16, 1);
        assert!(!handle.is_null());
        let mut index = Box::from_raw(handle);
        let vector = [3.0f32, 4.0];
        assert_eq!(
            vanedb_capi::vanedb_rs_index_add(&mut *index, 9, vector.as_ptr()),
            0
        );

        let mut through_get = [0.0f32; 2];
        let mut through_get_vector = [0.0f32; 2];
        assert_eq!(
            vanedb_capi::vanedb_rs_index_get(&*index, 9, through_get.as_mut_ptr()),
            0
        );
        assert_eq!(
            vanedb_capi::vanedb_rs_index_get_vector(&*index, 9, through_get_vector.as_mut_ptr()),
            0
        );
        assert_eq!(through_get, vector);
        assert_eq!(through_get, through_get_vector);

        let store = vanedb_capi::vanedb_rs_store_new(2, 0);
        let mut store = Box::from_raw(store);
        assert_eq!(
            vanedb_capi::vanedb_rs_store_add(&mut *store, 9, vector.as_ptr()),
            0
        );
        let mut from_store = [0.0f32; 2];
        assert_eq!(
            vanedb_capi::vanedb_rs_store_get_vector(&*store, 9, from_store.as_mut_ptr()),
            0
        );
        assert_eq!(from_store, vector);
    }
}

/// A consumer cannot otherwise check that the shared object it loaded matches
/// the header it compiled against.
#[test]
fn the_library_reports_its_own_version() {
    unsafe {
        let version = std::ffi::CStr::from_ptr(vanedb_capi::vanedb_rs_version())
            .to_str()
            .unwrap();
        assert_eq!(version, env!("CARGO_PKG_VERSION"));
    }
}

/// A *failing* ABI call from a thread-local destructor must not abort.
///
/// `set_code` writes to `LAST_MESSAGE`, a thread-local with drop glue, so it
/// enters the "destroyed" state at thread exit and `LocalKey::with` panics
/// there with `AccessError`. That panic is raised outside `catch_unwind`, and
/// a panic crossing an `extern "C"` boundary aborts the process. The pattern
/// is ordinary in C++: a `thread_local` handle whose destructor calls back
/// into the library at thread exit.
///
/// Ordering matters and is why the handle is registered *before* the first
/// failing call: destructors run in reverse registration order, so
/// `LAST_MESSAGE` must be registered later to be destroyed first. `LAST_ERROR`
/// alone would not reproduce it — a `Cell<u32>` has no drop glue, so its TLS
/// is never destroyed and `with` never fails on it.
///
/// If this regresses the whole test binary aborts rather than reporting a
/// failure. That is the point: the failure mode is a dead host process.
#[test]
fn a_failing_abi_call_from_a_thread_local_destructor_does_not_abort() {
    struct CallsAtThreadExit(*mut vanedb_capi::vanedb_rs_store);
    impl Drop for CallsAtThreadExit {
        fn drop(&mut self) {
            // SAFETY: the handle came from `vanedb_rs_store_new` and is freed
            // exactly once, here.
            unsafe { vanedb_capi::vanedb_rs_store_free(self.0) };
            // A guarded call that FAILS, so it reaches `set_code` and the
            // message TLS rather than only the code cell.
            // SAFETY: a zero dimension is rejected; nothing is allocated.
            let rejected = unsafe { vanedb_capi::vanedb_rs_store_new(0, 0) };
            assert!(rejected.is_null());
            let _ = vanedb_capi::vanedb_rs_last_error();
            let _ = vanedb_capi::vanedb_rs_last_error_message();
        }
    }
    thread_local! {
        static HANDLE: std::cell::RefCell<Option<CallsAtThreadExit>> =
            const { std::cell::RefCell::new(None) };
    }

    std::thread::spawn(|| {
        // SAFETY: valid arguments.
        let handle = unsafe { vanedb_capi::vanedb_rs_store_new(2, 0) };
        assert!(!handle.is_null());
        // Registered first, so its destructor runs last.
        HANDLE.with(|slot| *slot.borrow_mut() = Some(CallsAtThreadExit(handle)));
        // Registers LAST_MESSAGE now, after HANDLE, so it is destroyed first.
        // SAFETY: a duplicate add fails and records a message.
        unsafe {
            let v = [1.0f32, 0.0];
            vanedb_capi::vanedb_rs_store_add(handle, 1, v.as_ptr());
            vanedb_capi::vanedb_rs_store_add(handle, 1, v.as_ptr());
        }
        assert_eq!(
            vanedb_capi::vanedb_rs_last_error(),
            vanedb_capi::VANEDB_RS_DUPLICATE_ID
        );
    })
    .join()
    .expect("the thread must exit cleanly, not abort");
}

/// Every error code reachable from C must be produced by something, and no
/// failure may report success.
///
/// A mutation sweep found six of the sixteen codes never reached by any test:
/// coverage showed the `code_for` arms for `Io`, `InvalidK`,
/// `DimensionMismatch`, `BatchLengthMismatch`, `ZeroDimension` and `Backend`
/// never executing. Mapping `Io` to `VANEDB_RS_OK` — an I/O failure reporting
/// *success* to a C caller — survived the whole suite, as did routing
/// `InvalidK` to `VANEDB_RS_IO`.
///
/// Two of those six are not reachable through this ABI at all: it derives a
/// batch's vector count as `n * dim` rather than accepting a length, so a
/// caller cannot present a mismatched batch or a wrong-width vector. Those
/// codes exist because `code_for` maps every `VaneError` variant; the header
/// records which a C caller can actually see.
#[test]
fn every_reachable_error_code_is_actually_produced() {
    unsafe {
        let handle = vanedb_capi::vanedb_rs_store_new(3, 0);
        assert!(!handle.is_null());
        let mut store = Box::from_raw(handle);
        let v = [1.0f32, 2.0, 3.0];
        assert_eq!(
            vanedb_capi::vanedb_rs_store_add(&mut *store, 1, v.as_ptr()),
            0
        );

        let mut ids = [0u64; 4];
        let mut distances = [0.0f32; 4];

        // InvalidK: k = 0. Previously produced by nothing.
        assert_eq!(
            vanedb_capi::vanedb_rs_store_search(
                &mut *store,
                v.as_ptr(),
                0,
                ids.as_mut_ptr(),
                distances.as_mut_ptr(),
            ),
            0
        );
        assert_eq!(
            vanedb_capi::vanedb_rs_last_error(),
            vanedb_capi::VANEDB_RS_INVALID_K,
            "k = 0 must report InvalidK, not a bare zero count"
        );

        // ZeroDimension: a store of dimension 0.
        assert!(vanedb_capi::vanedb_rs_store_new(0, 0).is_null());
        assert_eq!(
            vanedb_capi::vanedb_rs_last_error(),
            vanedb_capi::VANEDB_RS_ZERO_DIMENSION
        );

        // Io: a save whose parent directory does not exist. The mutation that
        // mapped this to VANEDB_RS_OK — a failed write reporting success —
        // survived everything before this assertion existed.
        let index = vanedb_capi::vanedb_rs_index_new(3, 0, 8, 4, 16, 1);
        assert!(!index.is_null());
        let mut index = Box::from_raw(index);
        assert_eq!(
            vanedb_capi::vanedb_rs_index_add(&mut *index, 1, v.as_ptr()),
            0
        );
        let nowhere = std::ffi::CString::new(
            std::env::temp_dir()
                .join(format!(
                    "vanedb-absent-{}/deeper/still/g.vndb",
                    std::process::id()
                ))
                .to_str()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(
            vanedb_capi::vanedb_rs_index_save(&mut *index, nowhere.as_ptr()),
            1
        );
        assert_eq!(
            vanedb_capi::vanedb_rs_last_error(),
            vanedb_capi::VANEDB_RS_FILE_NOT_FOUND,
            "a missing parent directory is FileNotFound"
        );

        // A genuine `Io`, which is a different arm: `from_io` routes NotFound
        // to FileNotFound and everything else to Io, so a missing directory
        // never reaches it. Saving *onto* an existing directory does.
        let dir = std::env::temp_dir().join(format!("vanedb-isdir-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let onto_dir = std::ffi::CString::new(dir.to_str().unwrap()).unwrap();
        assert_eq!(
            vanedb_capi::vanedb_rs_index_save(&mut *index, onto_dir.as_ptr()),
            1
        );
        let code = vanedb_capi::vanedb_rs_last_error();
        assert_ne!(
            code,
            vanedb_capi::VANEDB_RS_OK,
            "a failed save must never report success -- mapping Io to OK \
             survived the entire suite before this assertion existed"
        );
        assert_eq!(
            code,
            vanedb_capi::VANEDB_RS_IO,
            "writing onto a directory is an Io failure, got {code}"
        );
        let _ = std::fs::remove_dir_all(&dir);

        // NonFiniteValue through the batch path, with a correctly sized buffer:
        // this ABI computes the vector count as `n * dim`, so a short buffer is
        // undefined behaviour in the caller, not a reportable error.
        let batch_ids = [7u64, 8];
        let with_nan = [1.0f32, 2.0, 3.0, f32::NAN, 0.0, 0.0];
        assert_eq!(
            vanedb_capi::vanedb_rs_store_add_batch(
                &mut *store,
                batch_ids.as_ptr(),
                with_nan.as_ptr(),
                2,
            ),
            1
        );
        assert_eq!(
            vanedb_capi::vanedb_rs_last_error(),
            vanedb_capi::VANEDB_RS_NON_FINITE_VALUE
        );
        // All-or-nothing: the good vector in that batch must not have landed.
        assert!(!vanedb_capi::vanedb_rs_store_contains(&*store, 7));
    }
}
