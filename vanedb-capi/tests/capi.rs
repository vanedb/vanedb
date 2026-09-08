// Behavior tests for the vanedb_rs_* C ABI. Functions are unsafe (raw pointers).

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
    let path = std::ffi::CString::new("rs_capi_hnsw.bin").unwrap();
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
    let _ = std::fs::remove_file("rs_capi_hnsw.bin");
}

#[test]
fn mmap() {
    let ids_in = [10u64, 20];
    let vecs = [0.0f32, 0.0, 1.0, 1.0]; // row-major: id10=(0,0), id20=(1,1)
    let q = [0.1f32, 0.1];
    let path = std::ffi::CString::new("rs_capi_mmap.bin").unwrap();
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
    let _ = std::fs::remove_file("rs_capi_mmap.bin");
}

/// n == 0 with null ids/vecs must build a valid empty store, matching the
/// null-safe-when-empty contract of the add_batch entry points.
#[test]
fn mmap_build_empty_with_null_pointers() {
    let path = std::ffi::CString::new("rs_capi_mmap_empty.bin").unwrap();
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
    let _ = std::fs::remove_file("rs_capi_mmap_empty.bin");
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
