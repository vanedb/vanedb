// Behavior tests for the vanedb_rs_* C ABI. Functions are unsafe (raw pointers).

#[test]
fn hnsw() {
    let v0 = [0.0f32, 0.0];
    let v1 = [1.0f32, 1.0];
    let q = [0.1f32, 0.1];
    let path = std::ffi::CString::new("rs_capi_hnsw.bin").unwrap();
    unsafe {
        let h = vanedb_capi::vanedb_rs_hnsw_new(2, 0, 100, 16, 200, 42);
        assert!(!h.is_null());
        assert_eq!(vanedb_capi::vanedb_rs_hnsw_add(h, 10, v0.as_ptr()), 0);
        assert_eq!(vanedb_capi::vanedb_rs_hnsw_add(h, 20, v1.as_ptr()), 0);
        let mut ids = [0u64; 2];
        let mut ds = [0.0f32; 2];
        let n = vanedb_capi::vanedb_rs_hnsw_search(
            h,
            q.as_ptr(),
            2,
            50,
            ids.as_mut_ptr(),
            ds.as_mut_ptr(),
        );
        assert_eq!(n, 2);
        assert_eq!(ids[0], 10);
        assert_eq!(vanedb_capi::vanedb_rs_hnsw_save(h, path.as_ptr()), 0);
        vanedb_capi::vanedb_rs_hnsw_free(h);

        let h2 = vanedb_capi::vanedb_rs_hnsw_load(path.as_ptr());
        assert!(!h2.is_null());
        let mut ids2 = [0u64; 1];
        let mut ds2 = [0.0f32; 1];
        let n2 = vanedb_capi::vanedb_rs_hnsw_search(
            h2,
            q.as_ptr(),
            1,
            50,
            ids2.as_mut_ptr(),
            ds2.as_mut_ptr(),
        );
        assert_eq!(n2, 1);
        assert_eq!(ids2[0], 10);
        vanedb_capi::vanedb_rs_hnsw_free(h2);
        // negative paths
        assert!(vanedb_capi::vanedb_rs_hnsw_new(0, 0, 100, 16, 200, 42).is_null());
        assert_eq!(
            vanedb_capi::vanedb_rs_hnsw_add(std::ptr::null_mut(), 1, v0.as_ptr()),
            1
        );
        assert_eq!(
            vanedb_capi::vanedb_rs_hnsw_search(
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
            vanedb_capi::vanedb_rs_hnsw_save(std::ptr::null_mut(), path.as_ptr()),
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
            vanedb_capi::vanedb_rs_mmap_build(
                path.as_ptr(),
                2,
                0,
                ids_in.as_ptr(),
                vecs.as_ptr(),
                2
            ),
            0
        );
        let m = vanedb_capi::vanedb_rs_mmap_open(path.as_ptr());
        assert!(!m.is_null());
        let mut ids = [0u64; 2];
        let mut ds = [0.0f32; 2];
        let n =
            vanedb_capi::vanedb_rs_mmap_search(m, q.as_ptr(), 2, ids.as_mut_ptr(), ds.as_mut_ptr());
        assert_eq!(n, 2);
        assert_eq!(ids[0], 10);
        vanedb_capi::vanedb_rs_mmap_free(m);
        // negative path
        assert_eq!(
            vanedb_capi::vanedb_rs_mmap_search(
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
